"""Minimal Flask TTS API backed by KittenTTS.

POST /api/tts {text, voice?, speed?} -> audio/wav bytes
GET  /api/voices -> available voices
GET  /health -> liveness check

Concurrency model: single worker process, ONE shared model, FOUR
simultaneous generation slots, chunk-parallel synthesis. See gunicorn.conf.py.
"""

import os

# Four simultaneous inference slots inside this one worker process.
GEN_MAX_SLOTS = int(os.environ.get("GEN_MAX_SLOTS", "4"))
GEN_QUEUE_TIMEOUT = float(os.environ.get("GEN_QUEUE_TIMEOUT", "120"))

# Per-request chunk parallelism for long texts (chunk_text -> N chunks).
# Chunks are independent: phonemize is serialized by _ESPEAK_LOCK (~4ms each),
# session.run() is thread-safe and GIL-releasing, results concatenated in order.
CHUNK_WORKERS = max(1, int(os.environ.get("CHUNK_WORKERS", "4")))


def _default_intra_op_threads() -> int:
    # One ORT intra-op pool per concurrent generation slot.
    # Benchmarked on 8 vCPU: intra=4 -> ~1.97s/run, default -> ~2.60s/run,
    # intra=8 -> ~3.14s/run (oversubscribed). So cores // slots.
    try:
        cpu = os.cpu_count() or 4
    except NotImplementedError:
        cpu = 4
    return max(1, cpu // max(1, GEN_MAX_SLOTS))


ORT_INTRA_OP_NUM_THREADS = int(
    os.environ.get("ORT_INTRA_OP_NUM_THREADS", str(_default_intra_op_threads()))
)
ORT_INTER_OP_NUM_THREADS = int(os.environ.get("ORT_INTER_OP_NUM_THREADS", "1"))

# Cap native thread pools BEFORE onnxruntime loads. KittenTTS does not
# forward onnx SessionOptions, so env vars are the no-fork way to avoid
# oversubscription (slots x chunk-workers x intra-op threads). Overridden by real env if set.
os.environ.setdefault("OMP_NUM_THREADS", str(ORT_INTRA_OP_NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(ORT_INTRA_OP_NUM_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(ORT_INTRA_OP_NUM_THREADS))

import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
from flask import Flask, Response, jsonify, request

import onnxruntime as ort
from kittentts import KittenTTS
from kittentts.onnx_model import chunk_text

# FP32 nano is the speed default. The official int8 repo
# (KittenML/kitten-tts-nano-0.8-int8, 24MB vs 54MB) benchmarked ~4-7x
# SLOWER on this CPU (dynamic-quant kernels, no VNNI): 2-chunk sample
# FP32 intra=4 -> 2.2s, int8 default -> 8.6s, int8 intra=4 -> 15.7s.
# Keep it opt-in via KITTEN_MODEL for memory-constrained edge deploys only.
MODEL_NAME = os.environ.get("KITTEN_MODEL", "KittenML/kitten-tts-nano-0.8")
SAMPLE_RATE = 24000
DEFAULT_VOICE = "Bella"
DEFAULT_SPEED = 1.0
MIN_SPEED = 0.5
MAX_SPEED = 2.0
MAX_TEXT_CHARS = 5000

app = Flask(__name__)

# KittenTTS builds ort.InferenceSession(model_path) with defaults and offers
# no SessionOptions hook, so patch it process-wide before model load.
_orig_InferenceSession = ort.InferenceSession


def _tuned_InferenceSession(path, *args, **kwargs):
    if kwargs.get("sess_options") is None:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = ORT_INTRA_OP_NUM_THREADS
        so.inter_op_num_threads = ORT_INTER_OP_NUM_THREADS
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        kwargs["sess_options"] = so
    kwargs.setdefault("providers", ["CPUExecutionProvider"])
    return _orig_InferenceSession(path, *args, **kwargs)


ort.InferenceSession = _tuned_InferenceSession

# Windows: point phonemizer at a usable espeak-ng library before KittenTTS
# builds its EspeakBackend. Prefers an explicit user override, then the
# espeakng-loader bundled copy, then the stock espeak-ng .msi install
# (phonemizer>=3.4 auto-finds C:\Program Files\eSpeak NG\libespeak-ng.dll).
# No-op on Linux/macOS.
if os.name == "nt" and not os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
    try:
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    except Exception as e:  # noqa: BLE001 - fall back to system espeak-ng
        print(f"[espeak] bundled loader unavailable ({e}), using system espeak-ng")

print(
    f"Loading KittenTTS model ({MODEL_NAME})... "
    f"[ORT intra={ORT_INTRA_OP_NUM_THREADS} inter={ORT_INTER_OP_NUM_THREADS} "
    f"opt=ALL exec=SEQUENTIAL slots={GEN_MAX_SLOTS}]"
)
model = KittenTTS(MODEL_NAME)
print("Available voices:", model.available_voices)

# --- Concurrency guards (per worker process) ---
# Semaphore: at most GEN_MAX_SLOTS requests inside generation at once.
# A further concurrent request waits up to GEN_QUEUE_TIMEOUT, then gets 503.
_GEN_SLOTS = threading.BoundedSemaphore(GEN_MAX_SLOTS)

# Lock: EspeakBackend wraps libespeak-ng via ctypes with mutable global
# voice state (set_voice_by_name), so concurrent phonemize() calls on the
# shared instance can race. Serialize ONLY phonemization; the ONNX
# session.run() calls (GIL-releasing, thread-safe) still overlap.
_ESPEAK_LOCK = threading.Lock()
_orig_phonemize = model.model.phonemizer.phonemize


def _locked_phonemize(*args, **kwargs):
    with _ESPEAK_LOCK:
        return _orig_phonemize(*args, **kwargs)


model.model.phonemizer.phonemize = _locked_phonemize


def generate_parallel(text: str, voice: str, speed: float):
    """Synthesize long text with chunk-level parallelism.

    Mirrors KittenTTS.generate() with clean_text=False (the wrapper default
    used by this endpoint, so no TextPreprocessor pass), but runs
    generate_single_chunk() concurrently. Chunk order is preserved.
    Note: sample-level bit-equality across runs is not guaranteed even
    sequentially (multithreaded ORT kernels are nondeterministic); parallel
    output differs only within that same run-to-run variance.
    """
    inner = model.model  # KittenTTS_1_Onnx
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No text chunks to synthesize.")
    if len(chunks) == 1 or CHUNK_WORKERS == 1:
        return inner.generate_single_chunk(chunks[0], voice, speed)
    workers = max(1, min(CHUNK_WORKERS, len(chunks)))
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="kitten-chunk"
    ) as ex:
        audios = list(
            ex.map(lambda c: inner.generate_single_chunk(c, voice, speed), chunks)
        )
    return np.concatenate(audios, axis=-1)


@app.get("/health")
def health():
    # Lock-free on purpose: liveness must answer even when all slots busy.
    return jsonify({"status": "ok"})


@app.get("/api/voices")
def voices():
    return jsonify(
        {
            "voices": model.available_voices,
            "default": DEFAULT_VOICE,
            "sample_rate": SAMPLE_RATE,
        }
    )


@app.post("/api/tts")
def tts():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be JSON like {'text': '...'}."}), 400

    text = data.get("text", "")
    voice = data.get("voice", DEFAULT_VOICE)
    speed = data.get("speed", DEFAULT_SPEED)

    if not isinstance(text, str) or not text.strip():
        return (
            jsonify({"error": "Field 'text' is required and must be a non-empty string."}),
            400,
        )
    if len(text) > MAX_TEXT_CHARS:
        return (
            jsonify({"error": f"Field 'text' is too long (max {MAX_TEXT_CHARS} chars)."}),
            400,
        )
    if not isinstance(voice, str) or not voice:
        return jsonify({"error": "Field 'voice' must be a non-empty string."}), 400
    try:
        speed = float(speed)
    except (TypeError, ValueError):
        return jsonify({"error": "Field 'speed' must be a number."}), 400
    if not (MIN_SPEED <= speed <= MAX_SPEED):
        return (
            jsonify({"error": f"Field 'speed' must be between {MIN_SPEED} and {MAX_SPEED}."}),
            400,
        )

    # Validation done: now take a generation slot (bounded wait).
    if not _GEN_SLOTS.acquire(timeout=GEN_QUEUE_TIMEOUT):
        resp = jsonify(
            {
                "error": f"Server busy: all {GEN_MAX_SLOTS} generation slots occupied. Retry shortly."
            }
        )
        resp.headers["Retry-After"] = "5"
        return resp, 503

    start = time.monotonic()
    print(f"[tts] slot acquired voice={voice} chars={len(text)}", flush=True)
    try:
        try:
            audio = generate_parallel(text, voice=voice, speed=speed)
        except ValueError as e:
            # e.g. unknown voice from onnx_model._prepare_inputs
            return jsonify({"error": str(e)}), 400
        except Exception as e:  # noqa: BLE001 - surface as 500 JSON
            return jsonify({"error": f"Synthesis failed: {e}"}), 500

        if audio is None or len(audio) == 0:
            return jsonify({"error": "Synthesis produced no audio."}), 500

        # Per-request buffer: never shared, never a temp file, so concurrent
        # responses cannot overwrite each other.
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV")
        buf.seek(0)
        return Response(buf.getvalue(), mimetype="audio/wav")
    finally:
        _GEN_SLOTS.release()
        print(f"[tts] slot released in {time.monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    # Dev only. Production: gunicorn -c gunicorn.conf.py app:app
    app.run(host="0.0.0.0", port=5001, threaded=True)
