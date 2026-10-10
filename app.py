"""Minimal Flask TTS API backed by KittenTTS.

POST /api/tts {text, voice?, speed?} -> audio/wav bytes
GET  /api/voices -> available voices
GET  /health -> liveness check

Concurrency model: single worker process, ONE shared model, TWO
simultaneous generation slots. See gunicorn.conf.py.
"""

import os

# Cap native thread pools BEFORE onnxruntime loads. KittenTTS does not
# forward onnx SessionOptions, so env vars are the no-fork way to avoid
# oversubscription (2 slots x N threads). Overridden by real env if set.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import io
import threading
import time

import soundfile as sf
from flask import Flask, Response, jsonify, request

from kittentts import KittenTTS

MODEL_NAME = "KittenML/kitten-tts-mini-0.8"
SAMPLE_RATE = 24000
DEFAULT_VOICE = "Bella"
DEFAULT_SPEED = 1.0
MIN_SPEED = 0.5
MAX_SPEED = 2.0
MAX_TEXT_CHARS = 5000

# Two simultaneous inference slots inside this one worker process.
GEN_MAX_SLOTS = int(os.environ.get("GEN_MAX_SLOTS", "2"))
GEN_QUEUE_TIMEOUT = float(os.environ.get("GEN_QUEUE_TIMEOUT", "120"))

app = Flask(__name__)

print(f"Loading KittenTTS model ({MODEL_NAME})...")
model = KittenTTS(MODEL_NAME)
print("Available voices:", model.available_voices)

# --- Concurrency guards (per worker process) ---
# Semaphore: at most GEN_MAX_SLOTS requests inside generation at once.
# A 3rd concurrent request waits up to GEN_QUEUE_TIMEOUT, then gets 503.
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


@app.get("/health")
def health():
    # Lock-free on purpose: liveness must answer even when both slots busy.
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
            {"error": "Server busy: both generation slots occupied. Retry shortly."}
        )
        resp.headers["Retry-After"] = "5"
        return resp, 503

    start = time.monotonic()
    print(f"[tts] slot acquired voice={voice} chars={len(text)}", flush=True)
    try:
        try:
            audio = model.generate(text, voice=voice, speed=speed)
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
