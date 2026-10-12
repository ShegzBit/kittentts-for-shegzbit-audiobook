# Windows 11 dev setup (Python 3.13, `uv` online)

Dev-only flow: `uv run python app.py` (Flask threaded server).
`gunicorn.conf.py` is Linux-only — uv skips it on Windows via a
`sys_platform != "win32"` marker, so there is a single dependency set.

## 1. Prerequisites

- Windows 11 x64, Python 3.13 64-bit from python.org ("Add to PATH").
  If `torch` has no 3.13 wheel, fall back to python.org Python 3.11/3.12.
- Install `uv` (e.g. `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`).
- Visual C++ 2019 runtime (required by onnxruntime).
- Optional fallback: espeak-ng `espeak-ng-X64.msi` 1.52.0 (stock install to
  `C:\Program Files\eSpeak NG\`). Usually not needed — `espeakng-loader`
  ships a bundled copy and `phonemizer==3.4.0` auto-finds the stock install.

## 2. Install

```powershell
uv sync --frozen
```

One command installs everything — CPU `torch` (via the PyTorch CPU index
in `pyproject.toml`), KittenTTS (GitHub release `.whl`), and all app deps
— into `.venv`. No separate torch/whl steps, no requirements files.

## 3. espeak-ng

`app.py` runs a Windows-only shim before model load: if
`PHONEMIZER_ESPEAK_LIBRARY` is unset, it points phonemizer at the
`espeakng-loader` bundled library + data. Override only if needed:

```powershell
$env:PHONEMIZER_ESPEAK_LIBRARY = "C:\Program Files\eSpeak NG\libespeak-ng.dll"
uv run python -c "from phonemizer.backend import EspeakBackend; print(EspeakBackend.is_available())"
# expect True
```

## 4. Run

```powershell
$env:GEN_MAX_SLOTS = "2"   # optional; use 2 on low-core laptops (default 4)
uv run python app.py
# GET  http://127.0.0.1:5001/health
# GET  http://127.0.0.1:5001/api/voices
# POST http://127.0.0.1:5001/api/tts  {"text": "hello"}
```

Accept the Firewall prompt, or bind to `127.0.0.1` if locked down.
`Ctrl+C` stops the server. `*.wav` test outputs are already gitignored.
