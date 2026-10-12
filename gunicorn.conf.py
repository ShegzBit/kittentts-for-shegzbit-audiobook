# Production WSGI config: 1 worker process, 1 shared KittenTTS model,
# 4 simultaneous generation slots (see app.py BoundedSemaphore).
#
# workers=1 is deliberate: each gunicorn worker imports app.py and loads its
# OWN model copy (~78 MB weights + runtime). Two workers = two model copies
# (~2x RAM) with no correctness benefit here, since the espeak race is
# already handled in-process. One worker + gthread threads + semaphore gives
# 4 inference slots on 1 model copy.
workers = 1
worker_class = "gthread"
threads = 6  # 4 gen slots + headroom so /health + /api/voices stay responsive
bind = "0.0.0.0:5000"

# Hang protection: semaphore wait is 120 s (GEN_QUEUE_TIMEOUT); kill a worker
# that still hasn't finished well past that.
timeout = 150
graceful_timeout = 30
keepalive = 5

# Bound native-memory growth from long-lived ONNX/espeak state.
max_requests = 200
max_requests_jitter = 20

# Load the model post-fork, once per worker (not pre-fork in the master).
preload_app = False
