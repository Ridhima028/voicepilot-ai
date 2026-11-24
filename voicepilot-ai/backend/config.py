import os

# Both configurable via .env — defaults are sane for local dev.
# In production you'd likely set these lower.

MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "200"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

MAX_DURATION_MINUTES = int(os.getenv("MAX_DURATION_MINUTES", "90"))
MAX_DURATION_SECONDS = MAX_DURATION_MINUTES * 60

# Language Whisper transcribes into. "en" by default: Whisper "small" can't
# write Hindi reliably, and on Hinglish lectures auto-detect picks Hindi and
# produces garbled text (and ~3x slower, from repeated re-decoding). Forcing
# English gives clean English text, with Hindi speech rendered in English.
# Set to "auto" to let Whisper detect the language instead.
_whisper_language = os.getenv("WHISPER_LANGUAGE", "en").strip().lower()
WHISPER_LANGUAGE = None if _whisper_language in ("", "auto") else _whisper_language

# Whisper needs an actual audio format it can decode. UploadFile.content_type
# comes from the browser/client and isn't fully trustworthy on its own
# (e.g. browsers recording via MediaRecorder often send "video/webm" or
# "application/octet-stream" even though it's audio-only), so we check file
# extension too as a second signal.
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac"}

# --- Gemini (Google AI Studio) ---
# Free tier, no credit card. Get a key at https://aistudio.google.com
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

# --- OpenRouter / Qwen ---
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen/qwen3-8b")
# Upper bound on each reply (thinking + answer). Without it OpenRouter
# reserves credit for the model's maximum (65k tokens) on every request,
# which fails with 402 on a low balance even though real replies are small.
QWEN_MAX_TOKENS = int(os.getenv("QWEN_MAX_TOKENS", "8000"))
QWEN_EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")

# --- Object storage (S3 API: MinIO locally, AWS S3 in production) ---
# S3_ENDPOINT_URL is how containers reach storage (http://minio:9000).
# Leave it unset for real AWS S3 — boto3 then uses the AWS default endpoint.
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL") or None
# Presigned URLs are signed for a specific host, so they must be generated
# with the host the *browser* will use. Defaults to the internal endpoint
# (correct for AWS, where both are the same).
S3_PUBLIC_ENDPOINT_URL = os.getenv("S3_PUBLIC_ENDPOINT_URL") or S3_ENDPOINT_URL
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "voicepilot-audio")
S3_PRESIGNED_EXPIRY_SECONDS = int(os.getenv("S3_PRESIGNED_EXPIRY_SECONDS", "900"))

# --- YouTube import ---
# Off on cloud deployments: YouTube often blocks downloads from server IPs.
YOUTUBE_IMPORT_ENABLED = os.getenv("YOUTUBE_IMPORT_ENABLED", "true").lower() == "true"

# --- Automatic study pack ---
# Generate all study material in the worker right after transcription, so
# it's usually ready before the student opens the lecture. Set to false to
# only generate on demand (e.g. to control LLM cost).
AUTO_STUDY_PACK = os.getenv("AUTO_STUDY_PACK", "true").lower() == "true"

# --- Self-healing for stuck lectures (tasks.sweep_stuck_jobs) ---
# How often a busy worker writes "still alive" (+ progress) for its lecture.
HEARTBEAT_INTERVAL_SECONDS = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "15"))
# A heartbeat older than this means the worker is presumed dead: another
# delivery may take the job over, and the sweeper re-queues it.
LEASE_SECONDS = int(os.getenv("LEASE_SECONDS", "600"))
# A lecture "queued" longer than this is re-queued (its message was likely
# lost). Keep it above the longest backlog you expect behind busy workers.
QUEUED_STALE_SECONDS = int(os.getenv("QUEUED_STALE_SECONDS", "1800"))
# A study pack "queued"/"generating" longer than this is re-queued once,
# then failed so the manual Generate buttons unlock.
STUDY_PACK_STALE_SECONDS = int(os.getenv("STUDY_PACK_STALE_SECONDS", "900"))
# How many times the sweeper re-queues one lecture before marking it failed.
MAX_REQUEUES = int(os.getenv("MAX_REQUEUES", "2"))
# How often Celery Beat runs the sweeper.
SWEEP_INTERVAL_SECONDS = int(os.getenv("SWEEP_INTERVAL_SECONDS", "300"))
