"""
Celery tasks. Only the worker imports this module — the API enqueues jobs
by name via celery_app.send_task(), so Whisper never loads in the API.

Two entry points share one pipeline:
  tasks.transcribe_lecture  uploaded file already in S3 -> transcribe
  tasks.import_youtube      YouTube link -> download -> S3 -> transcribe

Either one then queues (if AUTO_STUDY_PACK):
  tasks.generate_study_pack summary, notes, quiz, ... via the LLM

Self-healing (run by Celery Beat):
  tasks.sweep_stuck_jobs    re-queues or fails lectures whose worker died
"""

import asyncio
import os
import random
import tempfile
import time
import uuid
from datetime import datetime, timedelta

from celery.utils.log import get_task_logger
from sqlalchemy import func, or_, update
from sqlalchemy.exc import OperationalError

from celery_app import celery_app
from config import (
    AUTO_STUDY_PACK, HEARTBEAT_INTERVAL_SECONDS, LEASE_SECONDS, MAX_REQUEUES,
    QUEUED_STALE_SECONDS, STUDY_PACK_STALE_SECONDS,
)
from database import SessionLocal
from models.lecture import Lecture, set_status, set_study_pack_status
from services import storage, youtube_service
import models.user  # noqa: F401 — registers "users" so Lecture.user_id's ForeignKey resolves

logger = get_task_logger(__name__)

MAX_RETRIES = 3
RETRY_BASE_SECONDS = 10

# Statuses in which a worker is actively working on the lecture.
ACTIVE_STATUSES = ("downloading", "processing")
STUDY_PACK_RUNNING = ("queued", "generating")
# The sweeper re-queues a stuck study pack this many times, then fails it.
MAX_STUDY_PACK_REQUEUES = 1

STUCK_MESSAGE = "Processing stopped unexpectedly. Please click Retry."
STUDY_PACK_TIMEOUT_MESSAGE = "Generation timed out. You can generate items manually."


def _backoff_with_jitter(retries: int) -> int:
    """Exponential backoff (10s, 20s, 40s) with full jitter, so several
    failing jobs don't all hammer the DB again at the same instant."""
    return random.randint(0, RETRY_BASE_SECONDS * (2 ** retries))


@celery_app.task(name="tasks.transcribe_lecture", bind=True, max_retries=MAX_RETRIES)
def transcribe_lecture(self, lecture_id: int):
    _run_job(self, lecture_id, _transcribe_stored_audio)


@celery_app.task(name="tasks.import_youtube", bind=True, max_retries=MAX_RETRIES)
def import_youtube(self, lecture_id: int):
    _run_job(self, lecture_id, _import_youtube_audio)


@celery_app.task(name="tasks.generate_study_pack", bind=True, max_retries=MAX_RETRIES)
def generate_study_pack(self, lecture_id: int):
    """
    Generates all study material right after transcription, so it's usually
    ready before the student opens the lecture. A separate task (not part of
    transcription) so an LLM problem can never mark a good transcript failed.
    """
    from services.study_pack import build_study_pack

    db = SessionLocal()
    try:
        lecture = db.query(Lecture).filter(Lecture.id == lecture_id).first()
        if not lecture or lecture.status != "completed" or not lecture.transcript:
            return
        # Idempotency: a redelivered message must not redo a finished pack.
        if lecture.study_pack_status == "ready":
            return

        set_study_pack_status(lecture, "generating")
        lecture.study_pack_error = None
        db.commit()

        result = asyncio.run(build_study_pack(lecture))

        nothing_saved = not result["generated"] and not result["skipped"]
        set_study_pack_status(lecture, "failed" if nothing_saved else "ready")
        lecture.study_pack_error = "; ".join(
            f"{field}: {message}" for field, message in result["failed"].items()
        ) or None
        db.commit()

    except OperationalError as e:
        # Transient: DB connection dropped / restarting.
        db.rollback()
        if self.request.retries >= MAX_RETRIES:
            _mark_study_pack_failed(lecture_id, f"Database unavailable after {MAX_RETRIES} retries: {e}")
            raise e
        raise self.retry(exc=e, countdown=_backoff_with_jitter(self.request.retries))

    except Exception as e:
        # Anything else: record it on the study pack only. The transcript
        # stays "completed"; the student can still generate items by hand.
        db.rollback()
        _mark_study_pack_failed(lecture_id, f"Study pack generation failed: {e}")

    finally:
        db.close()


# --------------------------------------------------------------------------
# Heartbeat
# --------------------------------------------------------------------------

class Heartbeat:
    """
    Throttled "this worker is still alive" signal for one lecture, optionally
    carrying transcription progress. Called often (every Whisper segment,
    every yt-dlp progress tick) but writes at most once per interval.

    Best-effort: a failed write is logged and ignored, never allowed to break
    the transcription it's reporting on.
    """

    def __init__(self, lecture_id: int, interval: float = HEARTBEAT_INTERVAL_SECONDS,
                 clock=time.monotonic, writer=None):
        self.lecture_id = lecture_id
        self.interval = interval
        self._clock = clock
        self._writer = writer or _write_heartbeat
        self._last_write = None

    def beat(self, progress: float | None = None, force: bool = False) -> bool:
        now = self._clock()
        if not force and self._last_write is not None and now - self._last_write < self.interval:
            return False
        self._last_write = now
        try:
            self._writer(self.lecture_id, progress)
        except Exception as e:
            logger.warning("Heartbeat for lecture %s failed: %s", self.lecture_id, e)
        return True


def _write_heartbeat(lecture_id: int, progress: float | None) -> None:
    """Direct UPDATE in its own short session, so it never touches (or
    flushes) the task's main session and its half-built lecture object."""
    values = {"heartbeat_at": datetime.utcnow()}
    if progress is not None:
        values["progress"] = round(progress, 1)
    db = SessionLocal()
    try:
        db.execute(update(Lecture).where(Lecture.id == lecture_id).values(**values))
        db.commit()
    finally:
        db.close()


def _clear_heartbeat(lecture_id: int) -> None:
    """Best-effort. Used before a Celery retry, so the retry isn't mistaken
    for a duplicate by the lease check."""
    db = SessionLocal()
    try:
        db.execute(update(Lecture).where(Lecture.id == lecture_id).values(heartbeat_at=None))
        db.commit()
    except Exception as e:
        db.rollback()
        logger.warning("Could not clear heartbeat for lecture %s: %s", lecture_id, e)
    finally:
        db.close()


def lease_is_held(lecture, now: datetime, lease_seconds: int = LEASE_SECONDS) -> bool:
    """True if a live worker is working on this lecture right now: it's mid-job
    and its heartbeat is fresh. An expired heartbeat means that worker is
    presumed dead and the job may be taken over."""
    return (
        lecture.status in ACTIVE_STATUSES
        and lecture.heartbeat_at is not None
        and lecture.heartbeat_at > now - timedelta(seconds=lease_seconds)
    )


# --------------------------------------------------------------------------
# Job steps
# --------------------------------------------------------------------------

def _import_youtube_audio(db, lecture: Lecture) -> None:
    """Downloads the video's audio into S3 (unless an earlier attempt already
    did), then continues exactly like an upload."""
    if not lecture.source_url:
        raise ValueError("This lecture has no YouTube link to import from.")

    # A retry after a transcription failure shouldn't download again.
    already_stored = bool(lecture.audio_path) and storage.object_exists(lecture.audio_path)

    if not already_stored:
        heartbeat = Heartbeat(lecture.id)
        set_status(lecture, "downloading")
        db.commit()
        heartbeat.beat(force=True)

        # TemporaryDirectory deletes the download and any ffmpeg leftovers
        # when the block exits — on success, failure or retry.
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_file = youtube_service.download_audio(
                lecture.source_url, tmp_dir, on_activity=heartbeat.beat,
            )
            object_key = f"audio/{lecture.user_id}/{uuid.uuid4()}.m4a"
            storage.upload_file(audio_file, object_key, "audio/mp4")

        lecture.audio_path = object_key
        db.commit()

    _transcribe_stored_audio(db, lecture)


def _transcribe_stored_audio(db, lecture: Lecture) -> None:
    """S3 object -> Whisper -> semantic chunks -> saved. Shared by both tasks."""
    # Imported here rather than at module level so the Whisper model loads
    # inside the worker's child process (on the first job), not in the
    # parent before Celery forks it.
    from services.stt_service import transcribe_audio
    from services.semantic_segmenter import semantic_chunk

    heartbeat = Heartbeat(lecture.id)
    set_status(lecture, "processing")
    db.commit()
    heartbeat.beat(progress=0, force=True)

    # Whisper needs a real file, so pull the object down to a temp file
    # local to THIS worker. Any worker on any machine can do this.
    audio_tmp = storage.download_to_tempfile(lecture.audio_path)
    try:
        result = transcribe_audio(audio_tmp, on_progress=heartbeat.beat)
    finally:
        # Always remove the temp copy, or every job would leak one audio
        # file onto the worker's disk.
        try:
            os.unlink(audio_tmp)
        except FileNotFoundError:
            pass

    lecture.transcript = result["text"]
    lecture.segments = result["segments"]
    lecture.language = result["language"]
    lecture.duration_seconds = result.get("duration")

    # Semantic chunking calls an external API, so it gets its own guard:
    # if it fails we still keep a perfectly usable raw transcript rather
    # than marking the whole lecture as failed.
    try:
        lecture.processed_segments = semantic_chunk(result["segments"])
    except Exception as e:
        lecture.processed_segments = None
        lecture.error_message = f"Semantic segmentation skipped: {e}"

    set_status(lecture, "completed")  # also clears heartbeat_at
    lecture.progress = 100
    if AUTO_STUDY_PACK:
        # Committed together with "completed", so the UI never sees a finished
        # transcript without knowing the study pack is on its way.
        set_study_pack_status(lecture, "queued")
    db.commit()

    if AUTO_STUDY_PACK:
        celery_app.send_task("tasks.generate_study_pack", args=[lecture.id])


# --------------------------------------------------------------------------
# Shared session, idempotency and retry handling
# --------------------------------------------------------------------------

def _run_job(task, lecture_id: int, step) -> None:
    db = SessionLocal()
    try:
        lecture = db.query(Lecture).filter(Lecture.id == lecture_id).first()

        # Idempotency guard: with acks_late a message can be delivered more
        # than once, so a duplicate must not redo finished work.
        if not lecture:
            return
        if lecture.status == "completed":
            return
        # Duplicate protection: a redelivered or re-queued message must not
        # start a second copy of a job a live worker is still running.
        if lease_is_held(lecture, datetime.utcnow()):
            logger.info(
                "Lecture %s is %s with a fresh heartbeat (%s); another worker owns it, skipping.",
                lecture.id, lecture.status, lecture.heartbeat_at,
            )
            return

        step(db, lecture)

    except OperationalError as e:
        # Transient: DB connection dropped / restarting. Worth retrying.
        _retry_or_fail(task, db, lecture_id, e, "Database unavailable")

    except storage.TRANSIENT_ERRORS as e:
        # Transient: object storage unreachable (restarting, network blip).
        _retry_or_fail(task, db, lecture_id, e, "File storage unavailable")

    except youtube_service.YouTubeTransientError as e:
        # Transient: couldn't reach YouTube (timeout, DNS, 5xx).
        _retry_or_fail(task, db, lecture_id, e, "YouTube unreachable")

    except Exception as e:
        # Permanent: YouTubeError (private, removed, blocked...),
        # AudioTooLongError, undecodable audio, AudioObjectNotFound...
        # Retrying would fail the same way, so record it and stop.
        db.rollback()
        lecture = db.query(Lecture).filter(Lecture.id == lecture_id).first()
        if lecture:
            set_status(lecture, "failed")
            lecture.error_message = str(e)
            db.commit()

    finally:
        db.close()


def _retry_or_fail(task, db, lecture_id: int, error: Exception, what: str) -> None:
    db.rollback()
    if task.request.retries >= MAX_RETRIES:
        _mark_failed(lecture_id, f"{what} after {MAX_RETRIES} retries: {error}")
        raise error
    # Our own retry must not look like a duplicate to the lease check.
    _clear_heartbeat(lecture_id)
    raise task.retry(exc=error, countdown=_backoff_with_jitter(task.request.retries))


def _mark_study_pack_failed(lecture_id: int, message: str):
    """Best-effort, like _mark_failed, but leaves the transcript's status alone."""
    db = SessionLocal()
    try:
        lecture = db.query(Lecture).filter(Lecture.id == lecture_id).first()
        if lecture:
            set_study_pack_status(lecture, "failed")
            lecture.study_pack_error = message
            db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def _mark_failed(lecture_id: int, message: str):
    """Best-effort: the DB may still be down, in which case this fails too."""
    db = SessionLocal()
    try:
        lecture = db.query(Lecture).filter(Lecture.id == lecture_id).first()
        if lecture:
            set_status(lecture, "failed")
            lecture.error_message = message
            db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


# --------------------------------------------------------------------------
# Sweeper: self-healing for lectures whose worker died
# --------------------------------------------------------------------------

def _is_older_than(timestamp: datetime | None, now: datetime, seconds: int) -> bool:
    return timestamp is not None and timestamp < now - timedelta(seconds=seconds)


def decide_action(lecture, now: datetime, audio_exists=None) -> str | None:
    """
    Pure decision for one lecture: what, if anything, the sweeper should do.
    Returns "requeue_transcription", "requeue_youtube", "fail_transcription",
    "requeue_study_pack", "fail_study_pack" or None.

    "Older than" is strict: exactly at the threshold is still healthy.
    Rows from before these columns existed have null timestamps, so
    created_at stands in for them.

    audio_exists(key) -> bool is only used to choose between re-downloading
    a YouTube lecture and re-transcribing it; when not given, a non-empty
    audio_path is assumed to be in storage.
    """
    status = lecture.status

    if status == "queued":
        since = lecture.status_updated_at or lecture.created_at
        if _is_older_than(since, now, QUEUED_STALE_SECONDS):
            return _requeue_or_fail(lecture, audio_exists)
        return None

    if status in ACTIVE_STATUSES:
        # A healthy slow job keeps its heartbeat fresh and is never touched.
        last_alive = lecture.heartbeat_at or lecture.status_updated_at or lecture.created_at
        if _is_older_than(last_alive, now, LEASE_SECONDS):
            return _requeue_or_fail(lecture, audio_exists)
        return None

    if status == "completed" and lecture.study_pack_status in STUDY_PACK_RUNNING:
        since = lecture.study_pack_updated_at or lecture.created_at
        if _is_older_than(since, now, STUDY_PACK_STALE_SECONDS):
            if (lecture.study_pack_requeue_count or 0) < MAX_STUDY_PACK_REQUEUES:
                return "requeue_study_pack"
            return "fail_study_pack"

    return None


def _requeue_or_fail(lecture, audio_exists) -> str:
    if (lecture.requeue_count or 0) >= MAX_REQUEUES:
        return "fail_transcription"
    return "requeue_youtube" if _needs_youtube_download(lecture, audio_exists) else "requeue_transcription"


def _needs_youtube_download(lecture, audio_exists) -> bool:
    if not lecture.source_url:
        return False
    if not lecture.audio_path:
        return True
    return audio_exists is not None and not audio_exists(lecture.audio_path)


def _transcription_stale_clause(seen_status: str, now: datetime):
    """SQL version of decide_action's staleness rule, for compare-and-set."""
    if seen_status == "queued":
        since = func.coalesce(Lecture.status_updated_at, Lecture.created_at)
        return since < now - timedelta(seconds=QUEUED_STALE_SECONDS)
    last_alive = func.coalesce(Lecture.heartbeat_at, Lecture.status_updated_at, Lecture.created_at)
    return last_alive < now - timedelta(seconds=LEASE_SECONDS)


def _apply_action(db, lecture, action: str, now: datetime) -> bool:
    """
    Applies one sweeper decision as a compare-and-set UPDATE: it only changes
    the row if it's still in the state the decision was based on (same
    status, still stale). If a worker revived it or the student retried in
    the meantime, 0 rows match and nothing happens. The Celery task is sent
    only after the commit, and only if exactly 1 row changed.
    """
    task_name = None
    seen = Lecture.id == lecture.id

    if action in ("requeue_transcription", "requeue_youtube"):
        stmt = update(Lecture).where(
            seen,
            Lecture.status == lecture.status,
            _transcription_stale_clause(lecture.status, now),
            func.coalesce(Lecture.requeue_count, 0) < MAX_REQUEUES,
        ).values(
            status="queued", status_updated_at=now,
            requeue_count=func.coalesce(Lecture.requeue_count, 0) + 1,
            heartbeat_at=None, progress=None,
        )
        task_name = "tasks.import_youtube" if action == "requeue_youtube" else "tasks.transcribe_lecture"

    elif action == "fail_transcription":
        stmt = update(Lecture).where(
            seen,
            Lecture.status == lecture.status,
            _transcription_stale_clause(lecture.status, now),
        ).values(
            status="failed", status_updated_at=now, heartbeat_at=None,
            error_message=STUCK_MESSAGE,
        )

    elif action in ("requeue_study_pack", "fail_study_pack"):
        pack_stale = func.coalesce(Lecture.study_pack_updated_at, Lecture.created_at) \
            < now - timedelta(seconds=STUDY_PACK_STALE_SECONDS)
        conditions = [seen, Lecture.study_pack_status == lecture.study_pack_status, pack_stale]
        if action == "requeue_study_pack":
            conditions.append(func.coalesce(Lecture.study_pack_requeue_count, 0) < MAX_STUDY_PACK_REQUEUES)
            values = dict(
                study_pack_status="queued", study_pack_updated_at=now,
                study_pack_requeue_count=func.coalesce(Lecture.study_pack_requeue_count, 0) + 1,
            )
            task_name = "tasks.generate_study_pack"
        else:
            values = dict(
                study_pack_status="failed", study_pack_updated_at=now,
                study_pack_error=STUDY_PACK_TIMEOUT_MESSAGE,
            )
        stmt = update(Lecture).where(*conditions).values(**values)

    else:
        return False

    changed = db.execute(stmt.execution_options(synchronize_session=False)).rowcount
    db.commit()  # commit BEFORE send_task, same rule as everywhere else
    if changed != 1:
        return False
    if task_name:
        celery_app.send_task(task_name, args=[lecture.id])
    return True


@celery_app.task(name="tasks.sweep_stuck_jobs")
def sweep_stuck_jobs():
    """Finds lectures stuck because a worker died (or a message was lost) and
    re-queues them, or fails them once they've been re-queued too often."""
    db = SessionLocal()
    counts = {"requeued": 0, "failed": 0, "checked": 0}
    try:
        candidates = (
            db.query(Lecture)
            .filter(or_(
                Lecture.status.in_(("queued",) + ACTIVE_STATUSES),
                Lecture.study_pack_status.in_(STUDY_PACK_RUNNING),
            ))
            .order_by(Lecture.id)
            .limit(100)
            .all()
        )
        now = datetime.utcnow()

        for lecture in candidates:
            counts["checked"] += 1
            try:
                action = decide_action(lecture, now, audio_exists=storage.object_exists)
                if action and _apply_action(db, lecture, action, now):
                    counts["requeued" if action.startswith("requeue") else "failed"] += 1
                    logger.info("Sweeper: lecture %s -> %s", lecture.id, action)
            except storage.TRANSIENT_ERRORS as e:
                # Storage down: can't tell upload from YouTube re-download yet.
                db.rollback()
                logger.warning("Sweeper: skipped lecture %s, storage unavailable: %s", lecture.id, e)

        logger.info(
            "Sweeper run: checked %d, requeued %d, failed %d",
            counts["checked"], counts["requeued"], counts["failed"],
        )
        return counts
    finally:
        db.close()
