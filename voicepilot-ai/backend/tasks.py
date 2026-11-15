"""
Celery tasks. Only the worker imports this module — the API enqueues jobs
by name via celery_app.send_task(), so Whisper never loads in the API.

Two entry points share one pipeline:
  tasks.transcribe_lecture  uploaded file already in S3 -> transcribe
  tasks.import_youtube      YouTube link -> download -> S3 -> transcribe

Either one then queues (if AUTO_STUDY_PACK):
  tasks.generate_study_pack summary, notes, quiz, ... via the LLM
"""

import asyncio
import os
import random
import tempfile
import uuid

from sqlalchemy.exc import OperationalError

from celery_app import celery_app
from config import AUTO_STUDY_PACK
from database import SessionLocal
from models.lecture import Lecture
from services import storage, youtube_service
import models.user  # noqa: F401 — registers "users" so Lecture.user_id's ForeignKey resolves

MAX_RETRIES = 3
RETRY_BASE_SECONDS = 10


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

        lecture.study_pack_status = "generating"
        lecture.study_pack_error = None
        db.commit()

        result = asyncio.run(build_study_pack(lecture))

        nothing_saved = not result["generated"] and not result["skipped"]
        lecture.study_pack_status = "failed" if nothing_saved else "ready"
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
        lecture.status = "downloading"
        db.commit()

        # TemporaryDirectory deletes the download and any ffmpeg leftovers
        # when the block exits — on success, failure or retry.
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_file = youtube_service.download_audio(lecture.source_url, tmp_dir)
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

    lecture.status = "processing"
    db.commit()

    # Whisper needs a real file, so pull the object down to a temp file
    # local to THIS worker. Any worker on any machine can do this.
    audio_tmp = storage.download_to_tempfile(lecture.audio_path)
    try:
        result = transcribe_audio(audio_tmp)
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

    lecture.status = "completed"
    if AUTO_STUDY_PACK:
        # Committed together with "completed", so the UI never sees a finished
        # transcript without knowing the study pack is on its way.
        lecture.study_pack_status = "queued"
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
            lecture.status = "failed"
            lecture.error_message = str(e)
            db.commit()

    finally:
        db.close()


def _retry_or_fail(task, db, lecture_id: int, error: Exception, what: str) -> None:
    db.rollback()
    if task.request.retries >= MAX_RETRIES:
        _mark_failed(lecture_id, f"{what} after {MAX_RETRIES} retries: {error}")
        raise error
    raise task.retry(exc=error, countdown=_backoff_with_jitter(task.request.retries))


def _mark_study_pack_failed(lecture_id: int, message: str):
    """Best-effort, like _mark_failed, but leaves the transcript's status alone."""
    db = SessionLocal()
    try:
        lecture = db.query(Lecture).filter(Lecture.id == lecture_id).first()
        if lecture:
            lecture.study_pack_status = "failed"
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
            lecture.status = "failed"
            lecture.error_message = message
            db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()
