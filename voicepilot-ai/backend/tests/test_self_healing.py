"""
Self-healing for stuck lectures: the sweeper's pure decision logic, the
lease check, the heartbeat throttle and Whisper progress reporting.
No database, Redis, storage or Whisper model: all pure or faked.
"""

import os
import sys
import types
from datetime import datetime, timedelta
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "sqlite://")

import pytest  # noqa: E402

import tasks  # noqa: E402
from tasks import Heartbeat, decide_action, lease_is_held  # noqa: E402

NOW = datetime(2026, 11, 24, 12, 0, 0)
LEASE = timedelta(seconds=tasks.LEASE_SECONDS)
QUEUED_STALE = timedelta(seconds=tasks.QUEUED_STALE_SECONDS)
PACK_STALE = timedelta(seconds=tasks.STUDY_PACK_STALE_SECONDS)
SECOND = timedelta(seconds=1)


def lecture(**overrides):
    """A lecture-shaped object with healthy defaults; override what matters."""
    fields = dict(
        id=1, status="processing", created_at=NOW - timedelta(hours=1),
        status_updated_at=NOW, heartbeat_at=NOW, requeue_count=None,
        source_url=None, audio_path="audio/7/abc.wav",
        study_pack_status=None, study_pack_updated_at=None, study_pack_requeue_count=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


# --------------------------------------------------------------------------
# decide_action: queued
# --------------------------------------------------------------------------

def test_queued_exactly_at_threshold_is_left_alone():
    lec = lecture(status="queued", heartbeat_at=None, status_updated_at=NOW - QUEUED_STALE)
    assert decide_action(lec, NOW) is None


def test_queued_just_past_threshold_is_requeued():
    lec = lecture(status="queued", heartbeat_at=None, status_updated_at=NOW - QUEUED_STALE - SECOND)
    assert decide_action(lec, NOW) == "requeue_transcription"


def test_recently_queued_is_left_alone():
    lec = lecture(status="queued", heartbeat_at=None, status_updated_at=NOW - timedelta(minutes=5))
    assert decide_action(lec, NOW) is None


def test_queued_row_without_timestamp_falls_back_to_created_at():
    # Rows created before status_updated_at existed.
    stale = lecture(status="queued", status_updated_at=None, created_at=NOW - QUEUED_STALE - SECOND)
    fresh = lecture(status="queued", status_updated_at=None, created_at=NOW - timedelta(minutes=1))
    assert decide_action(stale, NOW) == "requeue_transcription"
    assert decide_action(fresh, NOW) is None


# --------------------------------------------------------------------------
# decide_action: downloading / processing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("status", ["downloading", "processing"])
def test_heartbeat_exactly_at_lease_is_left_alone(status):
    lec = lecture(status=status, heartbeat_at=NOW - LEASE)
    assert decide_action(lec, NOW) is None


@pytest.mark.parametrize("status", ["downloading", "processing"])
def test_heartbeat_just_past_lease_is_requeued(status):
    lec = lecture(status=status, heartbeat_at=NOW - LEASE - SECOND)
    assert decide_action(lec, NOW) == "requeue_transcription"


def test_healthy_slow_job_is_never_touched():
    # Running for 3 hours, but the worker heartbeat 10 seconds ago.
    lec = lecture(
        status="processing",
        status_updated_at=NOW - timedelta(hours=3),
        heartbeat_at=NOW - timedelta(seconds=10),
    )
    assert decide_action(lec, NOW) is None


def test_no_heartbeat_uses_status_updated_at():
    # Worker died before its first heartbeat.
    stale = lecture(status="processing", heartbeat_at=None, status_updated_at=NOW - LEASE - SECOND)
    fresh = lecture(status="processing", heartbeat_at=None, status_updated_at=NOW - timedelta(seconds=30))
    assert decide_action(stale, NOW) == "requeue_transcription"
    assert decide_action(fresh, NOW) is None


def test_requeue_limit_reached_fails():
    lec = lecture(heartbeat_at=NOW - LEASE - SECOND, requeue_count=tasks.MAX_REQUEUES)
    assert decide_action(lec, NOW) == "fail_transcription"


def test_one_below_requeue_limit_still_requeues():
    lec = lecture(heartbeat_at=NOW - LEASE - SECOND, requeue_count=tasks.MAX_REQUEUES - 1)
    assert decide_action(lec, NOW) == "requeue_transcription"


def test_queued_requeue_limit_reached_fails():
    lec = lecture(status="queued", heartbeat_at=None,
                  status_updated_at=NOW - QUEUED_STALE - SECOND, requeue_count=tasks.MAX_REQUEUES)
    assert decide_action(lec, NOW) == "fail_transcription"


# --------------------------------------------------------------------------
# decide_action: which task to requeue
# --------------------------------------------------------------------------

STUCK = dict(status="processing", heartbeat_at=NOW - LEASE - SECOND)


def test_upload_requeues_transcription():
    assert decide_action(lecture(**STUCK), NOW) == "requeue_transcription"


def test_youtube_without_audio_requeues_youtube_import():
    lec = lecture(**STUCK, source_url="https://www.youtube.com/watch?v=jNQXAC9IVRw", audio_path="")
    assert decide_action(lec, NOW) == "requeue_youtube"


def test_youtube_with_missing_object_requeues_youtube_import():
    lec = lecture(**STUCK, source_url="https://www.youtube.com/watch?v=jNQXAC9IVRw")
    assert decide_action(lec, NOW, audio_exists=lambda key: False) == "requeue_youtube"


def test_youtube_with_stored_audio_requeues_transcription_only():
    lec = lecture(**STUCK, source_url="https://www.youtube.com/watch?v=jNQXAC9IVRw")
    assert decide_action(lec, NOW, audio_exists=lambda key: True) == "requeue_transcription"


def test_storage_is_not_checked_for_uploads():
    def must_not_be_called(key):
        raise AssertionError("storage checked for a non-YouTube lecture")

    assert decide_action(lecture(**STUCK), NOW, audio_exists=must_not_be_called) == "requeue_transcription"


# --------------------------------------------------------------------------
# decide_action: study pack
# --------------------------------------------------------------------------

@pytest.mark.parametrize("pack_status", ["queued", "generating"])
def test_study_pack_exactly_at_threshold_is_left_alone(pack_status):
    lec = lecture(status="completed", heartbeat_at=None, study_pack_status=pack_status,
                  study_pack_updated_at=NOW - PACK_STALE)
    assert decide_action(lec, NOW) is None


@pytest.mark.parametrize("pack_status", ["queued", "generating"])
def test_stale_study_pack_is_requeued_once(pack_status):
    lec = lecture(status="completed", heartbeat_at=None, study_pack_status=pack_status,
                  study_pack_updated_at=NOW - PACK_STALE - SECOND)
    assert decide_action(lec, NOW) == "requeue_study_pack"


def test_stale_study_pack_already_requeued_fails():
    lec = lecture(status="completed", heartbeat_at=None, study_pack_status="generating",
                  study_pack_updated_at=NOW - PACK_STALE - SECOND, study_pack_requeue_count=1)
    assert decide_action(lec, NOW) == "fail_study_pack"


@pytest.mark.parametrize("pack_status", [None, "ready", "failed"])
def test_finished_or_absent_study_pack_is_left_alone(pack_status):
    lec = lecture(status="completed", heartbeat_at=None, study_pack_status=pack_status,
                  study_pack_updated_at=NOW - timedelta(days=1))
    assert decide_action(lec, NOW) is None


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_finished_lectures_are_left_alone(status):
    lec = lecture(status=status, heartbeat_at=None, status_updated_at=NOW - timedelta(days=3))
    assert decide_action(lec, NOW) is None


# --------------------------------------------------------------------------
# lease_is_held
# --------------------------------------------------------------------------

@pytest.mark.parametrize("status", ["downloading", "processing"])
def test_fresh_heartbeat_holds_lease(status):
    assert lease_is_held(lecture(status=status, heartbeat_at=NOW - timedelta(seconds=20)), NOW) is True


def test_expired_heartbeat_does_not_hold_lease():
    assert lease_is_held(lecture(heartbeat_at=NOW - LEASE - SECOND), NOW) is False


def test_lease_boundary_is_expired():
    # Exactly LEASE_SECONDS old: expired (the sweeper still waits one more
    # second before acting, so the two never disagree on a live job).
    assert lease_is_held(lecture(heartbeat_at=NOW - LEASE), NOW) is False


def test_no_heartbeat_does_not_hold_lease():
    assert lease_is_held(lecture(heartbeat_at=None), NOW) is False


@pytest.mark.parametrize("status", ["queued", "completed", "failed"])
def test_lease_only_applies_while_working(status):
    assert lease_is_held(lecture(status=status, heartbeat_at=NOW), NOW) is False


# --------------------------------------------------------------------------
# Heartbeat throttle
# --------------------------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make_heartbeat(interval=15):
    writes = []
    clock = FakeClock()
    hb = Heartbeat(42, interval=interval, clock=clock, writer=lambda lid, p: writes.append((lid, p)))
    return hb, clock, writes


def test_ten_calls_within_interval_write_once():
    hb, clock, writes = make_heartbeat()
    for i in range(10):
        hb.beat(progress=i * 5)
        clock.t += 1  # 10 calls spread over 10 s < 15 s interval
    assert writes == [(42, 0)]


def test_writes_again_after_interval():
    hb, clock, writes = make_heartbeat()
    hb.beat(progress=10)
    clock.t += 15
    hb.beat(progress=60)
    assert writes == [(42, 10), (42, 60)]


def test_force_bypasses_throttle():
    hb, clock, writes = make_heartbeat()
    hb.beat(progress=0)
    hb.beat(force=True)
    assert len(writes) == 2


def test_failed_write_never_breaks_the_job():
    def broken_writer(lecture_id, progress):
        raise RuntimeError("database down")

    hb = Heartbeat(1, interval=15, clock=FakeClock(), writer=broken_writer)
    assert hb.beat(progress=50) is True  # logged, not raised


# --------------------------------------------------------------------------
# Whisper progress (model faked; the real one is never loaded)
# --------------------------------------------------------------------------

class FakeWhisperModel:
    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, audio_path, **kwargs):
        ends = [2.5, 5.0, 7.5, 10.0, 10.4]  # last segment runs slightly past duration
        segments = (
            SimpleNamespace(start=end - 2.5, end=end, text=f" part {i} ")
            for i, end in enumerate(ends)
        )
        info = SimpleNamespace(duration=10.0, language="en", language_probability=0.99)
        return segments, info


@pytest.fixture
def stt(monkeypatch):
    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    monkeypatch.delitem(sys.modules, "services.stt_service", raising=False)
    import services.stt_service as module
    return module


def test_transcribe_reports_increasing_progress_capped_at_100(stt):
    seen = []
    result = stt.transcribe_audio("lecture.wav", on_progress=seen.append)

    assert seen == sorted(seen)
    assert seen[0] == pytest.approx(25.0)
    assert seen[-1] == 100.0          # 104% capped
    assert all(0 < p <= 100 for p in seen)
    assert len(result["segments"]) == 5


def test_transcribe_without_callback_still_works(stt):
    result = stt.transcribe_audio("lecture.wav")
    assert result["text"].startswith("part 0")
