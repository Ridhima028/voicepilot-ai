"""
Automatic study pack: the shared build logic, the worker task's status
transitions, and the API refusing manual generation while it runs.
No LLM, database, Redis or MinIO: all replaced with fakes.
"""

import asyncio
import os
from datetime import datetime
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "sqlite://")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
import tasks  # noqa: E402
from database import get_db  # noqa: E402
from dependencies import get_current_user  # noqa: E402
from models.lecture import Lecture  # noqa: E402
from services import study_pack  # noqa: E402

FIELDS = ["summary", "notes", "key_concepts", "quiz", "flashcards", "mindmap", "chapters"]


def make_lecture(**overrides) -> Lecture:
    lecture = Lecture(
        id=19, user_id=7, title="Lecture", audio_path="audio/7/x.m4a",
        status="completed", transcript="Objects live in buckets.",
        segments=[{"start": 0, "end": 5, "text": "Objects live in buckets."}],
        created_at=datetime.utcnow(),
    )
    for key, value in overrides.items():
        setattr(lecture, key, value)
    return lecture


@pytest.fixture
def fake_generators(monkeypatch):
    """Every generator returns a marker; names listed in .fail raise instead."""
    state = SimpleNamespace(fail=set(), calls=[])

    def fake(name, value):
        async def generator(*args, **kwargs):
            state.calls.append(name)
            if name in state.fail:
                raise RuntimeError(f"{name} exploded")
            return value
        return generator

    for name, value in [
        ("generate_summary", "a summary"), ("generate_notes", "## Notes"),
        ("generate_key_concepts", [{"term": "Bucket"}]), ("generate_quiz", [{"question": "Q?"}]),
        ("generate_flashcards", [{"front": "F"}]), ("generate_mindmap", "mindmap\n  root"),
        ("generate_chapters", [{"start_seconds": 0, "title": "Intro"}]),
    ]:
        monkeypatch.setattr(study_pack, name, fake(name, value))
    return state


# --------------------------------------------------------------------------
# build_study_pack (shared by the endpoint and the worker)
# --------------------------------------------------------------------------

def test_build_fills_every_field(fake_generators):
    lecture = make_lecture()
    result = asyncio.run(study_pack.build_study_pack(lecture))

    assert sorted(result["generated"]) == sorted(FIELDS)
    assert result["failed"] == {}
    assert lecture.summary == "a summary" and lecture.quiz == [{"question": "Q?"}]


def test_build_skips_existing_unless_regenerate(fake_generators):
    lecture = make_lecture(summary="already here")

    result = asyncio.run(study_pack.build_study_pack(lecture))
    assert "summary" in result["skipped"]
    assert lecture.summary == "already here"
    assert "generate_summary" not in fake_generators.calls

    result = asyncio.run(study_pack.build_study_pack(lecture, regenerate=True))
    assert "summary" in result["generated"]
    assert lecture.summary == "a summary"


def test_build_keeps_successes_when_one_fails(fake_generators):
    fake_generators.fail = {"generate_quiz"}
    lecture = make_lecture()

    result = asyncio.run(study_pack.build_study_pack(lecture))

    assert result["failed"] == {"quiz": "generate_quiz exploded"}
    assert len(result["generated"]) == 6
    assert lecture.quiz is None and lecture.notes == "## Notes"


# --------------------------------------------------------------------------
# tasks.generate_study_pack status transitions
# --------------------------------------------------------------------------

class FakeQuery:
    def __init__(self, lecture):
        self.lecture = lecture

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self.lecture


class FakeSession:
    def __init__(self, lecture):
        self.lecture = lecture
        self.commits = 0

    def query(self, model):
        return FakeQuery(self.lecture)

    def commit(self):
        self.commits += 1

    def refresh(self, obj):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


@pytest.fixture
def run_task(monkeypatch):
    def run(lecture):
        session = FakeSession(lecture)
        monkeypatch.setattr(tasks, "SessionLocal", lambda: session)
        tasks.generate_study_pack.run(lecture.id)
        return session
    return run


def test_task_marks_ready(fake_generators, run_task):
    lecture = make_lecture(study_pack_status="queued")
    run_task(lecture)

    assert lecture.study_pack_status == "ready"
    assert lecture.study_pack_error is None
    assert lecture.flashcards == [{"front": "F"}]


def test_task_partial_failure_is_ready_with_error(fake_generators, run_task):
    fake_generators.fail = {"generate_quiz", "generate_notes"}
    lecture = make_lecture(study_pack_status="queued")
    run_task(lecture)

    assert lecture.study_pack_status == "ready"
    assert "quiz: generate_quiz exploded" in lecture.study_pack_error
    assert "notes: generate_notes exploded" in lecture.study_pack_error


def test_task_everything_failed_is_failed(fake_generators, run_task):
    fake_generators.fail = {f"generate_{f}" for f in FIELDS}
    lecture = make_lecture(study_pack_status="queued")
    run_task(lecture)

    assert lecture.study_pack_status == "failed"
    # The transcript itself is untouched.
    assert lecture.status == "completed"


def test_task_skips_already_ready_pack(fake_generators, run_task):
    lecture = make_lecture(study_pack_status="ready")
    session = run_task(lecture)

    assert fake_generators.calls == []   # duplicate delivery does nothing
    assert session.commits == 0


def test_task_skips_unfinished_transcript(fake_generators, run_task):
    lecture = make_lecture(status="processing", study_pack_status="queued")
    run_task(lecture)

    assert fake_generators.calls == []


# --------------------------------------------------------------------------
# API: manual generation refused while the automatic run is active
# --------------------------------------------------------------------------

@pytest.fixture
def api_client():
    def make(lecture):
        main.app.dependency_overrides[get_db] = lambda: FakeSession(lecture)
        main.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=7)
        return TestClient(main.app)
    yield make
    main.app.dependency_overrides.clear()


@pytest.mark.parametrize("pack_status", ["queued", "generating"])
@pytest.mark.parametrize("path", ["quiz", "summary", "key-concepts", "study-pack"])
def test_manual_generation_refused_while_auto_running(api_client, fake_generators, pack_status, path):
    http = api_client(make_lecture(study_pack_status=pack_status))

    res = http.post(f"/lectures/19/{path}")

    assert res.status_code == 409
    assert "automatically" in res.json()["detail"]
    assert fake_generators.calls == []


@pytest.mark.parametrize("pack_status", [None, "ready", "failed"])
def test_manual_generation_allowed_otherwise(api_client, fake_generators, monkeypatch, pack_status):
    from routers import lectures as lectures_router

    async def fake_summary(transcript):
        return "manual summary"

    monkeypatch.setattr(lectures_router, "generate_summary", fake_summary)
    http = api_client(make_lecture(study_pack_status=pack_status))

    res = http.post("/lectures/19/summary")

    assert res.status_code == 200
    assert res.json()["summary"] == "manual summary"
