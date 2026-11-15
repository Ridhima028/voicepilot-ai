"""
YOUTUBE_IMPORT_ENABLED switch, tested through the real FastAPI app.

The database session and the logged-in user are replaced with fakes, so no
Postgres, Redis, MinIO or YouTube is needed. The fake session records every
add(), which is how "no lecture row was created" is checked.
"""

import os
from datetime import datetime
from types import SimpleNamespace

# database.py builds its engine at import time; give it something valid when
# no backend/.env is present (e.g. in CI). create_engine doesn't connect.
os.environ.setdefault("DATABASE_URL", "sqlite://")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from database import get_db  # noqa: E402
from dependencies import get_current_user  # noqa: E402
from routers import lectures as lectures_router  # noqa: E402

VIDEO_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"


class FakeSession:
    def __init__(self):
        self.added = []
        self.commits = 0

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.commits += 1

    def refresh(self, obj):
        # What Postgres would fill in on INSERT.
        obj.id = 1
        obj.created_at = datetime.utcnow()


@pytest.fixture
def client(monkeypatch):
    session = FakeSession()
    sent_tasks = []

    main.app.dependency_overrides[get_db] = lambda: session
    main.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=7)
    monkeypatch.setattr(
        lectures_router.celery_app, "send_task",
        lambda name, args=None, **kw: sent_tasks.append((name, args)),
    )

    # TestClient without "with" skips the app's lifespan (DB migrations,
    # bucket creation), which these tests don't need.
    yield TestClient(main.app), session, sent_tasks

    main.app.dependency_overrides.clear()


def test_disabled_returns_404_and_creates_nothing(client, monkeypatch):
    http, session, sent_tasks = client
    monkeypatch.setattr(lectures_router, "YOUTUBE_IMPORT_ENABLED", False)

    def must_not_be_called(*args, **kwargs):
        raise AssertionError("YouTube was contacted while the feature is disabled")

    monkeypatch.setattr(lectures_router.youtube_service, "get_video_info", must_not_be_called)

    res = http.post("/lectures/from-youtube", json={"url": VIDEO_URL})

    assert res.status_code == 404
    assert res.json() == {"detail": "YouTube import is not enabled on this server."}
    assert session.added == []      # no lecture row
    assert session.commits == 0
    assert sent_tasks == []         # nothing queued


def test_disabled_check_runs_before_url_validation(client, monkeypatch):
    http, session, _ = client
    monkeypatch.setattr(lectures_router, "YOUTUBE_IMPORT_ENABLED", False)

    # An invalid link would normally be a 400; disabled must win.
    res = http.post("/lectures/from-youtube", json={"url": "https://vimeo.com/123"})

    assert res.status_code == 404
    assert session.added == []


def test_enabled_queues_import(client, monkeypatch):
    http, session, sent_tasks = client
    monkeypatch.setattr(lectures_router, "YOUTUBE_IMPORT_ENABLED", True)
    monkeypatch.setattr(
        lectures_router.youtube_service, "get_video_info",
        lambda url: {"title": "Me at the zoo", "duration": 19},
    )

    res = http.post("/lectures/from-youtube", json={"url": "https://youtu.be/jNQXAC9IVRw"})

    assert res.status_code == 202
    body = res.json()
    assert body["title"] == "Me at the zoo"
    assert body["source_url"] == VIDEO_URL
    assert body["status"] == "queued"
    assert len(session.added) == 1
    assert sent_tasks == [("tasks.import_youtube", [1])]


@pytest.mark.parametrize("enabled", [True, False])
def test_public_config_reports_flag(monkeypatch, enabled):
    monkeypatch.setattr(main, "YOUTUBE_IMPORT_ENABLED", enabled)

    # No auth header: the frontend calls this before login.
    res = TestClient(main.app).get("/config")

    assert res.status_code == 200
    assert res.json() == {"youtube_import": enabled}
