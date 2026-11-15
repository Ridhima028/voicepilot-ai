"""
Tests for services/youtube_service.py. No network: yt_dlp.YoutubeDL is
replaced with a fake for the metadata/download tests.
"""

import pytest
from yt_dlp.utils import DownloadError

from services import youtube_service
from services.youtube_service import (
    YouTubeError,
    YouTubeTransientError,
    get_video_info,
    validate_youtube_url,
)

VIDEO_ID = "dQw4w9WgXcQ"
CANONICAL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


# --------------------------------------------------------------------------
# validate_youtube_url
# --------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    f"https://www.youtube.com/watch?v={VIDEO_ID}",
    f"https://youtube.com/watch?v={VIDEO_ID}",
    f"http://www.youtube.com/watch?v={VIDEO_ID}",
    f"https://www.youtube.com/shorts/{VIDEO_ID}",
    f"https://www.youtube.com/live/{VIDEO_ID}",
    f"https://youtu.be/{VIDEO_ID}",
    f"https://m.youtube.com/watch?v={VIDEO_ID}",
    f"https://music.youtube.com/watch?v={VIDEO_ID}",
    f"https://www.youtube.com/watch?v={VIDEO_ID}&t=30s",
    f"https://www.youtube.com/watch?v={VIDEO_ID}&list=PLx0sYbCqOb8TBPRdmBHs5Iftvv9TPboYG",
    f"https://www.youtube.com/watch?list=PLx0sYbCqOb8TBPRdmBHs5Iftvv9TPboYG&v={VIDEO_ID}",
    f"https://youtu.be/{VIDEO_ID}?si=abc123&t=42",
    f"https://www.youtube.com/shorts/{VIDEO_ID}?feature=share",
    f"  https://www.youtube.com/watch?v={VIDEO_ID}  ",
    f"HTTPS://WWW.YOUTUBE.COM/watch?v={VIDEO_ID}",
])
def test_valid_urls_normalise_to_watch_link(url):
    assert validate_youtube_url(url) == CANONICAL


@pytest.mark.parametrize("url", [
    "",
    "   ",
    f"https://vimeo.com/{VIDEO_ID}",
    f"https://example.com/watch?v={VIDEO_ID}",
    f"https://youtube.com.evil.com/watch?v={VIDEO_ID}",
    f"https://evil.com/youtube.com/watch?v={VIDEO_ID}",
    f"https://youtube.com@evil.com/watch?v={VIDEO_ID}",
    f"https://notyoutube.com/watch?v={VIDEO_ID}",
    f"javascript:alert('https://www.youtube.com/watch?v={VIDEO_ID}')",
    f"ftp://www.youtube.com/watch?v={VIDEO_ID}",
    f"www.youtube.com/watch?v={VIDEO_ID}",  # no scheme
    "https://www.youtube.com/playlist?list=PLx0sYbCqOb8TBPRdmBHs5Iftvv9TPboYG",
    "https://www.youtube.com/@somechannel",
    "https://www.youtube.com/watch",
    "https://www.youtube.com/watch?v=tooShort",
    "https://www.youtube.com/watch?v=<script>xx",
    "https://youtu.be/",
])
def test_invalid_urls_raise_value_error(url):
    with pytest.raises(ValueError):
        validate_youtube_url(url)


# --------------------------------------------------------------------------
# get_video_info / error translation (yt_dlp mocked)
# --------------------------------------------------------------------------

class FakeYoutubeDL:
    """Stands in for yt_dlp.YoutubeDL. Set .result or .error per test."""
    result: dict = {}
    error: Exception | None = None
    last_opts: dict = {}

    def __init__(self, opts):
        FakeYoutubeDL.last_opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        if FakeYoutubeDL.error:
            raise FakeYoutubeDL.error
        return FakeYoutubeDL.result


@pytest.fixture
def fake_ydl(monkeypatch):
    FakeYoutubeDL.result = {}
    FakeYoutubeDL.error = None
    monkeypatch.setattr(youtube_service.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    return FakeYoutubeDL


def test_get_video_info_returns_title_and_duration(fake_ydl):
    fake_ydl.result = {"title": "Lecture 1: Intro", "duration": 1234, "formats": []}

    assert get_video_info(CANONICAL) == {"title": "Lecture 1: Intro", "duration": 1234}
    # Metadata only: never downloads, never expands playlists.
    assert fake_ydl.last_opts["skip_download"] is True
    assert fake_ydl.last_opts["noplaylist"] is True


def test_get_video_info_live_stream_has_no_duration(fake_ydl):
    fake_ydl.result = {"title": "Live now", "duration": None}
    assert get_video_info(CANONICAL)["duration"] is None


def test_download_error_becomes_youtube_error(fake_ydl):
    fake_ydl.error = DownloadError("ERROR: [youtube] dQw4w9WgXcQ: Something odd happened")
    with pytest.raises(YouTubeError) as exc:
        get_video_info(CANONICAL)
    # The yt-dlp prefix is stripped from the user-facing message.
    assert "ERROR: [youtube]" not in str(exc.value)
    assert "Something odd happened" in str(exc.value)


@pytest.mark.parametrize("raw,expected", [
    ("ERROR: [youtube] dQw4w9WgXcQ: Private video. Sign in if you've been granted access", "private"),
    ("ERROR: [youtube] dQw4w9WgXcQ: Sign in to confirm your age. This video may be inappropriate", "age-restricted"),
    ("ERROR: [youtube] dQw4w9WgXcQ: The uploader has not made this video available in your country", "region"),
    ("ERROR: [youtube] dQw4w9WgXcQ: Video unavailable. This video has been removed by the uploader", "removed"),
    ("ERROR: [youtube] aaaaaaaaaaa: This video is unavailable", "unavailable"),
    ("ERROR: [youtube] dQw4w9WgXcQ: Sign in to confirm you’re not a bot", "bot"),
])
def test_known_errors_get_friendly_messages(fake_ydl, raw, expected):
    fake_ydl.error = DownloadError(raw)
    with pytest.raises(YouTubeError) as exc:
        get_video_info(CANONICAL)
    assert expected in str(exc.value).lower()
    assert "ERROR:" not in str(exc.value)


def test_network_errors_are_transient(fake_ydl):
    fake_ydl.error = DownloadError(
        "ERROR: [youtube] dQw4w9WgXcQ: Unable to download webpage: <urlopen error timed out>"
    )
    with pytest.raises(YouTubeTransientError):
        get_video_info(CANONICAL)
