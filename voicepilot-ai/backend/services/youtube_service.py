"""
YouTube support: validate a link, read its metadata, and download its audio.

The API only calls validate_youtube_url() and get_video_info() (fast, no
download). The heavy download_audio() runs in the Celery worker, which then
hands the file to the normal S3 -> Whisper pipeline.
"""

import glob
import os
import re
from urllib.parse import parse_qs, urlsplit

import yt_dlp
from yt_dlp.utils import DownloadError

ALLOWED_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}
SHORT_HOST = "youtu.be"

# YouTube video ids are always 11 characters from this alphabet.
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


class YouTubeError(Exception):
    """Permanent problem with the video (private, removed, blocked...).
    Retrying won't help; the message is safe to show to the user."""
    pass


class YouTubeTransientError(Exception):
    """Network-level failure talking to YouTube. Worth retrying later."""
    pass


# --------------------------------------------------------------------------
# URL validation
# --------------------------------------------------------------------------

def validate_youtube_url(url: str) -> str:
    """
    Accepts watch, shorts, live and youtu.be links on the YouTube hosts only,
    and returns the canonical https://www.youtube.com/watch?v=<id>.
    Raises ValueError for anything else (other sites, playlists without a
    video id, javascript: URLs...).
    """
    url = (url or "").strip()
    if not url:
        raise ValueError("Please paste a YouTube link.")

    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ValueError("That doesn't look like a web link. It should start with https://")

    # .hostname is lower-cased and excludes any port or "user@" prefix, so
    # "youtube.com.evil.com" and "youtube.com@evil.com" don't slip through.
    host = (parts.hostname or "").lower()
    path_parts = [p for p in parts.path.split("/") if p]

    video_id = None
    if host == SHORT_HOST:
        video_id = path_parts[0] if path_parts else None
    elif host in ALLOWED_HOSTS:
        if parts.path.rstrip("/") == "/watch":
            video_id = parse_qs(parts.query).get("v", [None])[0]
        elif len(path_parts) >= 2 and path_parts[0] in ("shorts", "live"):
            video_id = path_parts[1]
    else:
        raise ValueError("Only YouTube links are supported (youtube.com or youtu.be).")

    if not video_id or not _VIDEO_ID.match(video_id):
        raise ValueError(
            "That YouTube link doesn't point to a single video. "
            "Paste a video link (not a playlist or channel)."
        )

    return f"https://www.youtube.com/watch?v={video_id}"


# --------------------------------------------------------------------------
# yt-dlp error translation
# --------------------------------------------------------------------------

# (substring in yt-dlp's error, message for the user). Checked in order.
_PERMANENT_ERRORS = [
    ("private video", "This video is private, so it can't be transcribed."),
    ("confirm your age", "This video is age-restricted, so it can't be downloaded without signing in."),
    ("age-restricted", "This video is age-restricted, so it can't be downloaded without signing in."),
    # yt-dlp: "The uploader has not made this video available in your country"
    ("available in your country", "This video is blocked in the server's region."),
    ("blocked it in your country", "This video is blocked in the server's region."),
    ("not a bot", "YouTube is asking the server to prove it isn't a bot, so it refused the download. "
                  "Try again later, or download the audio yourself and upload the file."),
    ("members-only", "This video is for channel members only."),
    ("join this channel", "This video is for channel members only."),
    ("premieres in", "This video hasn't premiered yet."),
    ("live event will begin", "This live stream hasn't started yet."),
    ("has been removed", "This video has been removed from YouTube."),
    ("account associated with this video has been terminated", "This video has been removed from YouTube."),
    ("video unavailable", "This video is unavailable (it may have been removed or made private)."),
    ("video is unavailable", "This video is unavailable (it may have been removed or made private)."),
    ("is not a valid url", "That link isn't a valid YouTube video."),
]

_TRANSIENT_HINTS = [
    "timed out", "timeout", "connection", "temporary failure in name resolution",
    "network is unreachable", "http error 5", "unable to download webpage",
    "errno", "remote end closed", "read operation timed out",
]


def _translate(error: DownloadError) -> Exception:
    raw = str(error)
    lowered = raw.lower()
    for needle, message in _PERMANENT_ERRORS:
        if needle in lowered:
            return YouTubeError(message)
    if any(hint in lowered for hint in _TRANSIENT_HINTS):
        return YouTubeTransientError(f"Couldn't reach YouTube: {_short(raw)}")
    return YouTubeError(f"Couldn't fetch this YouTube video: {_short(raw)}")


def _short(raw: str) -> str:
    # yt-dlp prefixes errors like "ERROR: [youtube] dQw4w9WgXcQ: ..."
    text = re.sub(r"^ERROR:\s*(\[[^\]]+\]\s*)?([A-Za-z0-9_-]{11}:\s*)?", "", raw.strip())
    return text[:200]


# --------------------------------------------------------------------------
# Metadata + download
# --------------------------------------------------------------------------

def get_video_info(url: str) -> dict:
    """Title and duration (seconds) without downloading anything."""
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "socket_timeout": 15,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except DownloadError as e:
        raise _translate(e) from e

    return {
        "title": info.get("title") or "YouTube video",
        # None for a live stream that is still running.
        "duration": info.get("duration"),
    }


def download_audio(url: str, out_dir: str, on_activity=None) -> str:
    """
    Downloads only the audio track into out_dir and returns the file path.
    Always ends up as .m4a (ffmpeg converts if YouTube served another codec),
    so the rest of the pipeline sees one predictable format.

    on_activity() is called on every yt-dlp progress update, if given; the
    worker uses it as a (throttled) heartbeat during long downloads.
    """
    opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "socket_timeout": 30,
        "retries": 3,
        "outtmpl": f"{out_dir}/%(id)s.%(ext)s",
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "m4a"},
        ],
    }
    if on_activity:
        opts["progress_hooks"] = [lambda _status: on_activity()]
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except DownloadError as e:
        raise _translate(e) from e

    # After post-processing, yt-dlp records the final path here.
    for item in info.get("requested_downloads") or []:
        path = item.get("filepath")
        if path and os.path.exists(path):
            return path

    # Fallback: the only .m4a in our private temp dir is ours.
    matches = glob.glob(os.path.join(out_dir, "*.m4a"))
    if matches:
        return matches[0]
    raise YouTubeError("The audio download finished but no audio file was produced.")
