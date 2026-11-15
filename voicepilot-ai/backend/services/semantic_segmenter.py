"""
Turns Whisper's raw, pause-based segments into semantically coherent chunks.

Whisper (via its VAD) breaks audio into segments purely based on silence —
it has no idea whether two adjacent segments are part of the same sentence,
the same thought, or completely unrelated topics. This module re-groups
those raw segments using two signals together:

  1. Pause length between segments  — a long silence often (but not always)
     marks a real break.
  2. Semantic similarity of the text — if the next segment is talking about
     something different, it's a break even if the pause was short. If it's
     continuing the same idea, it's NOT a break even if the pause was long
     (e.g. the speaker paused mid-sentence to think).

A chunk also has a hard max-duration cap, so one very consistent topic
doesn't turn into a five-minute unreadable block.

Uses an embedding API instead of a local model — this used to run on
sentence-transformers (which pulls in PyTorch, a 2GB+ download that isn't
needed anywhere else in this project). Calling an API instead means no
heavy local install, at the cost of needing network access + a key: Qwen via
OpenRouter when QWEN_API_KEY is set, otherwise Gemini.
"""

import time

import httpx
import numpy as np
from google.genai.types import EmbedContentConfig

from config import GEMINI_EMBEDDING_MODEL, QWEN_API_KEY, QWEN_EMBEDDING_MODEL
from services.gemini_client import get_client

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
EMBEDDING_BATCH_SIZE = 100
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class EmbeddingError(Exception):
    """The embedding API failed or returned something unusable."""
    pass


def embedding_model_name() -> str:
    """Which model _embed_texts uses. Vectors from different models can't be
    compared, so this gets stored alongside any embeddings we keep."""
    return QWEN_EMBEDDING_MODEL if QWEN_API_KEY else GEMINI_EMBEDDING_MODEL


def _embed_batch_openrouter(client: httpx.Client, texts: list[str]) -> list[list[float]]:
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": QWEN_EMBEDDING_MODEL, "input": texts}

    for attempt in range(3):
        response = client.post(OPENROUTER_EMBEDDINGS_URL, headers=headers, json=payload)
        if response.status_code < 400:
            break
        if response.status_code not in RETRYABLE_STATUS_CODES or attempt == 2:
            raise EmbeddingError(
                f"Qwen embedding failed: {response.status_code} {response.text[:300]}"
            )
        time.sleep(2 ** attempt)

    try:
        # The API may return items in any order; "index" ties each one to its input.
        items = sorted(response.json()["data"], key=lambda item: item["index"])
        vectors = [item["embedding"] for item in items]
    except (KeyError, TypeError, ValueError) as e:
        raise EmbeddingError("OpenRouter returned an unexpected embeddings response.") from e

    if len(vectors) != len(texts):
        raise EmbeddingError(
            f"Qwen embedding returned {len(vectors)} vectors for {len(texts)} texts."
        )
    return vectors


def _embed_texts_openrouter(texts: list[str]) -> np.ndarray:
    vectors: list[list[float]] = []
    # Sync client on purpose: this runs inside the Celery worker.
    with httpx.Client(timeout=60) as client:
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            vectors.extend(
                _embed_batch_openrouter(client, texts[start:start + EMBEDDING_BATCH_SIZE])
            )
    return np.array(vectors)


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Batched API calls for every segment's text, not one call per segment."""
    if QWEN_API_KEY:
        return _embed_texts_openrouter(texts)

    client = get_client()
    response = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=texts,
        config=EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
    )
    return np.array([embedding.values for embedding in response.embeddings])


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def semantic_chunk(
    raw_segments: list[dict],
    similarity_threshold: float = 0.55,
    max_pause_seconds: float = 1.5,
    max_chunk_duration: float = 25.0,
) -> list[dict]:
    """
    raw_segments: list of {"start": float, "end": float, "text": str},
    exactly what stt_service.transcribe_audio() already returns.

    Returns the same shape, but re-grouped into semantically coherent
    chunks instead of Whisper's raw pause-based ones.
    """
    if not raw_segments:
        return []

    if len(raw_segments) == 1:
        return [dict(raw_segments[0])]

    texts = [seg["text"] for seg in raw_segments]
    embeddings = _embed_texts(texts)

    chunks: list[dict] = []

    current = {
        "start": raw_segments[0]["start"],
        "end": raw_segments[0]["end"],
        "text": raw_segments[0]["text"],
    }
    # Running average embedding of everything merged into the current
    # chunk so far — comparing against the *chunk's* meaning as a whole,
    # not just the immediately previous sentence.
    current_embedding_sum = embeddings[0].copy()
    current_embedding_count = 1

    for i in range(1, len(raw_segments)):
        seg = raw_segments[i]
        seg_embedding = embeddings[i]

        pause = seg["start"] - current["end"]
        prospective_duration = seg["end"] - current["start"]

        avg_embedding = current_embedding_sum / current_embedding_count
        similarity = _cosine_similarity(avg_embedding, seg_embedding)

        should_break = (
            pause > max_pause_seconds
            or similarity < similarity_threshold
            or prospective_duration > max_chunk_duration
        )

        if should_break:
            chunks.append(current)
            current = {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            }
            current_embedding_sum = seg_embedding.copy()
            current_embedding_count = 1
        else:
            current["end"] = seg["end"]
            current["text"] = f"{current['text']} {seg['text']}"
            current_embedding_sum += seg_embedding
            current_embedding_count += 1

    chunks.append(current)
    return chunks
