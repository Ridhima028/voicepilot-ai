"""
Tests for services/semantic_segmenter.py. No network: httpx.Client.post is
replaced with a fake OpenRouter, and time.sleep with a recorder.
"""

import httpx
import numpy as np
import pytest

from services import semantic_segmenter as seg
from services.semantic_segmenter import EmbeddingError


def vector_for(text: str) -> list[float]:
    """Deterministic fake embedding: "t7" -> [7.0, 1.0]."""
    return [float(text[1:]), 1.0]


def ok_response(texts, shuffle=False) -> httpx.Response:
    data = [{"index": i, "embedding": vector_for(t)} for i, t in enumerate(texts)]
    if shuffle:
        data.reverse()
    return httpx.Response(200, json={"data": data})


@pytest.fixture
def openrouter(monkeypatch):
    """Records each request. Set .replies to queue responses; once they run
    out, every request gets a normal 200."""

    class Fake:
        calls = []
        replies = []
        sleeps = []
        shuffle = False

    Fake.calls, Fake.replies, Fake.sleeps, Fake.shuffle = [], [], [], False

    def fake_post(self, url, headers=None, json=None, **kwargs):
        Fake.calls.append({"url": url, "headers": headers, "json": json})
        if Fake.replies:
            return Fake.replies.pop(0)
        return ok_response(json["input"], shuffle=Fake.shuffle)

    monkeypatch.setattr(seg, "QWEN_API_KEY", "test-key")
    monkeypatch.setattr(httpx.Client, "post", fake_post)
    monkeypatch.setattr(seg.time, "sleep", Fake.sleeps.append)
    return Fake


def texts(n: int) -> list[str]:
    return [f"t{i}" for i in range(n)]


# --------------------------------------------------------------------------
# _embed_texts via OpenRouter
# --------------------------------------------------------------------------

def test_calls_openrouter_with_model_and_key(openrouter):
    seg._embed_texts(texts(2))

    call = openrouter.calls[0]
    assert call["url"] == "https://openrouter.ai/api/v1/embeddings"
    assert call["json"] == {"model": seg.QWEN_EMBEDDING_MODEL, "input": ["t0", "t1"]}
    assert call["headers"]["Authorization"] == "Bearer test-key"
    assert call["headers"]["Content-Type"] == "application/json"


def test_vectors_follow_input_order_when_response_is_shuffled(openrouter):
    openrouter.shuffle = True

    result = seg._embed_texts(texts(3))

    assert isinstance(result, np.ndarray)
    assert result.tolist() == [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]


def test_250_texts_go_out_as_3_batches(openrouter):
    result = seg._embed_texts(texts(250))

    assert [len(c["json"]["input"]) for c in openrouter.calls] == [100, 100, 50]
    assert result.shape == (250, 2)
    # Concatenated in order across batches.
    assert result[0][0] == 0.0 and result[100][0] == 100.0 and result[249][0] == 249.0


def test_429_then_200_succeeds(openrouter):
    openrouter.replies = [httpx.Response(429, text="slow down")]

    result = seg._embed_texts(texts(2))

    assert result.shape == (2, 2)
    assert len(openrouter.calls) == 2
    assert openrouter.sleeps == [1]


def test_retries_three_times_with_1s_then_2s_waits(openrouter):
    openrouter.replies = [httpx.Response(503, text="busy") for _ in range(3)]

    with pytest.raises(EmbeddingError, match="503"):
        seg._embed_texts(texts(2))

    assert len(openrouter.calls) == 3
    assert openrouter.sleeps == [1, 2]


def test_401_raises_immediately(openrouter):
    openrouter.replies = [httpx.Response(401, text="bad key")]

    with pytest.raises(EmbeddingError, match="401"):
        seg._embed_texts(texts(2))

    assert len(openrouter.calls) == 1
    assert openrouter.sleeps == []


def test_count_mismatch_raises(openrouter):
    openrouter.replies = [ok_response(texts(2))]

    with pytest.raises(EmbeddingError, match="2 vectors for 3 texts"):
        seg._embed_texts(texts(3))


def test_malformed_response_raises(openrouter):
    openrouter.replies = [httpx.Response(200, json={"error": {"message": "nope"}})]

    with pytest.raises(EmbeddingError, match="unexpected"):
        seg._embed_texts(texts(2))


# --------------------------------------------------------------------------
# Gemini fallback + model name
# --------------------------------------------------------------------------

def test_without_qwen_key_uses_gemini(monkeypatch):
    monkeypatch.setattr(seg, "QWEN_API_KEY", None)

    class FakeModels:
        def embed_content(self, model, contents, config):
            assert model == seg.GEMINI_EMBEDDING_MODEL
            return type("R", (), {"embeddings": [type("E", (), {"values": [1.0, 2.0]}) for _ in contents]})

    monkeypatch.setattr(seg, "get_client", lambda: type("C", (), {"models": FakeModels()}))

    def no_network(*args, **kwargs):
        raise AssertionError("OpenRouter was called without a Qwen key")

    monkeypatch.setattr(httpx.Client, "post", no_network)

    assert seg._embed_texts(["a", "b"]).tolist() == [[1.0, 2.0], [1.0, 2.0]]


def test_embedding_model_name(monkeypatch):
    monkeypatch.setattr(seg, "QWEN_API_KEY", "test-key")
    assert seg.embedding_model_name() == seg.QWEN_EMBEDDING_MODEL

    monkeypatch.setattr(seg, "QWEN_API_KEY", None)
    assert seg.embedding_model_name() == seg.GEMINI_EMBEDDING_MODEL


# --------------------------------------------------------------------------
# semantic_chunk
# --------------------------------------------------------------------------

def test_semantic_chunk_merges_similar_and_splits_dissimilar(monkeypatch):
    segments = [
        {"start": 0.0, "end": 4.0, "text": "Buckets hold objects."},
        {"start": 4.5, "end": 8.0, "text": "Each object has a key."},
        {"start": 8.5, "end": 12.0, "text": "Now, about message queues."},
    ]
    # First two point the same way; the third is orthogonal to them.
    fake = np.array([[1.0, 0.0], [0.99, 0.1], [0.0, 1.0]])
    monkeypatch.setattr(seg, "_embed_texts", lambda texts: fake)

    chunks = seg.semantic_chunk(segments)

    assert chunks == [
        {"start": 0.0, "end": 8.0, "text": "Buckets hold objects. Each object has a key."},
        {"start": 8.5, "end": 12.0, "text": "Now, about message queues."},
    ]
