"""Unit tests for the local sentence-transformers embedder.

Fast tests inject a fake SentenceTransformer so they need no model download and
verify the logic we actually own: query/passage prefix routing, token-based
truncation, normalize pass-through, and the provider factory. One optional
integration test exercises the real BAAI/bge-m3 model (dimension + L2 norm) and
is skipped if the weights are unavailable offline.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from src.rag_index import LocalEmbedder, build_embedder


class _FakeTokenizer:
    """Whitespace tokenizer good enough to exercise _truncate()."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


class _FakeST:
    """Stand-in for sentence_transformers.SentenceTransformer."""

    last_inputs: list[str] = []
    last_kwargs: dict = {}

    def __init__(self, model, device=None):
        self.model = model
        self.device = device
        self.max_seq_length = 512
        self.tokenizer = _FakeTokenizer()

    def encode(self, texts, **kwargs):
        _FakeST.last_inputs = list(texts)
        _FakeST.last_kwargs = kwargs
        # Return a deterministic 3-dim vector per input.
        return np.array([[float(len(t)), 1.0, 2.0] for t in texts])


@pytest.fixture
def fake_st(monkeypatch):
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    return _FakeST


def test_query_uses_query_prefix(fake_st):
    emb = LocalEmbedder("dummy", query_prefix="query: ", passage_prefix="passage: ")
    vecs = emb(["삼성전자 매출"])  # __call__ → query path
    assert fake_st.last_inputs == ["query: 삼성전자 매출"]
    assert len(vecs) == 1 and len(vecs[0]) == 3


def test_passage_uses_passage_prefix(fake_st):
    emb = LocalEmbedder("dummy", query_prefix="query: ", passage_prefix="passage: ")
    emb.embed(["연결 매출액은 100조원"])  # embed() → passage path
    assert fake_st.last_inputs == ["passage: 연결 매출액은 100조원"]


def test_no_prefix_for_bge_style(fake_st):
    emb = LocalEmbedder("dummy")  # default prefixes are ""
    emb(["질문"])
    assert fake_st.last_inputs == ["질문"]


def test_normalize_flag_passed_through(fake_st):
    emb = LocalEmbedder("dummy", normalize=True)
    emb(["x"])
    assert fake_st.last_kwargs.get("normalize_embeddings") is True
    emb2 = LocalEmbedder("dummy", normalize=False)
    emb2(["x"])
    assert fake_st.last_kwargs.get("normalize_embeddings") is False


def test_truncate_caps_long_input(fake_st):
    emb = LocalEmbedder("dummy", max_seq_length=10)
    long_text = " ".join(str(i) for i in range(100))  # 100 whitespace tokens
    truncated = emb._truncate(long_text)
    # budget = max_seq_length - 2 headroom for special tokens
    assert len(truncated.split()) == 8


def test_truncate_keeps_short_input(fake_st):
    emb = LocalEmbedder("dummy", max_seq_length=100)
    short = "짧은 문장 입니다"
    assert emb._truncate(short) == short


def test_build_embedder_local(fake_st):
    emb = build_embedder({"provider": "local", "model": "dummy"})
    assert isinstance(emb, LocalEmbedder)
    assert emb.name() == "local/dummy"


def test_build_embedder_unknown_provider():
    with pytest.raises(ValueError):
        build_embedder({"provider": "nope", "model": "x"})


@pytest.mark.integration
def test_bge_m3_real_dimension_and_norm():
    """Real model: 1024-dim, L2-normalized. Skipped if weights unavailable."""
    try:
        emb = LocalEmbedder("BAAI/bge-m3", normalize=True, batch_size=4)
    except Exception as e:  # offline / no weights
        pytest.skip(f"bge-m3 unavailable: {e}")
    vecs = emb(["삼성전자의 2022년 연결 매출액은 얼마입니까?"])
    assert len(vecs[0]) == 1024
    norm = float(np.linalg.norm(np.array(vecs[0])))
    assert abs(norm - 1.0) < 1e-3
