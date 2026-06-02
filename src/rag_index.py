"""ChromaDB-backed RAG index with pluggable embeddings.

The index stores one ChromaDB collection (`kdart_chunks` by default) with:
  ids:         chunk_id
  documents:   chunk text
  embeddings:  sentence-transformers vectors (default: BAAI/bge-m3, 1024-dim)
  metadatas:   {company, year, section, report_code, has_table, token_count}

The default embedder is a free, local `sentence-transformers` model
(`LocalEmbedder`) so the whole pipeline — index build, query, calibration —
runs with no OpenAI API key and zero embedding cost. The legacy
`OpenAIEmbedder` is kept so `rag.provider: openai` in config.yaml can switch
back. Use `build_embedder(cfg["rag"])` to construct the right one.

Metadata filtering uses Chroma's `where` clause so callers can constrain
retrieval by company or year (e.g. for evidence selection during Phase 2).

NOTE: switching providers (or models with a different `embedding_dim`) changes
the vector dimension, so the existing Chroma collection must be rebuilt with
`scripts/03_build_index.py --reset`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import chromadb
import tiktoken
from chromadb.api.types import EmbeddingFunction

from .common import ensure_dir, get_logger, require_env

log = get_logger("rag_index")

# OpenAI embedding models cap inputs at 8192 tokens. We leave headroom because
# tiktoken's count and the server-side count can diverge by a handful of tokens
# on edge cases (BOMs, escaped sequences inside HTML tables).
_EMBEDDING_TOKEN_LIMIT = 8000


@dataclass
class EmbeddingCostEstimate:
    n_chunks: int
    total_tokens: int
    estimated_usd: float

    def render(self) -> str:
        return (
            f"청크: {self.n_chunks:,}개, "
            f"총 토큰: {self.total_tokens:,}, "
            f"예상 비용: ${self.estimated_usd:.4f}"
        )


class OpenAIEmbedder(EmbeddingFunction):
    """Thin wrapper that lets ChromaDB call OpenAI for query-time embeddings."""

    def __init__(self, model: str, api_key: str | None = None, batch_size: int = 100):
        from openai import OpenAI  # lazy: keep the local-embedding path OpenAI-free

        self._client = OpenAI(api_key=api_key or require_env("OPENAI_API_KEY"))
        self._model = model
        self._batch = batch_size
        self._enc = tiktoken.get_encoding("cl100k_base")

    def name(self) -> str:
        return f"openai/{self._model}"

    def __call__(self, input: list[str]) -> list[list[float]]:  # type: ignore[override]
        return self.embed(input)

    def _truncate(self, text: str) -> str:
        """Cap a single input at the embedding model's token limit.

        DART filings sometimes produce a single oversize table chunk (the
        chunker keeps tables whole when preserve_tables=true). Without
        truncation, OpenAI rejects the whole batch with a 400.
        """
        toks = self._enc.encode(text)
        if len(toks) <= _EMBEDDING_TOKEN_LIMIT:
            return text
        log.warning(
            "Truncating embedding input from %d to %d tokens (full text still stored in chunk).",
            len(toks), _EMBEDDING_TOKEN_LIMIT,
        )
        return self._enc.decode(toks[:_EMBEDDING_TOKEN_LIMIT])

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            batch = [self._truncate(t) for t in texts[i:i + self._batch]]
            for attempt in range(5):
                try:
                    resp = self._client.embeddings.create(model=self._model, input=batch)
                    out.extend([d.embedding for d in resp.data])
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    log.warning("Embedding batch failed (attempt %d/5): %s; sleeping %ds", attempt + 1, e, wait)
                    time.sleep(wait)
            else:
                raise RuntimeError(f"Embedding failed after retries (batch starting {i})")
        return out


class LocalEmbedder(EmbeddingFunction):
    """Free, local sentence-transformers embedder (default: BAAI/bge-m3).

    Implements ChromaDB's ``EmbeddingFunction`` contract (``__call__`` + ``name``)
    so the rest of the pipeline is untouched. Multilingual models such as
    ``BAAI/bge-m3`` (1024-dim) and ``intfloat/multilingual-e5-large`` (1024-dim)
    handle Korean far better than OpenAI's cl100k tokenizer-based models and
    cost nothing.

    Query vs. passage asymmetry: e5 models expect ``"query: "`` / ``"passage: "``
    prefixes. We apply ``query_prefix`` in :meth:`__call__` (Chroma calls this
    for query texts) and ``passage_prefix`` in :meth:`embed` (used to embed the
    stored documents). bge-m3 needs no prefix, so both default to "".
    """

    def __init__(
        self,
        model: str,
        *,
        normalize: bool = True,
        query_prefix: str = "",
        passage_prefix: str = "",
        batch_size: int = 32,
        max_seq_length: int = 8192,
        max_batch_tokens: int = 6144,
        device: str = "auto",
    ) -> None:
        from sentence_transformers import SentenceTransformer  # lazy/heavy import

        resolved = self._resolve_device(device)
        log.info("Loading local embedding model %s on %s", model, resolved)
        self._st = SentenceTransformer(model, device=resolved)
        # Cap the model's own truncation as a safety net beyond our _truncate().
        if max_seq_length:
            self._st.max_seq_length = int(max_seq_length)
        self._model = model
        self._normalize = normalize
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._batch = batch_size
        self._max_seq = int(max_seq_length)
        # Memory bound: a batch is padded to its longest member, so peak
        # activation scales with (batch_count × longest_seq). We cap that product
        # at max_batch_tokens so a single long chunk (e.g. an oversize 5k-token
        # table) is embedded nearly alone while short chunks pack densely. This
        # keeps bge-m3 (fp32, ~2.3GB) within a small-RAM/CPU box.
        self._max_batch_tokens = int(max_batch_tokens)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device and device != "auto":
            return device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def name(self) -> str:
        return f"local/{self._model}"

    def __call__(self, input: list[str]) -> list[list[float]]:  # type: ignore[override]
        # Chroma calls this to embed *query* texts.
        return self._encode(input, self._query_prefix)

    def _truncate(self, text: str) -> str:
        """Cap a single input at the model's max sequence length (token-based).

        DART filings sometimes produce a single oversize table chunk (the
        chunker keeps tables whole when preserve_tables=true). We pre-truncate
        with the model's own tokenizer, reserving a small headroom for the
        special tokens sentence-transformers adds.
        """
        budget = max(self._max_seq - 2, 1)
        ids = self._st.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= budget:
            return text
        log.warning(
            "Truncating embedding input from %d to %d tokens (full text still stored in chunk).",
            len(ids), budget,
        )
        return self._st.tokenizer.decode(ids[:budget])

    def _length_aware_batches(self, lengths: list[int]) -> list[list[int]]:
        """Group text indices (sorted by token length) into memory-bounded batches.

        Greedy bin-packing on the *padded* cost: a batch padded to its longest
        member costs ``len(batch) × max_len``. We keep that ≤ max_batch_tokens
        and the count ≤ batch_size. Returns lists of original indices.
        """
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        batches: list[list[int]] = []
        cur: list[int] = []
        cur_max = 0
        for i in order:
            li = max(lengths[i], 1)
            new_max = max(cur_max, li)
            if cur and (
                len(cur) + 1 > self._batch
                or (len(cur) + 1) * new_max > self._max_batch_tokens
            ):
                batches.append(cur)
                cur, cur_max = [], 0
                new_max = li
            cur.append(i)
            cur_max = new_max
        if cur:
            batches.append(cur)
        return batches

    def _encode(self, texts: Sequence[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []
        prepared = [prefix + self._truncate(t) for t in texts]
        lengths = [
            min(len(self._st.tokenizer.encode(t, add_special_tokens=False)), self._max_seq)
            for t in prepared
        ]
        out: list[list[float]] = [None] * len(prepared)  # type: ignore[list-item]
        for batch_idx in self._length_aware_batches(lengths):
            batch_texts = [prepared[i] for i in batch_idx]
            vecs = self._st.encode(
                batch_texts,
                batch_size=len(batch_texts),
                normalize_embeddings=self._normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            for slot, v in zip(batch_idx, vecs):
                out[slot] = v.tolist()
        return out

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # Used to embed stored *documents* (passages).
        return self._encode(texts, self._passage_prefix)


def build_embedder(rag_cfg: dict) -> EmbeddingFunction:
    """Construct the embedder selected by ``rag.provider`` in config.yaml.

    ``local``  → :class:`LocalEmbedder` (free, default)
    ``openai`` → :class:`OpenAIEmbedder` (legacy, requires OPENAI_API_KEY)
    """
    provider = rag_cfg.get("provider", "local")
    if provider == "local":
        return LocalEmbedder(
            model=rag_cfg["model"],
            normalize=rag_cfg.get("normalize_embeddings", True),
            query_prefix=rag_cfg.get("query_prefix", "") or "",
            passage_prefix=rag_cfg.get("passage_prefix", "") or "",
            batch_size=rag_cfg.get("embedding_batch_size", 32),
            max_seq_length=rag_cfg.get("max_seq_length", 8192),
            max_batch_tokens=rag_cfg.get("max_batch_tokens", 6144),
            device=rag_cfg.get("device", "auto"),
        )
    if provider == "openai":
        return OpenAIEmbedder(
            model=rag_cfg["model"],
            batch_size=rag_cfg.get("embedding_batch_size", 100),
        )
    raise ValueError(f"Unknown rag.provider: {provider!r} (expected 'local' or 'openai')")


class RagIndex:
    def __init__(
        self,
        persist_dir: Path,
        *,
        collection_name: str,
        embedder: EmbeddingFunction,
    ) -> None:
        ensure_dir(persist_dir)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._embedder = embedder
        self.collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"embedding_model": embedder.name()},
            embedding_function=embedder,
        )

    # ---- write ----

    def add_chunks(
        self,
        chunks: list[dict],
        *,
        embeddings: list[list[float]] | None = None,
        upsert: bool = True,
    ) -> None:
        if not chunks:
            return
        ids = [c["chunk_id"] for c in chunks]
        docs = [c["text"] for c in chunks]
        metas = [
            {
                "company": c["company"],
                "year": c["year"],
                "section": c["section"],
                "report_code": c.get("report_code") or "",
                "has_table": bool(c.get("has_table", False)),
                "token_count": int(c.get("token_count", 0)),
            }
            for c in chunks
        ]
        if embeddings is None:
            embeddings = self._embedder.embed(docs)
        if upsert:
            self.collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
        else:
            self.collection.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)

    # ---- read ----

    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict]:
        result = self.collection.query(
            query_texts=[text],
            n_results=top_k,
            where=where,
        )
        hits: list[dict] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for i, cid in enumerate(ids):
            hits.append(
                {
                    "chunk_id": cid,
                    "text": docs[i],
                    "metadata": metas[i],
                    "distance": dists[i] if i < len(dists) else None,
                }
            )
        return hits

    def count(self) -> int:
        return self.collection.count()


def estimate_embedding_cost(
    chunks: Iterable[dict],
    *,
    price_per_1m_tokens_usd: float,
) -> EmbeddingCostEstimate:
    total = 0
    n = 0
    for c in chunks:
        total += int(c.get("token_count", 0))
        n += 1
    cost = total / 1_000_000 * price_per_1m_tokens_usd
    return EmbeddingCostEstimate(n_chunks=n, total_tokens=total, estimated_usd=cost)
