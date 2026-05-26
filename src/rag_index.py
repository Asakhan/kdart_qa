"""ChromaDB-backed RAG index with OpenAI embeddings.

The index stores one ChromaDB collection (`kdart_chunks` by default) with:
  ids:         chunk_id
  documents:   chunk text
  embeddings:  OpenAI text-embedding-3-small vectors
  metadatas:   {company, year, section, report_code, has_table, token_count}

Metadata filtering uses Chroma's `where` clause so callers can constrain
retrieval by company or year (e.g. for evidence selection during Phase 2).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import chromadb
import tiktoken
from chromadb.api.types import EmbeddingFunction
from openai import OpenAI

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


class RagIndex:
    def __init__(
        self,
        persist_dir: Path,
        *,
        collection_name: str,
        embedder: OpenAIEmbedder,
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
