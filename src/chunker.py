"""Token-budgeted chunker that keeps Markdown tables intact.

Algorithm:
  1. The section is pre-split into ordered ("text"|"table", payload) blocks
     by :mod:`src.table_converter`.
  2. We greedily pack blocks into chunks of ≤ `chunk_tokens`. A table block is
     atomic: if it doesn't fit, the current chunk is flushed and the table
     becomes its own chunk (even if it exceeds `chunk_tokens` — better one
     oversized chunk than a corrupted table).
  3. Text blocks are split at sentence-ish boundaries when they don't fit, with
     `overlap_tokens` re-included at the head of the next chunk so adjacent
     chunks share local context for retrieval.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Iterable

import tiktoken

from .common import get_logger
from .table_converter import split_html_into_blocks

log = get_logger("chunker")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    company: str
    year: int
    section: str
    report_code: str | None
    has_table: bool
    token_count: int
    block_kinds: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class Chunker:
    def __init__(
        self,
        *,
        tokenizer: str = "cl100k_base",
        chunk_tokens: int = 512,
        overlap_tokens: int = 50,
    ) -> None:
        self.enc = tiktoken.get_encoding(tokenizer)
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens

    def count_tokens(self, s: str) -> int:
        return len(self.enc.encode(s))

    # ---- internal helpers ----

    def _split_text_block(self, text: str, budget: int) -> tuple[str, str]:
        """Split `text` so that the first part fits in `budget` tokens.

        Returns (head, tail). Head is what we put into the current chunk,
        tail is what spills into the next.
        """
        ids = self.enc.encode(text)
        if len(ids) <= budget:
            return text, ""
        head_ids = ids[:budget]
        # Try to cut at a sentence/whitespace boundary for cleaner chunks.
        head = self.enc.decode(head_ids)
        # Walk back to last sentence end if we're mid-word.
        m = re.search(r"[\.!?。…\n][^.!?。…\n]*$", head)
        if m and m.start() > 0:
            cut = m.start() + 1
            head = head[:cut]
        tail = text[len(head):].lstrip()
        return head, tail

    def _overlap_tail(self, text: str) -> str:
        """Return the last `overlap_tokens` tokens of text, for chunk overlap."""
        if self.overlap_tokens <= 0:
            return ""
        ids = self.enc.encode(text)
        if len(ids) <= self.overlap_tokens:
            return text
        return self.enc.decode(ids[-self.overlap_tokens:])

    # ---- public ----

    def chunk_blocks(self, blocks: list[tuple[str, str]]) -> list[tuple[str, list[str], bool]]:
        """Pack (kind, payload) blocks into chunk-sized buckets.

        Returns list of (chunk_text, kinds_in_chunk, has_table).
        """
        chunks: list[tuple[str, list[str], bool]] = []
        buf: list[str] = []          # text parts currently in the chunk
        buf_kinds: list[str] = []
        buf_has_table = False
        buf_tokens = 0

        def flush() -> None:
            nonlocal buf, buf_kinds, buf_has_table, buf_tokens
            if not buf:
                return
            chunk_text = "\n\n".join(buf).strip()
            chunks.append((chunk_text, list(buf_kinds), buf_has_table))
            # Set up overlap for the next chunk (text-only overlap).
            overlap = self._overlap_tail(chunk_text)
            buf = []
            buf_kinds = []
            buf_has_table = False
            buf_tokens = 0
            if overlap:
                buf.append(overlap)
                buf_kinds.append("overlap")
                buf_tokens = self.count_tokens(overlap)

        for kind, payload in blocks:
            if not payload.strip():
                continue
            ptokens = self.count_tokens(payload)

            if kind == "table":
                # Atomic: try to fit, else flush and emit alone.
                if buf_tokens + ptokens <= self.chunk_tokens:
                    buf.append(payload)
                    buf_kinds.append("table")
                    buf_has_table = True
                    buf_tokens += ptokens
                else:
                    flush()
                    # Even if the table alone exceeds budget, keep it whole.
                    if ptokens > self.chunk_tokens:
                        log.info(
                            "Oversized table chunk (%d tokens > %d budget) — kept whole",
                            ptokens, self.chunk_tokens,
                        )
                    buf.append(payload)
                    buf_kinds.append("table")
                    buf_has_table = True
                    buf_tokens = ptokens
                    flush()
                continue

            # text block — may need splitting
            remaining = payload
            while remaining:
                budget = self.chunk_tokens - buf_tokens
                if budget <= 0:
                    flush()
                    budget = self.chunk_tokens - buf_tokens
                head, tail = self._split_text_block(remaining, budget)
                if not head:
                    flush()
                    continue
                buf.append(head)
                buf_kinds.append("text")
                buf_tokens += self.count_tokens(head)
                remaining = tail
                if remaining:
                    flush()

        flush()
        return chunks


def chunk_section_html(
    html: str,
    *,
    company: str,
    year: int,
    section: str,
    report_code: str | None,
    chunker: Chunker,
) -> list[Chunk]:
    """Convenience: HTML → blocks → Chunks with metadata."""
    blocks = split_html_into_blocks(html)
    packed = chunker.chunk_blocks(blocks)
    out: list[Chunk] = []
    for i, (text, kinds, has_table) in enumerate(packed):
        cid = f"{company}_{year}_{section}_{i:03d}"
        out.append(
            Chunk(
                chunk_id=cid,
                text=text,
                company=company,
                year=year,
                section=section,
                report_code=report_code,
                has_table=has_table,
                token_count=chunker.count_tokens(text),
                block_kinds=kinds,
            )
        )
    return out
