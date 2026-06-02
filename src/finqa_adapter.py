"""Work 3.5: convert K-DART-QA items into FinQA's item schema.

The downstream verification experiment reads FinQA originals and K-DART through a
single schema, so this adapter emits items shaped exactly like FinQA's. Crucially
the *evidence text is embedded inside each item* (pre_text / post_text / table),
so the experiment needs no retrieval step, no ChromaDB, and no API calls — only
the offline `data/final/*.jsonl` + `data/chunks/*.jsonl` already in the repo.

FinQA item schema (top-level dict; the full file is a single JSON array):
    id, pre_text[], post_text[], table[[...]], filename,
    qa: {question, answer, exe_ans, program, explanation, gold_inds{}}

Mapping decisions (kept simple and consistent):
  * Evidence text comes from the gold `evidence_chunks`, looked up by chunk_id in
    `data/chunks/{company}_{year}.jsonl` (offline). An optional ChromaDB fallback
    exists for chunk ids missing from the files.
  * Each chunk's text is segmented into text vs. Markdown-table blocks. Across all
    evidence chunks (in order) the **first** table becomes the `table` field;
    text before it becomes `pre_text`, text after it `post_text`. Any *additional*
    tables are absorbed into `post_text` as raw Markdown lines (single-table
    assumption — see _assemble_evidence). Chunks with no table contribute only to
    `pre_text` and leave `table` == [].
  * `gold_inds` uses K-DART `evidence_quotes`: a quote containing '|' is a table
    row → "table_N", otherwise prose → "text_N" (FinQA's key convention). If
    evidence_quotes is empty we fall back to the first table row / first sentence.
  * `exe_ans` = gold_answer parsed with the calibrator's parse_number() (commas /
    units stripped); the raw string is kept if it is non-numeric.
  * `program` = "" (not synthesised — user decision; grading is exe_ans-based).
  * `explanation` = reasoning_steps joined by newlines (traceability).
  * Original K-DART metadata is preserved under `kdart_meta` (FinQA code ignores
    it; kept for analysis).
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import Any

from .calibrator import parse_number
from .common import get_logger

log = get_logger("finqa_adapter")

_SEP_CELL_RE = re.compile(r"^:?-{2,}:?$")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?。])\s+")


# ---- chunk text source ----

def load_chunk_texts(chunks_dir: Path) -> dict[str, dict]:
    """Load every chunk JSONL into {chunk_id: chunk_dict} (offline, no API)."""
    out: dict[str, dict] = {}
    for path in glob.glob(str(chunks_dir / "*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                c = json.loads(line)
                out[c["chunk_id"]] = c
    return out


# ---- markdown table parsing ----

def is_table_line(line: str) -> bool:
    return line.strip().startswith("|")


def parse_markdown_table(md: str) -> list[list[str]]:
    """Parse a Markdown table block into a 2-D array (header + body rows).

    Pipe-delimited; the `---`/`:---:` separator row is dropped. Cells are
    stripped. Non-pipe lines are ignored. Rows are padded to a uniform width so
    the result is a rectangular matrix.
    """
    rows: list[list[str]] = []
    for raw in md.splitlines():
        if not is_table_line(raw):
            continue
        inner = raw.strip()
        # Drop the leading/trailing pipe, then split.
        if inner.startswith("|"):
            inner = inner[1:]
        if inner.endswith("|"):
            inner = inner[:-1]
        cells = [c.strip() for c in inner.split("|")]
        # Separator row: every cell is dashes/colons.
        if cells and all(_SEP_CELL_RE.match(c.replace(" ", "")) for c in cells if c != "") \
                and any(c for c in cells):
            continue
        rows.append(cells)
    if not rows:
        return []
    width = max(len(r) for r in rows)
    return [r + [""] * (width - len(r)) for r in rows]


def segment_blocks(text: str) -> list[tuple[str, str]]:
    """Split chunk text into ordered ("text"|"table", payload) blocks.

    Contiguous lines beginning with '|' form a table block; everything else is
    text. Robust to the "\n\n"-joined chunk layout produced by the chunker.
    """
    blocks: list[tuple[str, str]] = []
    buf: list[str] = []
    buf_kind: str | None = None

    def flush():
        nonlocal buf, buf_kind
        if buf and buf_kind:
            payload = "\n".join(buf).strip()
            if payload:
                blocks.append((buf_kind, payload))
        buf, buf_kind = [], None

    for line in text.splitlines():
        kind = "table" if is_table_line(line) else "text"
        if line.strip() == "":
            # Blank lines separate blocks but belong to neither.
            continue
        if buf_kind is None:
            buf_kind = kind
        elif kind != buf_kind:
            flush()
            buf_kind = kind
        buf.append(line)
    flush()
    return blocks


def split_sentences(text: str) -> list[str]:
    """Split prose into sentence strings (FinQA pre/post_text are sentence lists)."""
    text = " ".join(text.split())
    if not text:
        return []
    parts = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return parts or [text]


# ---- evidence assembly ----

def _assemble_evidence(chunk_texts: list[str]) -> tuple[list[str], list[str], list[list[str]]]:
    """Build (pre_text, post_text, table) from ordered evidence chunk texts.

    Single-table rule: the first Markdown table encountered becomes `table`;
    prose before it → pre_text, prose after it → post_text. Additional tables are
    flattened into post_text as their raw Markdown lines so no evidence is lost.
    """
    pre: list[str] = []
    post: list[str] = []
    table: list[list[str]] = []
    seen_table = False

    for ct in chunk_texts:
        for kind, payload in segment_blocks(ct):
            if kind == "table":
                if not seen_table:
                    table = parse_markdown_table(payload)
                    seen_table = True
                else:
                    # Extra table → keep as raw text lines in post_text.
                    post.extend(line for line in payload.splitlines() if line.strip())
            else:  # text
                target = post if seen_table else pre
                target.extend(split_sentences(payload))
    return pre, post, table


# ---- gold_inds ----

def build_gold_inds(evidence_quotes: list[str], table: list[list[str]],
                    pre: list[str], post: list[str]) -> dict[str, str]:
    """Map evidence quotes to FinQA gold_inds keys (table_N / text_N).

    A quote containing '|' is treated as a table row; otherwise prose. Falls back
    to the first table row / first sentence when no quotes are provided.
    """
    gold: dict[str, str] = {}
    table_i = 0
    text_i = 0
    quotes = [q for q in (evidence_quotes or []) if str(q).strip()]
    for q in quotes:
        q = str(q).strip()
        if "|" in q:
            gold[f"table_{table_i}"] = q
            table_i += 1
        else:
            gold[f"text_{text_i}"] = q
            text_i += 1
    if not gold:
        if table and len(table) > 1:
            gold["table_0"] = " | ".join(table[1])  # first body row
        elif pre:
            gold["text_0"] = pre[0]
        elif post:
            gold["text_0"] = post[0]
    return gold


# ---- exe_ans ----

def compute_exe_ans(gold_answer: Any) -> Any:
    if isinstance(gold_answer, (int, float)):
        return float(gold_answer)
    num = parse_number(str(gold_answer))
    return num if num is not None else gold_answer


# ---- per-item conversion ----

def convert_item(item: dict, chunk_lookup: dict[str, dict],
                 *, index_get=None) -> dict:
    """Convert one K-DART item to a FinQA-schema dict.

    `chunk_lookup` is {chunk_id: chunk}. `index_get(chunk_id) -> text|None` is an
    optional fallback (e.g. ChromaDB) used only when a chunk is missing offline.
    """
    chunk_texts: list[str] = []
    for cid in item.get("evidence_chunks", []):
        c = chunk_lookup.get(cid)
        if c is not None:
            chunk_texts.append(c["text"])
        elif index_get is not None:
            txt = index_get(cid)
            if txt:
                chunk_texts.append(txt)
            else:
                log.warning("Evidence chunk %s not found (item %s)", cid, item.get("id"))
        else:
            log.warning("Evidence chunk %s not found offline (item %s)", cid, item.get("id"))

    pre, post, table = _assemble_evidence(chunk_texts)
    gold_inds = build_gold_inds(item.get("evidence_quotes", []), table, pre, post)

    kdart_meta = {
        "task_type": item.get("task_type"),
        "task_type_design": item.get("task_type_design"),
        "reasoning_hops": item.get("reasoning_hops"),
        "source_company": item.get("source_company"),
        "source_report": item.get("source_report"),
        "source_section": item.get("source_section"),
        "gold_answer_unit": item.get("gold_answer_unit"),
        "evidence_chunks": item.get("evidence_chunks", []),
    }
    if "difficulty_group" in item:
        kdart_meta["difficulty_group"] = item["difficulty_group"]
    if "finqa_matched" in item:
        kdart_meta["finqa_matched"] = item["finqa_matched"]

    return {
        "id": item["id"],
        "pre_text": pre,
        "post_text": post,
        "table": table,
        "filename": item.get("source_report") or item.get("source_company") or item["id"],
        "qa": {
            "question": item["question"],
            "answer": str(item.get("gold_answer", "")),
            "exe_ans": compute_exe_ans(item.get("gold_answer")),
            "program": "",  # not synthesised — exe_ans-based grading
            "explanation": "\n".join(item.get("reasoning_steps", []) or []),
            "gold_inds": gold_inds,
        },
        "kdart_meta": kdart_meta,
    }


def convert_dataset(items: list[dict], chunk_lookup: dict[str, dict],
                    *, index_get=None) -> list[dict]:
    return [convert_item(it, chunk_lookup, index_get=index_get) for it in items]


# ---- self-checks ----

def validate(converted: list[dict]) -> dict[str, Any]:
    """Self-check: required fields present, tables rectangular, exe_ans filled."""
    required_top = {"id", "pre_text", "post_text", "table", "filename", "qa"}
    required_qa = {"question", "answer", "exe_ans", "program", "explanation", "gold_inds"}
    issues: list[str] = []
    non_rect = 0
    missing_exe = 0
    have_table = 0
    for it in converted:
        miss = required_top - set(it)
        if miss:
            issues.append(f"{it.get('id')}: missing {sorted(miss)}")
        qmiss = required_qa - set(it.get("qa", {}))
        if qmiss:
            issues.append(f"{it.get('id')}: qa missing {sorted(qmiss)}")
        table = it.get("table", [])
        if table:
            have_table += 1
            widths = {len(r) for r in table}
            if len(widths) > 1:
                non_rect += 1
                issues.append(f"{it.get('id')}: non-rectangular table widths={widths}")
        if it.get("qa", {}).get("exe_ans") in (None, ""):
            missing_exe += 1
    return {
        "n": len(converted),
        "with_table": have_table,
        "non_rectangular_tables": non_rect,
        "missing_exe_ans": missing_exe,
        "issues": issues,
    }
