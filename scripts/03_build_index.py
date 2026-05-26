#!/usr/bin/env python3
"""Phase 1-3 & 1-4: Chunk extracted sections and build the ChromaDB RAG index.

Pipeline:
  data/sections/*.html  →  data/chunks/{company}_{year}.jsonl  →  data/index/
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import Chunker, chunk_section_html  # noqa: E402
from src.common import ensure_dir, get_logger, load_config, project_path  # noqa: E402
from src.rag_index import OpenAIEmbedder, RagIndex, estimate_embedding_cost  # noqa: E402

log = get_logger("build_index", "phase1_build_index.log")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunk-only", action="store_true",
                   help="Stop after writing chunk JSONL (skip embeddings + ChromaDB).")
    p.add_argument("--yes", action="store_true",
                   help="Skip the cost-confirmation prompt.")
    p.add_argument("--reset", action="store_true",
                   help="Drop the existing Chroma collection before indexing.")
    return p.parse_args()


def discover_section_files(sections_dir: Path) -> list[tuple[str, int, str, Path]]:
    """Return list of (company, year, section_id, html_path)."""
    out = []
    for path in sorted(sections_dir.glob("*.html")):
        stem = path.stem  # company_year_sectionid
        parts = stem.split("_", 2)
        if len(parts) != 3:
            log.warning("Skipping malformed section filename: %s", path.name)
            continue
        company, year_str, section_id = parts
        try:
            year = int(year_str)
        except ValueError:
            log.warning("Skipping (bad year): %s", path.name)
            continue
        out.append((company, year, section_id, path))
    return out


def load_report_codes(manifest_path: Path) -> dict[tuple[str, int], str | None]:
    if not manifest_path.exists():
        return {}
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {(r["company"], r["year"]): r.get("report_code") for r in rows}


def write_chunk_jsonl(chunks_by_company_year: dict[tuple[str, int], list[dict]], out_dir: Path) -> None:
    ensure_dir(out_dir)
    for (company, year), rows in chunks_by_company_year.items():
        path = out_dir / f"{company}_{year}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info("Wrote %d chunks → %s", len(rows), path)


def main() -> int:
    args = parse_args()
    cfg = load_config()

    sections_dir = project_path(cfg["paths"]["sections"])
    chunks_dir = ensure_dir(project_path(cfg["paths"]["chunks"]))
    index_dir = project_path(cfg["paths"]["index"])
    manifest_path = project_path(cfg["paths"]["raw"]) / "manifest.json"

    section_files = discover_section_files(sections_dir)
    if not section_files:
        log.error("No section HTML files found under %s — run scripts/02_extract_sections.py first.", sections_dir)
        return 1
    report_codes = load_report_codes(manifest_path)

    chunker = Chunker(
        tokenizer=cfg["chunker"]["tokenizer"],
        chunk_tokens=cfg["chunker"]["chunk_tokens"],
        overlap_tokens=cfg["chunker"]["overlap_tokens"],
    )

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for company, year, section_id, html_path in section_files:
        html = html_path.read_text(encoding="utf-8")
        chunks = chunk_section_html(
            html,
            company=company,
            year=year,
            section=section_id,
            report_code=report_codes.get((company, year)),
            chunker=chunker,
        )
        grouped[(company, year)].extend(c.to_dict() for c in chunks)

    all_chunks = [c for rows in grouped.values() for c in rows]
    log.info("Built %d chunks across %d (company, year) pairs", len(all_chunks), len(grouped))

    write_chunk_jsonl(grouped, chunks_dir)

    if args.chunk_only:
        log.info("--chunk-only set; skipping indexing.")
        return 0

    # ---- Embedding cost confirmation ----
    est = estimate_embedding_cost(all_chunks, price_per_1m_tokens_usd=cfg["rag"]["cost_per_1m_tokens_usd"])
    log.info("임베딩 비용 예상: %s", est.render())
    if not args.yes:
        answer = input(f"임베딩을 진행하시겠습니까? ({est.render()}) [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            log.info("Aborted by user.")
            return 0

    # ---- Build / refresh ChromaDB collection ----
    embedder = OpenAIEmbedder(
        model=cfg["rag"]["model"],
        batch_size=cfg["rag"]["embedding_batch_size"],
    )
    if args.reset and index_dir.exists():
        import shutil
        shutil.rmtree(index_dir)
        log.info("Reset Chroma index at %s", index_dir)

    index = RagIndex(
        persist_dir=index_dir,
        collection_name=cfg["rag"]["collection_name"],
        embedder=embedder,
    )
    batch = cfg["rag"]["embedding_batch_size"]
    for i in range(0, len(all_chunks), batch):
        index.add_chunks(all_chunks[i:i + batch])
        log.info("Indexed %d / %d", min(i + batch, len(all_chunks)), len(all_chunks))

    log.info("Collection size now: %d", index.count())
    log.info("Embedding model: %s | dim: %d", cfg["rag"]["model"], cfg["rag"]["embedding_dim"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
