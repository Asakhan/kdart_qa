#!/usr/bin/env python3
"""Phase 2: produce the 40-question draft via Claude, self-verified."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import ensure_dir, get_logger, load_config, project_path  # noqa: E402
from src.question_generator import QuestionGenerator  # noqa: E402
from src.rag_index import OpenAIEmbedder, RagIndex  # noqa: E402

log = get_logger("generate_drafts", "phase2_drafts.log")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    return p.parse_args()


def build_pool(manifest_path: Path) -> list[tuple[str, int, str]]:
    """Return (company, year, report_label) tuples for items actually on disk."""
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    pool = []
    for r in rows:
        if r["status"] in {"downloaded", "cached"}:
            label = f"{r['year']}_{(r.get('report_code') or '?')[:5]}"
            pool.append((r["company"], r["year"], label))
    return pool


def main() -> int:
    args = parse_args()
    cfg = load_config()

    index_dir = project_path(cfg["paths"]["index"])
    drafts_dir = ensure_dir(project_path(cfg["paths"]["drafts"]))
    manifest_path = project_path(cfg["paths"]["raw"]) / "manifest.json"
    if not manifest_path.exists():
        log.error("manifest.json missing — run scripts/01_fetch_dart.py first.")
        return 1

    embedder = OpenAIEmbedder(model=cfg["rag"]["model"], batch_size=cfg["rag"]["embedding_batch_size"])
    index = RagIndex(
        persist_dir=index_dir,
        collection_name=cfg["rag"]["collection_name"],
        embedder=embedder,
    )
    if index.count() == 0:
        log.error("RAG collection is empty — run scripts/03_build_index.py first.")
        return 1

    pool = build_pool(manifest_path)
    log.info("Company-year pool size: %d", len(pool))

    gen = QuestionGenerator(
        index=index,
        model=cfg["generation"]["drafter_model"],
        max_tokens=cfg["generation"]["drafter_max_tokens"],
        max_attempts=cfg["generation"]["max_regenerate_attempts"],
        max_per_company_year=cfg["generation"]["max_per_company_year"],
        seed=args.seed,
    )
    items = gen.generate_all(cfg["generation"]["distribution"], company_year_pool=pool)

    out_path = Path(args.out) if args.out else drafts_dir / cfg["generation"]["draft_filename"]
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote %d drafts → %s", len(items), out_path)

    # Stats
    by_type = Counter(i["task_type"] for i in items)
    by_cy = Counter((i["source_company"], i.get("source_report", "")) for i in items)
    avg_hops = sum(i["reasoning_hops"] for i in items) / max(len(items), 1)
    unique_chunks = {cid for i in items for cid in i["evidence_chunks"]}

    log.info("==== Phase 2 통계 ====")
    log.info("유형별 시도/성공/폐기:")
    for code in cfg["generation"]["distribution"]:
        log.info(
            "  %s — attempts=%d success=%d discard=%d",
            code,
            gen.stats.attempts.get(code, 0),
            gen.stats.successes.get(code, 0),
            gen.stats.discards.get(code, 0),
        )
    log.info("회사·보고서별 분포: %s", dict(by_cy))
    log.info("평균 reasoning_hops: %.2f", avg_hops)
    log.info("사용된 unique evidence_chunks: %d", len(unique_chunks))
    if gen.stats.discard_reasons:
        log.info("폐기 사유 샘플: %s", gen.stats.discard_reasons[:10])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
