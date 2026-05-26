#!/usr/bin/env python3
"""Replenish missing-quota drafts after Phase 2.5 review.

Regenerates items for task types where reviewed quota fell short, using new
slot indices so IDs don't collide with the original draft set. Writes a
supplement JSON that the reviewer can walk through next.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import ensure_dir, get_logger, load_config, project_path  # noqa: E402
from src.question_generator import QuestionGenerator, TASK_TYPES  # noqa: E402
from src.rag_index import OpenAIEmbedder, RagIndex  # noqa: E402

log = get_logger("replenish_drafts", "phase2_replenish.log")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--need", default="T1:1,T2:2,T3:6,T4:4,T5:3",
                   help="Comma-separated need-per-type (default targets the slots that came up short).")
    p.add_argument("--out", default=None)
    return p.parse_args()


def parse_need(spec: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        k, v = chunk.split(":")
        out[k.strip()] = int(v.strip())
    return out


def build_pool(manifest_path: Path) -> list[tuple[str, int, str]]:
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    pool = []
    for r in rows:
        if r["status"] in {"downloaded", "cached"}:
            label = f"{r['year']}_{(r.get('report_code') or '?')[:5]}"
            pool.append((r["company"], r["year"], label))
    return pool


def load_existing_ids(drafts_path: Path) -> set[str]:
    if not drafts_path.exists():
        return set()
    items = json.loads(drafts_path.read_text(encoding="utf-8"))
    return {it["id"] for it in items}


def next_slot(existing_ids: set[str], code: str) -> int:
    used = []
    prefix = f"DART_{code}_"
    for eid in existing_ids:
        if eid.startswith(prefix):
            tail = eid[len(prefix):]
            if tail.isdigit():
                used.append(int(tail))
    return (max(used) + 1) if used else 1


def main() -> int:
    args = parse_args()
    cfg = load_config()

    drafts_dir = ensure_dir(project_path(cfg["paths"]["drafts"]))
    main_draft = drafts_dir / cfg["generation"]["draft_filename"]
    manifest_path = project_path(cfg["paths"]["raw"]) / "manifest.json"
    if not manifest_path.exists():
        log.error("manifest.json missing.")
        return 1

    need = parse_need(args.need)
    log.info("Replenish need: %s", need)

    embedder = OpenAIEmbedder(
        model=cfg["rag"]["model"],
        batch_size=cfg["rag"]["embedding_batch_size"],
    )
    index = RagIndex(
        persist_dir=project_path(cfg["paths"]["index"]),
        collection_name=cfg["rag"]["collection_name"],
        embedder=embedder,
    )
    if index.count() == 0:
        log.error("RAG collection empty.")
        return 1

    gen = QuestionGenerator(
        index=index,
        model=cfg["generation"]["drafter_model"],
        max_tokens=cfg["generation"]["drafter_max_tokens"],
        max_attempts=cfg["generation"]["max_regenerate_attempts"],
        max_per_company_year=cfg["generation"]["max_per_company_year"],
        seed=args.seed,
    )

    pool = build_pool(manifest_path)
    gen._rng.shuffle(pool)
    log.info("Pool size: %d", len(pool))

    existing_ids = load_existing_ids(main_draft)

    new_items: list[dict] = []
    per_cy_count: dict[tuple[str, int], int] = {}
    for code, quota in need.items():
        spec = TASK_TYPES[code]
        slot_idx = next_slot(existing_ids, code)
        produced = 0
        attempt_budget = quota * (gen.max_attempts + 3)
        tries = 0
        pool_idx = 0
        while produced < quota and tries < attempt_budget:
            tries += 1
            company, year, report = pool[pool_idx % len(pool)]
            pool_idx += 1
            if per_cy_count.get((company, year), 0) >= gen.max_per_company_year:
                continue
            item = gen.generate_one(spec, company=company, year=year, report=report, slot_index=slot_idx)
            if item is not None:
                new_items.append(item)
                existing_ids.add(item["id"])
                per_cy_count[(company, year)] = per_cy_count.get((company, year), 0) + 1
                slot_idx += 1
                produced += 1
        if produced < quota:
            log.warning("Quota for %s unmet: produced %d / %d", code, produced, quota)

    out_path = Path(args.out) if args.out else drafts_dir / "kdart_qa_draft_v1_supplement.json"
    out_path.write_text(json.dumps(new_items, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote %d supplement drafts → %s", len(new_items), out_path)

    by_type = Counter(i["task_type"] for i in new_items)
    by_cy = Counter((i["source_company"], i.get("source_report", "")) for i in new_items)
    log.info("==== Replenish 통계 ====")
    log.info("유형별 시도/성공/폐기:")
    for code in need:
        log.info(
            "  %s — attempts=%d success=%d discard=%d",
            code,
            gen.stats.attempts.get(code, 0),
            gen.stats.successes.get(code, 0),
            gen.stats.discards.get(code, 0),
        )
    log.info("유형별 산출: %s", dict(by_type))
    log.info("회사·보고서별 분포: %s", dict(by_cy))
    if gen.stats.discard_reasons:
        log.info("폐기 사유 샘플: %s", gen.stats.discard_reasons[:15])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
