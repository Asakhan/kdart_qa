#!/usr/bin/env python3
"""Work 2: measure retrieval recall of the RAG index.

For each item we run ``RagIndex.query(question)`` and check whether the item's
gold ``evidence_chunks`` land in the top-k results. We report overall recall and
a per-``task_type`` breakdown so we can confirm that "fake difficulty" (items
the calibrator marks too_hard only because the gold chunk was never retrieved)
has been removed after switching to local embeddings + a larger top_k.

By default we mirror the calibrator's retrieval exactly: a company+year metadata
``where`` filter, falling back to an unfiltered query when the filter returns
nothing. Use ``--no-filter`` to measure raw (unfiltered) recall, and ``--top-k``
to sweep k. This reads the same config (and therefore the same embedder) the
calibrator uses, so no embedding/LLM cost beyond loading the local model.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import get_logger, load_config, project_path  # noqa: E402
from src.rag_index import RagIndex, build_embedder  # noqa: E402

log = get_logger("eval_retrieval", "phase1_eval_retrieval.log")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=None,
                   help="Items JSONL (default: data/final/<export.jsonl_filename> or kdart_qa_gpt_quota.jsonl).")
    p.add_argument("--top-k", type=int, default=None,
                   help="Override rag.top_k for this evaluation.")
    p.add_argument("--no-filter", action="store_true",
                   help="Do not apply the company+year metadata filter (raw recall).")
    p.add_argument("--show-misses", action="store_true",
                   help="Print each item whose gold chunk was not retrieved.")
    return p.parse_args()


def load_items(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def year_of(item: dict) -> int:
    head = str(item.get("source_report", ""))[:4]
    return int(head) if head.isdigit() else 0


def build_where(item: dict) -> dict:
    return {"$and": [
        {"company": item["source_company"]},
        {"year": year_of(item)},
    ]}


def main() -> int:
    args = parse_args()
    cfg = load_config()

    if args.input:
        in_path = Path(args.input)
    else:
        default = project_path(cfg["paths"]["final"]) / "kdart_qa_gpt_quota.jsonl"
        in_path = default if default.exists() else \
            project_path(cfg["paths"]["final"]) / cfg["export"]["jsonl_filename"]
    if not in_path.exists():
        log.error("Items file not found: %s", in_path)
        return 1

    top_k = args.top_k if args.top_k is not None else cfg["rag"]["top_k"]
    items = load_items(in_path)
    log.info("Evaluating retrieval recall @ k=%d on %d items (%s, filter=%s)",
             top_k, len(items), in_path.name, "off" if args.no_filter else "company+year")

    index = RagIndex(
        persist_dir=project_path(cfg["paths"]["index"]),
        collection_name=cfg["rag"]["collection_name"],
        embedder=build_embedder(cfg["rag"]),
    )

    hit_total = 0
    by_type_hit: dict[str, int] = defaultdict(int)
    by_type_n: dict[str, int] = defaultdict(int)
    misses: list[tuple[str, str, list[str]]] = []

    for item in items:
        gold = set(item.get("evidence_chunks", []))
        ttype = item.get("task_type", "?")
        by_type_n[ttype] += 1

        where = None if args.no_filter else build_where(item)
        hits = index.query(item["question"], top_k=top_k, where=where)
        if not hits and where is not None:
            hits = index.query(item["question"], top_k=top_k)  # mirror calibrator fallback
        retrieved = {h["chunk_id"] for h in hits}

        # "Recall" = gold evidence chunk(s) present in top-k. Items have ≥1 gold
        # chunk; we count a hit when ALL gold chunks are retrieved (strict). With
        # single-chunk items this is the standard "gold in top-k".
        if gold and gold.issubset(retrieved):
            hit_total += 1
            by_type_hit[ttype] += 1
        else:
            misses.append((item["id"], ttype, sorted(gold - retrieved)))

    n = len(items)
    overall = hit_total / n if n else 0.0
    log.info("=" * 60)
    log.info("Retrieval recall @ k=%d", top_k)
    log.info("=" * 60)
    log.info("Overall: %d/%d = %.3f", hit_total, n, overall)
    log.info("By task_type:")
    for t in sorted(by_type_n):
        h, tot = by_type_hit[t], by_type_n[t]
        log.info("  %-4s: %2d/%2d = %.3f", t, h, tot, h / tot if tot else 0.0)

    if misses:
        log.info("Missed %d item(s):", len(misses))
        if args.show_misses:
            for iid, ttype, missing in misses:
                log.info("  %s [%s] missing gold: %s", iid, ttype, missing)
        else:
            log.info("  %s", [m[0] for m in misses])

    log.info("→ target recall ≥ 0.90 %s", "PASS ✅" if overall >= 0.90 else "FAIL ❌")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
