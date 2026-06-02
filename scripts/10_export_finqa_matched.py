#!/usr/bin/env python3
"""Work 3: export the FinQA difficulty-matched subset.

Carves a reasoning-hop-aligned subset (hops ≤ --max-hops, default 2) out of an
existing final dataset so the verification experiment can separate the *language*
effect from the *reasoning-depth* effect when comparing against FinQA (whose
items are mostly 1–2 step). Each emitted item gains `difficulty_group` and
`finqa_matched` labels. The source file is left unchanged — this only adds a new
file (academic-honesty rule). Runs fully offline; no API/index.

    python scripts/10_export_finqa_matched.py
    python scripts/10_export_finqa_matched.py --max-hops 2 \
        --input data/final/kdart_qa_gpt_quota.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import ensure_dir, get_logger, load_config, project_path  # noqa: E402
from src.exporter import build_finqa_matched, finqa_difficulty_group  # noqa: E402

log = get_logger("export_finqa_matched", "phase4_export_finqa_matched.log")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=None,
                   help="Source final JSONL (default: data/final/kdart_qa_gpt_quota.jsonl).")
    p.add_argument("--out", default=None,
                   help="Output JSONL (default: data/final/kdart_qa_finqa_matched.jsonl).")
    p.add_argument("--max-hops", type=int, default=2,
                   help="Keep items with reasoning_hops ≤ this (default: 2).")
    return p.parse_args()


def load_items(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> int:
    args = parse_args()
    cfg = load_config()
    final_dir = ensure_dir(project_path(cfg["paths"]["final"]))

    in_path = Path(args.input) if args.input else final_dir / "kdart_qa_gpt_quota.jsonl"
    out_path = Path(args.out) if args.out else final_dir / "kdart_qa_finqa_matched.jsonl"
    if not in_path.exists():
        log.error("Input not found: %s", in_path)
        return 1

    items = load_items(in_path)
    subset = build_finqa_matched(items, max_hops=args.max_hops)

    full_groups = Counter(finqa_difficulty_group(it.get("reasoning_hops")) for it in items)
    sub_types = Counter(it.get("task_type") for it in subset)

    log.info("Source: %d items (%s)", len(items), in_path.name)
    log.info("Difficulty groups (full set): %s", dict(full_groups))
    log.info("FinQA-matched subset (hops ≤ %d): %d items", args.max_hops, len(subset))
    log.info("Subset task_type distribution: %s", dict(sub_types))

    with out_path.open("w", encoding="utf-8") as f:
        for it in subset:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    log.info("Wrote %d items → %s", len(subset), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
