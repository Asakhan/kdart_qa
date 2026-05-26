#!/usr/bin/env python3
"""Phase 3-1 (A/B): difficulty calibration with gpt-4o-mini for comparison.

Same RAG + same prompt + same grading rule as 06_calibrate.py (Gemini).
Writes results to data/calibration/results_openai.json so the two runs
can be compared head-to-head.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibrator import Calibrator  # noqa: E402
from src.common import ensure_dir, get_logger, load_config, project_path  # noqa: E402
from src.rag_index import OpenAIEmbedder, RagIndex  # noqa: E402

log = get_logger("calibrate_openai", "phase3_calibrate_openai.log")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--input", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--yes", action="store_true")
    return p.parse_args()


def load_reviewed_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if "items" in data and isinstance(data["items"], list):
        return data["items"]
    if "states" in data:
        return [s["item"] for s in data["states"] if s.get("status") in {"accepted", "edited"}]
    raise ValueError("Unrecognized reviewed JSON shape")


def main() -> int:
    args = parse_args()
    cfg = load_config()

    in_path = Path(args.input) if args.input else project_path(cfg["paths"]["reviewed"]) / cfg["review"]["reviewed_filename"]
    out_dir = ensure_dir(project_path(cfg["paths"]["calibration"]))
    out_path = Path(args.out) if args.out else out_dir / "results_openai.json"

    if not in_path.exists():
        log.error("Reviewed file not found: %s", in_path)
        return 1
    items = load_reviewed_items(in_path)
    if not items:
        log.error("No reviewed items found in %s", in_path)
        return 1

    # gpt-4o-mini pricing (Jan 2026): input $0.15/1M, output $0.60/1M
    in_price = 0.15
    out_price = 0.60
    est_in = len(items) * cfg["calibration"]["attempts_per_question"] * 1500
    est_out = len(items) * cfg["calibration"]["attempts_per_question"] * 200
    est_usd = est_in / 1_000_000 * in_price + est_out / 1_000_000 * out_price
    log.info(
        "Calibration (%s) on %d items × %d attempts (est cost ≈ $%.4f)",
        args.model, len(items), cfg["calibration"]["attempts_per_question"], est_usd,
    )
    if not args.yes:
        ans = input(f"Proceed? (est ${est_usd:.4f}) [y/N] ").strip().lower()
        if ans not in {"y", "yes"}:
            return 0

    embedder = OpenAIEmbedder(model=cfg["rag"]["model"], batch_size=cfg["rag"]["embedding_batch_size"])
    index = RagIndex(
        persist_dir=project_path(cfg["paths"]["index"]),
        collection_name=cfg["rag"]["collection_name"],
        embedder=embedder,
    )

    calib = Calibrator(
        index=index,
        model_name=args.model,
        attempts=cfg["calibration"]["attempts_per_question"],
        rel_tol=cfg["calibration"]["numeric_relative_tolerance"],
        top_k=cfg["rag"]["top_k"],
        input_price_per_1m=in_price,
        output_price_per_1m=out_price,
        provider="openai",
    )

    results = []
    for i, item in enumerate(items, start=1):
        log.info("[%d/%d] %s", i, len(items), item["id"])
        cr = calib.calibrate(item)
        results.append(cr.to_dict())
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    by_class = Counter(r["classification"] for r in results)
    log.info("=" * 60)
    log.info("Calibration 결과 (%s)", args.model)
    log.info("=" * 60)
    log.info("Total: %d", len(results))
    for k in ("accepted", "too_easy", "too_hard"):
        cnt = by_class.get(k, 0)
        pct = cnt / max(len(results), 1) * 100
        log.info("  %-9s: %3d (%.1f%%)", k, cnt, pct)
    log.info("실측 API 비용 (%s): $%.4f", args.model, calib.estimate_usd())
    log.info("→ %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
