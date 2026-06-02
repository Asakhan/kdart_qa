#!/usr/bin/env python3
"""Phase 3-1: difficulty calibration with gemini-2.5-flash.

Input:  data/reviewed/kdart_qa_reviewed_v1.json  (Phase 2.5)
Output: data/calibration/results.json
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
from src.rag_index import RagIndex, build_embedder  # noqa: E402

log = get_logger("calibrate", "phase3_calibrate.log")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=None, help="Override reviewed JSON path.")
    p.add_argument("--model", default=None, help="Override Gemini model name (default: config calibration.llm_model).")
    p.add_argument("--out", default=None, help="Override output results JSON path.")
    p.add_argument("--yes", action="store_true", help="Skip cost confirmation.")
    p.add_argument("--max-workers", type=int, default=None,
                   help="Parallel calibration workers (default: config calibration.max_workers).")
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
    out_path = Path(args.out) if args.out else out_dir / cfg["calibration"]["results_filename"]
    model_name = args.model or cfg["calibration"]["llm_model"]

    if not in_path.exists():
        log.error("Reviewed file not found: %s", in_path)
        return 1
    items = load_reviewed_items(in_path)
    if not items:
        log.error("No reviewed items found in %s", in_path)
        return 1

    # rough budget estimate: ~1500 input tokens * 3 attempts * N items
    est_in = len(items) * cfg["calibration"]["attempts_per_question"] * 1500
    est_out = len(items) * cfg["calibration"]["attempts_per_question"] * 200
    est_usd = (
        est_in / 1_000_000 * cfg["calibration"]["cost_per_1m_input_tokens_usd"]
        + est_out / 1_000_000 * cfg["calibration"]["cost_per_1m_output_tokens_usd"]
    )
    log.info(
        "Calibration (%s) on %d items × %d attempts (est cost ≈ $%.4f)",
        model_name, len(items), cfg["calibration"]["attempts_per_question"], est_usd,
    )
    if not args.yes:
        ans = input(f"Proceed? (est $%.4f) [y/N] " % est_usd).strip().lower()
        if ans not in {"y", "yes"}:
            return 0

    embedder = build_embedder(cfg["rag"])
    index = RagIndex(
        persist_dir=project_path(cfg["paths"]["index"]),
        collection_name=cfg["rag"]["collection_name"],
        embedder=embedder,
    )

    calib = Calibrator(
        index=index,
        model_name=model_name,
        attempts=cfg["calibration"]["attempts_per_question"],
        rel_tol=cfg["calibration"]["numeric_relative_tolerance"],
        top_k=cfg["rag"]["top_k"],
        input_price_per_1m=cfg["calibration"]["cost_per_1m_input_tokens_usd"],
        output_price_per_1m=cfg["calibration"]["cost_per_1m_output_tokens_usd"],
    )

    max_workers = args.max_workers if args.max_workers is not None else cfg["calibration"].get("max_workers", 4)
    log.info("Calibrating with %d parallel worker(s)", max_workers)

    done = {"n": 0}
    saved: dict[str, dict] = {}

    def on_result(cr, item):
        # Runs in the main thread as each result completes → save stays sequential.
        done["n"] += 1
        saved[cr.item_id] = cr.to_dict()
        log.info("[%d/%d] %s → %s", done["n"], len(items), cr.item_id, cr.classification)
        ordered = [saved[it["id"]] for it in items if it["id"] in saved]
        out_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")

    calibrations = calib.calibrate_many(items, max_workers=max_workers, on_result=on_result)
    results = [c.to_dict() for c in calibrations]

    # summary
    by_class = Counter(r["classification"] for r in results)
    log.info("=" * 60)
    log.info("Calibration 결과")
    log.info("=" * 60)
    log.info("Total: %d", len(results))
    for k in ("accepted", "too_easy", "too_hard"):
        cnt = by_class.get(k, 0)
        pct = cnt / max(len(results), 1) * 100
        log.info("  %-9s: %3d (%.1f%%)", k, cnt, pct)
    log.info("실측 API 비용 (gemini): $%.4f", calib.estimate_usd())

    too_easy = [r["item_id"] for r in results if r["classification"] == "too_easy"]
    too_hard = [r["item_id"] for r in results if r["classification"] == "too_hard"]
    if too_easy:
        log.info("too_easy: %s", too_easy)
    if too_hard:
        log.info("too_hard: %s", too_hard)
    log.info("→ %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
