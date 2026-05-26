#!/usr/bin/env python3
"""Phase 3-2: build the final K-DART-QA dataset + statistics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import ensure_dir, get_logger, load_config, project_path  # noqa: E402
from src.exporter import (  # noqa: E402
    build_statistics_md,
    edit_rate_from_log,
    load_calibration,
    load_reviewed,
    select_accepted,
    summarize_calibration,
    write_jsonl,
    write_parquet,
)

log = get_logger("export", "phase3_export.log")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-size", type=int, default=40,
                   help="Warn if the final dataset has fewer items than this.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()

    reviewed_path = project_path(cfg["paths"]["reviewed"]) / cfg["review"]["reviewed_filename"]
    calib_path = project_path(cfg["paths"]["calibration"]) / cfg["calibration"]["results_filename"]
    changes_log = project_path(cfg["paths"]["reviewed"]) / cfg["review"]["changes_log"]
    final_dir = ensure_dir(project_path(cfg["paths"]["final"]))

    if not reviewed_path.exists():
        log.error("Reviewed file missing: %s", reviewed_path); return 1
    if not calib_path.exists():
        log.error("Calibration results missing: %s", calib_path); return 1

    reviewed = load_reviewed(reviewed_path)
    calib = load_calibration(calib_path)
    final = select_accepted(reviewed, calib)

    if len(final) < args.target_size:
        log.warning(
            "최종 데이터셋이 %d 문항으로 목표 %d에 미달합니다. "
            "추가 작성을 위해 Phase 2.5에서 수정 후 다시 calibration 하세요.",
            len(final), args.target_size,
        )

    jsonl_path = final_dir / cfg["export"]["jsonl_filename"]
    parquet_path = final_dir / cfg["export"]["parquet_filename"]
    stats_path = final_dir / cfg["export"]["stats_filename"]

    write_jsonl(final, jsonl_path)
    log.info("Wrote %d items → %s", len(final), jsonl_path)
    try:
        write_parquet(final, parquet_path)
        log.info("Wrote Parquet → %s", parquet_path)
    except Exception as e:
        log.error("Parquet write failed (%s). JSONL is still authoritative.", e)

    calib_summary = summarize_calibration(calib)
    edits = edit_rate_from_log(changes_log, n_total=len(reviewed))
    md = build_statistics_md(
        final,
        reviewed_count=len(reviewed),
        calib_summary=calib_summary,
        edits=edits,
    )
    stats_path.write_text(md, encoding="utf-8")
    log.info("Wrote stats → %s", stats_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
