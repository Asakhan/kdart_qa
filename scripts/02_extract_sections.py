#!/usr/bin/env python3
"""Phase 1-2: Extract target sections from each report in data/raw/.

Reads data/raw/manifest.json produced by 01_fetch_dart.py and writes per-section
HTML+TXT files into data/sections/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import (  # noqa: E402
    PROJECT_ROOT,
    ensure_dir,
    get_logger,
    load_config,
    project_path,
)
from src.section_extractor import (  # noqa: E402
    SectionSpec,
    extract_sections_from_zip,
    save_results,
)

log = get_logger("extract_sections", "phase1_extract.log")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default=None, help="Path to manifest.json (default: data/raw/manifest.json)")
    p.add_argument("--company", action="append", help="Filter by company name (repeatable).")
    p.add_argument("--year", action="append", type=int, help="Filter by year (repeatable).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    sections_dir = ensure_dir(project_path(cfg["paths"]["sections"]))
    manifest_path = Path(args.manifest) if args.manifest else project_path(cfg["paths"]["raw"]) / "manifest.json"
    if not manifest_path.exists():
        log.error("Manifest not found: %s. Run scripts/01_fetch_dart.py first.", manifest_path)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    specs = [SectionSpec.from_dict(s) for s in cfg["sections"]]
    log.info("Loaded %d section specs", len(specs))

    extraction_log: list[dict] = []
    for row in manifest:
        if row["status"] not in {"downloaded", "cached"}:
            continue
        if args.company and row["company"] not in args.company:
            continue
        if args.year and row["year"] not in args.year:
            continue

        zip_path = PROJECT_ROOT / row["file"]
        if not zip_path.exists():
            log.error("Missing ZIP file (manifest says present): %s", zip_path)
            continue

        log.info("Extracting %s %s …", row["company"], row["year"])
        results = extract_sections_from_zip(zip_path, specs)
        save_results(results, sections_dir, company=row["company"], year=row["year"])

        for r in results:
            extraction_log.append(
                {
                    "company": row["company"],
                    "year": row["year"],
                    "spec_id": r.spec_id,
                    "found": r.found,
                    "strategy": r.strategy,
                    "source_file": r.source_file,
                }
            )

    log_path = sections_dir / "extraction_log.json"
    log_path.write_text(json.dumps(extraction_log, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Extraction log → %s", log_path)

    # Summary
    total = len(extraction_log)
    found = sum(1 for r in extraction_log if r["found"])
    log.info("Extracted %d / %d (success rate %.1f%%)", found, total, 100 * found / max(total, 1))
    by_strategy: dict[str, int] = {}
    for r in extraction_log:
        if r["found"]:
            by_strategy[r["strategy"]] = by_strategy.get(r["strategy"], 0) + 1
    log.info("By strategy: %s", by_strategy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
