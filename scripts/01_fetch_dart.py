#!/usr/bin/env python3
"""Phase 1-1: Fetch DART business/quarterly reports for the K-DART-QA company list.

For each (company, year), prefer 사업보고서(11011) over 분기보고서(11013).
Persists:
  data/raw/_corpcode/CORPCODE.zip          — corp_code mapping (cached)
  data/raw/{company}_{year}_{code}.zip     — raw XBRL/XML report ZIP
  data/raw/manifest.json                   — what was fetched / what is missing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `src` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import (  # noqa: E402
    PROJECT_ROOT,
    Company,
    companies_from_config,
    ensure_dir,
    get_logger,
    load_config,
    project_path,
)
from src.dart_client import (  # noqa: E402
    REPORT_CODE_TO_PBLNTF,
    DartClient,
    DartError,
    pick_report,
    year_range,
)

log = get_logger("fetch_dart", "phase1_fetch.log")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--companies",
        nargs="*",
        help="Optional subset of company names to fetch.",
    )
    p.add_argument(
        "--years",
        nargs="*",
        type=int,
        help="Optional subset of years to fetch.",
    )
    p.add_argument(
        "--refresh-corpcode",
        action="store_true",
        help="Re-download corpCode.xml even if cached.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be fetched.",
    )
    return p.parse_args()


def ensure_corp_codes(
    client: DartClient, raw_dir: Path, refresh: bool
) -> list[dict[str, str]]:
    zip_path = raw_dir / "_corpcode" / "CORPCODE.zip"
    if refresh or not zip_path.exists():
        log.info("Downloading corpCode.xml …")
        client.fetch_corp_code_zip(zip_path)
    else:
        log.info("Using cached corpCode at %s", zip_path)
    return client.parse_corp_code_zip(zip_path)


def resolve_company_corp_code(
    client: DartClient, company: Company, corp_map: list[dict[str, str]]
) -> str:
    resolved = client.resolve_corp_code(company.stock, corp_map)
    if resolved and resolved != company.corp_code:
        log.warning(
            "corp_code mismatch for %s: config=%s, resolved-from-stock=%s — using resolved",
            company.name, company.corp_code, resolved,
        )
        return resolved
    if not resolved:
        log.warning(
            "No corp_code found for stock %s (%s); trusting config value %s",
            company.stock, company.name, company.corp_code,
        )
        return company.corp_code
    return resolved


def fetch_one(
    client: DartClient,
    company: Company,
    corp_code: str,
    year: int,
    preferred: list[str],
    raw_dir: Path,
    dry_run: bool,
) -> dict:
    """Try each preferred report code; return manifest row."""
    bgn, end = year_range(year)
    row = {
        "company": company.name,
        "stock": company.stock,
        "corp_code": corp_code,
        "year": year,
        "report_code": None,
        "rcept_no": None,
        "report_nm": None,
        "file": None,
        "status": "missing",
        "reason": None,
    }
    for code in preferred:
        try:
            items = client.list_disclosures(
                corp_code=corp_code,
                bgn_de=bgn,
                end_de=end,
                pblntf_detail_ty=REPORT_CODE_TO_PBLNTF[code],
            )
        except DartError as e:
            log.error("list.json failed for %s %s %s: %s", company.name, year, code, e)
            row["reason"] = f"list_error:{e}"
            continue

        picked = pick_report(items, code)
        if picked is None:
            log.info(
                "No %s found for %s %s (%d hits in window)",
                code, company.name, year, len(items),
            )
            continue

        row["report_code"] = code
        row["rcept_no"] = picked.rcept_no
        row["report_nm"] = picked.report_nm

        out_zip = raw_dir / f"{company.name}_{year}_{code}.zip"
        if out_zip.exists():
            log.info("Already on disk: %s", out_zip.name)
            row["file"] = str(out_zip.relative_to(PROJECT_ROOT))
            row["status"] = "cached"
            return row

        if dry_run:
            log.info("[dry-run] would download %s → %s", picked.rcept_no, out_zip.name)
            row["status"] = "dry_run"
            return row

        try:
            client.download_document(picked.rcept_no, out_zip)
            row["file"] = str(out_zip.relative_to(PROJECT_ROOT))
            row["status"] = "downloaded"
        except DartError as e:
            log.error("download failed for %s: %s", picked.rcept_no, e)
            row["reason"] = f"download_error:{e}"
            row["status"] = "error"
        return row

    row["reason"] = "no_matching_report_in_year"
    return row


def main() -> int:
    args = parse_args()
    cfg = load_config()
    raw_dir = ensure_dir(project_path(cfg["paths"]["raw"]))

    all_companies = companies_from_config(cfg)
    companies = [
        c for c in all_companies
        if not args.companies or c.name in args.companies
    ]
    years = args.years or cfg["dart"]["years"]
    preferred = cfg["dart"]["report_codes_preferred"]

    log.info(
        "Fetching %d companies × %d years (preferred order: %s)",
        len(companies), len(years), preferred,
    )

    client = DartClient(
        rate_limit_per_minute=cfg["dart"]["rate_limit_per_minute"],
        timeout_seconds=cfg["dart"]["request_timeout_seconds"],
        max_retries=cfg["dart"]["max_retries"],
    )
    corp_map = ensure_corp_codes(client, raw_dir, args.refresh_corpcode)

    manifest = []
    for company in companies:
        corp_code = resolve_company_corp_code(client, company, corp_map)
        for year in years:
            row = fetch_one(client, company, corp_code, year, preferred, raw_dir, args.dry_run)
            manifest.append(row)

    manifest_path = raw_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Wrote manifest with %d rows → %s", len(manifest), manifest_path)

    # Summary
    by_status: dict[str, int] = {}
    for r in manifest:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    log.info("Summary: %s", by_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
