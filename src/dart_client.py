"""Thin client over DART OpenAPI.

Endpoints we use:
  GET /api/corpCode.xml                   — corp_code ↔ stock_code map (ZIP of XML)
  GET /api/list.json                      — disclosure list filtered by corp_code/date/type
  GET /api/document.xml?rcept_no=...      — full report document (ZIP of XBRL/XML)

The client deliberately keeps its surface small and pushes parsing concerns
(corp-code resolution, list filtering by report name) up to the caller, so
this module is easy to unit-test without API access.
"""
from __future__ import annotations

import io
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .common import Company, ensure_dir, get_logger, require_env

log = get_logger("dart_client")

DART_BASE = "https://opendart.fss.or.kr/api"

# Map our preferred report_code to the DART pblntf_detail_ty used by list.json.
REPORT_CODE_TO_PBLNTF: dict[str, str] = {
    "11011": "A001",  # 사업보고서
    "11012": "A002",  # 반기보고서
    "11013": "A003",  # 분기보고서
}


class DartError(RuntimeError):
    """Raised when DART returns a non-success status_code in its JSON envelope."""


@dataclass
class DisclosureItem:
    """One row from DART list.json."""
    rcept_no: str
    corp_code: str
    corp_name: str
    report_nm: str
    rcept_dt: str  # YYYYMMDD


class RateLimiter:
    """Simple thread-safe sliding-window limiter (calls per minute)."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._lock = threading.Lock()
        self._timestamps: list[float] = []

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            cutoff = now - 60.0
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            if len(self._timestamps) >= self.per_minute:
                sleep_for = 60.0 - (now - self._timestamps[0]) + 0.05
                if sleep_for > 0:
                    log.info("Rate limit reached; sleeping %.2fs", sleep_for)
                    time.sleep(sleep_for)
                now = time.monotonic()
                cutoff = now - 60.0
                self._timestamps = [t for t in self._timestamps if t > cutoff]
            self._timestamps.append(now)


class DartClient:
    """Polite DART HTTP client with retries + sliding-window rate limit."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        rate_limit_per_minute: int = 200,
        timeout_seconds: int = 30,
        max_retries: int = 5,
    ) -> None:
        self.api_key = api_key or require_env("DART_API_KEY")
        self.timeout = timeout_seconds
        self._limiter = RateLimiter(rate_limit_per_minute)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "kdart-qa/0.1 (+research)"})
        self._max_retries = max_retries

    # ---------- low-level GET ----------

    def _do_get(self, path: str, params: dict[str, Any]) -> requests.Response:
        @retry(
            reraise=True,
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1.5, min=1, max=20),
            retry=retry_if_exception_type(
                (requests.ConnectionError, requests.Timeout, DartError)
            ),
        )
        def _call() -> requests.Response:
            self._limiter.acquire()
            full = {"crtfc_key": self.api_key, **params}
            log.debug("GET %s/%s params=%s", DART_BASE, path, {k: v for k, v in full.items() if k != "crtfc_key"})
            resp = self._session.get(f"{DART_BASE}/{path}", params=full, timeout=self.timeout)
            if resp.status_code >= 500:
                raise DartError(f"DART server {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            return resp

        return _call()

    # ---------- corpCode.xml ----------

    def fetch_corp_code_zip(self, dest: Path) -> Path:
        """Download and persist the corpCode mapping ZIP."""
        ensure_dir(dest.parent)
        resp = self._do_get("corpCode.xml", {})
        dest.write_bytes(resp.content)
        log.info("Saved corpCode ZIP to %s (%d bytes)", dest, dest.stat().st_size)
        return dest

    @staticmethod
    def parse_corp_code_zip(zip_path: Path) -> list[dict[str, str]]:
        """Return list of {corp_code, corp_name, stock_code, modify_date}."""
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
            if not names:
                raise DartError(f"No XML inside corpCode zip: {zip_path}")
            with zf.open(names[0]) as f:
                tree = ET.parse(f)
        rows: list[dict[str, str]] = []
        for el in tree.iter("list"):
            rows.append(
                {
                    "corp_code": (el.findtext("corp_code") or "").strip(),
                    "corp_name": (el.findtext("corp_name") or "").strip(),
                    "stock_code": (el.findtext("stock_code") or "").strip(),
                    "modify_date": (el.findtext("modify_date") or "").strip(),
                }
            )
        return rows

    def resolve_corp_code(self, stock_code: str, corp_map: list[dict[str, str]]) -> str | None:
        """Return DART corp_code for a 6-digit listed stock_code, if listed."""
        for row in corp_map:
            if row["stock_code"] == stock_code:
                return row["corp_code"]
        return None

    # ---------- list.json ----------

    def list_disclosures(
        self,
        *,
        corp_code: str,
        bgn_de: str,
        end_de: str,
        pblntf_detail_ty: str | None = None,
        page_count: int = 100,
    ) -> list[DisclosureItem]:
        """Iterate all pages of /list.json for a corp×date range."""
        items: list[DisclosureItem] = []
        page_no = 1
        while True:
            params: dict[str, Any] = {
                "corp_code": corp_code,
                "bgn_de": bgn_de,
                "end_de": end_de,
                "page_no": page_no,
                "page_count": page_count,
            }
            if pblntf_detail_ty:
                params["pblntf_detail_ty"] = pblntf_detail_ty
            resp = self._do_get("list.json", params)
            body = resp.json()
            status = body.get("status")
            if status == "013":
                # 013 = "조회된 데이터가 없습니다" — treat as empty, not error.
                break
            if status != "000":
                raise DartError(f"list.json error {status}: {body.get('message')}")
            for row in body.get("list", []):
                items.append(
                    DisclosureItem(
                        rcept_no=row["rcept_no"],
                        corp_code=row["corp_code"],
                        corp_name=row["corp_name"],
                        report_nm=row["report_nm"],
                        rcept_dt=row.get("rcept_dt", ""),
                    )
                )
            total_page = body.get("total_page", 1)
            if page_no >= total_page:
                break
            page_no += 1
        return items

    # ---------- document.xml ----------

    def download_document(self, rcept_no: str, dest_zip: Path) -> Path:
        """Download the report ZIP (XBRL/XML) and save it to disk."""
        ensure_dir(dest_zip.parent)
        resp = self._do_get("document.xml", {"rcept_no": rcept_no})
        ctype = resp.headers.get("Content-Type", "")
        # Successful payload is a ZIP (binary). An error envelope arrives as XML/JSON text.
        if "application/zip" not in ctype and not resp.content[:2] == b"PK":
            # Try to interpret as XML status payload for a useful error.
            try:
                root = ET.fromstring(resp.content)
                status = root.findtext("status") or "?"
                msg = root.findtext("message") or resp.text[:200]
                raise DartError(f"document.xml error {status}: {msg}")
            except ET.ParseError as e:
                raise DartError(f"Unexpected non-zip response: {resp.text[:200]}") from e
        dest_zip.write_bytes(resp.content)
        log.info("Saved document %s → %s (%d bytes)", rcept_no, dest_zip, dest_zip.stat().st_size)
        return dest_zip


# ---------- high-level helpers used by scripts/01_fetch_dart.py ----------

# Heuristics to recognize the right report from the verbose report_nm.
REPORT_NAME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "11011": ("사업보고서",),
    "11012": ("반기보고서",),
    "11013": ("분기보고서",),
}


def pick_report(
    items: Iterable[DisclosureItem], report_code: str
) -> DisclosureItem | None:
    """Among list.json hits, return the most recent matching report (handles 정정 filings)."""
    keys = REPORT_NAME_KEYWORDS[report_code]
    cands = [it for it in items if any(k in it.report_nm for k in keys)]
    if not cands:
        return None
    # The latest rcept_dt wins (handles 정정공시 — corrected disclosures).
    cands.sort(key=lambda it: it.rcept_dt, reverse=True)
    return cands[0]


def year_range(year: int) -> tuple[str, str]:
    """For a target fiscal year, fetch reports filed within the following year as well
    (사업보고서 for FY N is typically filed in Q1 of N+1)."""
    return f"{year}0101", f"{year + 1}0630"
