"""Extract the 4 target sections out of a DART report ZIP.

DART XML reports vary in structure across filings. We try two strategies in order:

  Strategy A (structured): walk SECTION-1 / SECTION-2 / TITLE elements typical of
  modern e-DRS XML filings.

  Strategy B (text scan): treat the document as flat text + tables and locate
  section boundaries with regex heading patterns. Used when SECTION-* markup
  is absent or non-conforming.

The extractor preserves <table> markup so the downstream chunker / table
converter can do its work.
"""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup, NavigableString, Tag

from .common import ensure_dir, get_logger

log = get_logger("section_extractor")

# Roman numeral and CJK numeral normalization for headings.
_ROMAN_MAP = {
    "Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "Ⅳ": "IV", "Ⅴ": "V",
    "Ⅵ": "VI", "Ⅶ": "VII", "Ⅷ": "VIII", "Ⅸ": "IX", "Ⅹ": "X",
    "ⅰ": "I", "ⅱ": "II", "ⅲ": "III", "ⅳ": "IV", "ⅴ": "V",
}

# Heading regex used by Strategy B. Top-level: "II. 사업의 내용", "Ⅲ. 재무...".
_TOP_HEADING_RE = re.compile(
    r"^\s*(?P<num>[IVX]{1,5})\s*[..．]\s*(?P<title>.{1,40})$"
)
# Sub-level: "5. 매출 및 수주상황", "1. 요약재무정보".
_SUB_HEADING_RE = re.compile(
    r"^\s*(?P<num>\d{1,2})\s*[..．]\s*(?P<title>.{1,40})$"
)


@dataclass
class SectionSpec:
    """One target section described in config.yaml."""
    id: str
    parent_patterns: list[str]
    title_patterns: list[str]

    @classmethod
    def from_dict(cls, d: dict) -> "SectionSpec":
        return cls(
            id=d["id"],
            parent_patterns=list(d["parent_patterns"]),
            title_patterns=list(d["title_patterns"]),
        )


def normalize_heading(text: str) -> str:
    """Strip whitespace, normalize roman numerals and CJK spaces."""
    if not text:
        return ""
    for k, v in _ROMAN_MAP.items():
        text = text.replace(k, v)
    # Full-width to half-width period
    text = text.replace("．", ".").replace("．", ".")
    return re.sub(r"\s+", " ", text).strip()


def heading_matches(text: str, patterns: Iterable[str]) -> bool:
    """Lenient containment match after normalization (spaces removed)."""
    norm = normalize_heading(text).replace(" ", "")
    for p in patterns:
        if p.replace(" ", "") in norm:
            return True
    return False


# ---------- Zip / XML loading ----------

def iter_zip_xml(zip_path: Path) -> Iterable[tuple[str, bytes]]:
    """Yield (member_name, raw_bytes) for every .xml inside a DART report ZIP."""
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".xml"):
                yield name, zf.read(name)


def load_soup(raw: bytes) -> BeautifulSoup:
    """Parse XML/HTML bytes with the most forgiving parser available."""
    # Many DART files declare encoding inside; let BS auto-detect.
    try:
        return BeautifulSoup(raw, "lxml-xml")
    except Exception:
        return BeautifulSoup(raw, "html.parser")


# ---------- Strategy A: structured SECTION-* walk ----------

def _title_text(node: Tag) -> str:
    title = node.find("TITLE")
    if title is None:
        # Some templates put it as <title> or as the first child line.
        title = node.find("title")
    return normalize_heading(title.get_text(" ", strip=True)) if title else ""


def _section_inner(node: Tag) -> str:
    """Return inner content of a SECTION-* node minus its own TITLE."""
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            s = str(child).strip()
            if s:
                parts.append(s)
            continue
        if isinstance(child, Tag) and child.name and child.name.lower() == "title":
            continue
        parts.append(str(child))
    return "\n".join(parts)


def extract_structured(soup: BeautifulSoup, spec: SectionSpec) -> str | None:
    """Try to find spec inside a structured SECTION-1 / SECTION-2 tree."""
    top_nodes = soup.find_all(re.compile(r"^SECTION-1$", re.I))
    if not top_nodes:
        return None
    for top in top_nodes:
        if not heading_matches(_title_text(top), spec.parent_patterns):
            continue
        for sub in top.find_all(re.compile(r"^SECTION-2$", re.I)):
            if heading_matches(_title_text(sub), spec.title_patterns):
                return _section_inner(sub)
    return None


# ---------- Strategy B: linear scan over the visible flow ----------

def _flatten_blocks(soup: BeautifulSoup) -> list[tuple[str, str]]:
    """Return a flat sequence of (kind, html_or_text) blocks in document order.

    kind ∈ {"heading", "para", "table", "image"}.
    Headings are detected by their text matching _TOP_HEADING_RE or _SUB_HEADING_RE.
    """
    blocks: list[tuple[str, str]] = []
    # We look at TABLE elements and everything else as a flat text stream.
    body = soup.find("DOCUMENT-BODY") or soup.find("body") or soup
    for el in body.descendants:
        if not isinstance(el, Tag):
            continue
        name = (el.name or "").lower()
        if name == "table":
            # Avoid emitting nested tables twice — only top-level tables.
            if el.find_parent("table"):
                continue
            blocks.append(("table", str(el)))
            continue
        if name == "image":
            blocks.append(("image", str(el)))
            continue
        # Heading-ish: TITLE, P, span containing only short text matching heading regex.
        if name in {"title", "p", "h1", "h2", "h3", "h4"}:
            text = el.get_text(" ", strip=True)
            if not text:
                continue
            n = normalize_heading(text)
            kind = "para"
            if _TOP_HEADING_RE.match(n) or _SUB_HEADING_RE.match(n):
                kind = "heading"
            blocks.append((kind, n if kind == "heading" else str(el)))
    return blocks


def extract_textual(soup: BeautifulSoup, spec: SectionSpec) -> str | None:
    """Locate spec by scanning the flat flow for parent + sub headings."""
    blocks = _flatten_blocks(soup)
    in_parent = False
    in_sub = False
    out: list[str] = []
    for kind, payload in blocks:
        if kind != "heading":
            if in_sub:
                out.append(payload)
            continue
        # heading
        if _TOP_HEADING_RE.match(payload):
            # Entering a new top-level section terminates any open sub.
            if heading_matches(payload, spec.parent_patterns):
                in_parent = True
                in_sub = False
            else:
                if in_sub:
                    break  # left the target subsection
                in_parent = False
                in_sub = False
            continue
        if _SUB_HEADING_RE.match(payload):
            if in_sub and not heading_matches(payload, spec.title_patterns):
                break  # next sibling subsection — done
            if in_parent and heading_matches(payload, spec.title_patterns):
                in_sub = True
            continue
    return "\n".join(out) if out else None


# ---------- Public API ----------

@dataclass
class ExtractionResult:
    spec_id: str
    found: bool
    html: str | None
    source_file: str | None  # name of the XML member inside the ZIP
    strategy: str | None     # "structured" or "textual"


def extract_sections_from_zip(
    zip_path: Path, specs: list[SectionSpec]
) -> list[ExtractionResult]:
    """For each spec, return the first successful extraction across XML members."""
    results: dict[str, ExtractionResult] = {
        s.id: ExtractionResult(spec_id=s.id, found=False, html=None,
                               source_file=None, strategy=None)
        for s in specs
    }

    for member, raw in iter_zip_xml(zip_path):
        try:
            soup = load_soup(raw)
        except Exception as e:
            log.warning("Could not parse %s in %s: %s", member, zip_path.name, e)
            continue
        for spec in specs:
            if results[spec.id].found:
                continue
            html = extract_structured(soup, spec)
            strategy = "structured"
            if html is None:
                html = extract_textual(soup, spec)
                strategy = "textual"
            if html:
                results[spec.id] = ExtractionResult(
                    spec_id=spec.id, found=True, html=html,
                    source_file=member, strategy=strategy,
                )

    return list(results.values())


def save_results(
    results: list[ExtractionResult],
    out_dir: Path,
    *,
    company: str,
    year: int,
) -> list[Path]:
    """Write each successful extraction to out_dir as both .html and .txt."""
    ensure_dir(out_dir)
    written: list[Path] = []
    for r in results:
        if not r.found or r.html is None:
            log.warning("Section %s not found for %s %s", r.spec_id, company, year)
            continue
        html_path = out_dir / f"{company}_{year}_{r.spec_id}.html"
        html_path.write_text(r.html, encoding="utf-8")
        # plain-text variant: strip tags but keep table layout sketch
        txt = BeautifulSoup(r.html, "html.parser").get_text("\n", strip=True)
        txt_path = out_dir / f"{company}_{year}_{r.spec_id}.txt"
        txt_path.write_text(txt, encoding="utf-8")
        written.extend([html_path, txt_path])
    return written
