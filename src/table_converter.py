"""Convert DART <table> markup to GitHub-flavored Markdown."""
from __future__ import annotations

import io
from typing import Iterable

import pandas as pd
from bs4 import BeautifulSoup, Tag

from .common import get_logger

log = get_logger("table_converter")


def _clean_cell(s: object) -> str:
    """Normalize a cell value to a single-line string."""
    if pd.isna(s):
        return ""
    s = str(s).replace("\n", " ").replace("\r", " ").replace("|", "/")
    return " ".join(s.split())


def html_table_to_markdown(html: str) -> str:
    """Convert one <table>...</table> HTML snippet to Markdown.

    Falls back to a BeautifulSoup walker if pandas can't parse the table
    (rowspan/colspan-heavy DART tables sometimes trip read_html)."""
    try:
        # read_html returns a list of DataFrames; pick the first / largest.
        # `thousands="\x00"` prevents pandas from silently turning "1,000"
        # into the integer 1000 — we want the human-readable cell preserved.
        dfs = pd.read_html(io.StringIO(html), flavor="lxml", thousands="\x00")
    except Exception as e:
        log.debug("pandas failed (%s); falling back to manual converter", e)
        return _bs4_table_to_markdown(html)
    if not dfs:
        return _bs4_table_to_markdown(html)
    df = max(dfs, key=lambda d: d.size)
    # pandas 2.1+ renamed DataFrame.applymap → DataFrame.map.
    df = df.map(_clean_cell) if hasattr(df, "map") and callable(getattr(df, "map")) else df.applymap(_clean_cell)
    # Drop empty header rows pandas sometimes invents.
    if all(isinstance(c, int) for c in df.columns):
        df.columns = [f"col{i}" for i in df.columns]
    try:
        return df.to_markdown(index=False)
    except Exception:
        return _bs4_table_to_markdown(html)


def _bs4_table_to_markdown(html: str) -> str:
    """Fallback: BS4-only table → markdown for tricky DART tables."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return ""
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [_clean_cell(td.get_text(" ", strip=True))
                 for td in tr.find_all(["th", "td"])]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header = rows[0]
    body = rows[1:] if len(rows) > 1 else []
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] * width) + " |"]
    for r in body:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def split_html_into_blocks(html: str) -> list[tuple[str, str]]:
    """Walk an HTML/XML fragment and return ordered (kind, payload) blocks.

    kind ∈ {"text", "table"}. Tables are returned as already-rendered Markdown.
    Adjacent text fragments are merged.
    """
    soup = BeautifulSoup(html, "html.parser")
    blocks: list[tuple[str, str]] = []
    text_buf: list[str] = []

    def flush_text() -> None:
        if text_buf:
            joined = " ".join(s for s in text_buf if s)
            joined = " ".join(joined.split())
            if joined:
                blocks.append(("text", joined))
            text_buf.clear()

    def walk(node) -> None:
        if isinstance(node, Tag):
            name = (node.name or "").lower()
            if name == "table":
                flush_text()
                md = html_table_to_markdown(str(node))
                if md:
                    blocks.append(("table", md))
                return
            for child in node.children:
                walk(child)
            # add a paragraph break for block-level elements
            if name in {"p", "br", "div", "li", "tr", "title"}:
                text_buf.append("\n")
        else:
            s = str(node)
            if s.strip():
                text_buf.append(s.strip())

    walk(soup)
    flush_text()
    return blocks
