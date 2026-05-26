"""Offline tests for the section extractor's heading match + strategies."""
from __future__ import annotations

from src.section_extractor import (
    SectionSpec,
    extract_structured,
    extract_textual,
    heading_matches,
    load_soup,
    normalize_heading,
)


def test_normalize_heading_roman():
    assert normalize_heading("Ⅱ.  사업의   내용") == "II. 사업의 내용"
    # Full-width period is normalized to ASCII period; spacing not invented.
    assert normalize_heading("Ⅲ．재무에 관한 사항") == "III.재무에 관한 사항"


def test_heading_matches_collapses_whitespace():
    assert heading_matches("5. 매출 및 수주상황", ["매출 및 수주"])
    assert heading_matches("5.매출및수주상황", ["매출 및 수주"])
    assert not heading_matches("5. 매출 및 수주상황", ["연구개발"])


STRUCTURED_XML = """<?xml version='1.0' encoding='utf-8'?>
<DOCUMENT-BODY>
  <SECTION-1>
    <TITLE>II. 사업의 내용</TITLE>
    <SECTION-2>
      <TITLE>5. 매출 및 수주상황</TITLE>
      <P>2024년 3분기 매출액은 1,234억원이다.</P>
      <TABLE><tr><th>부문</th><th>매출</th></tr><tr><td>반도체</td><td>500</td></tr></TABLE>
    </SECTION-2>
    <SECTION-2>
      <TITLE>7. 주요계약 및 연구개발활동</TITLE>
      <P>연구개발비 99억원.</P>
    </SECTION-2>
  </SECTION-1>
</DOCUMENT-BODY>""".encode("utf-8")


def test_extract_structured_finds_target():
    soup = load_soup(STRUCTURED_XML)
    spec = SectionSpec(
        id="사업의내용_5_매출",
        parent_patterns=["사업의 내용"],
        title_patterns=["매출 및 수주"],
    )
    html = extract_structured(soup, spec)
    assert html is not None
    assert "1,234억원" in html
    assert "<table" in html.lower() or "<TABLE" in html


def test_extract_structured_misses_when_subsection_absent():
    soup = load_soup(STRUCTURED_XML)
    spec = SectionSpec(
        id="x",
        parent_patterns=["사업의 내용"],
        title_patterns=["조직 구조"],
    )
    assert extract_structured(soup, spec) is None


TEXTUAL_HTML = """<html><body>
<p>II. 사업의 내용</p>
<p>1. 사업의 개요</p>
<p>여기는 개요 텍스트.</p>
<p>5. 매출 및 수주상황</p>
<p>2024년 3분기 매출액은 1,234억원.</p>
<table><tr><th>부문</th><th>매출</th></tr><tr><td>반도체</td><td>500</td></tr></table>
<p>7. 주요계약 및 연구개발활동</p>
<p>다른 섹션 시작.</p>
</body></html>""".encode("utf-8")


def test_extract_textual_walks_flat_flow():
    soup = load_soup(TEXTUAL_HTML)
    spec = SectionSpec(
        id="사업의내용_5_매출",
        parent_patterns=["사업의 내용"],
        title_patterns=["매출 및 수주"],
    )
    html = extract_textual(soup, spec)
    assert html is not None
    assert "1,234억원" in html
    assert "다른 섹션 시작" not in html
