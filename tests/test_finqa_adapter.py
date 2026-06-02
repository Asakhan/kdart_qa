"""Unit tests for the FinQA-schema adapter (Work 3.5)."""
from __future__ import annotations

from src.finqa_adapter import (
    build_gold_inds,
    compute_exe_ans,
    convert_item,
    parse_markdown_table,
    segment_blocks,
    split_sentences,
    validate,
)


# ---- markdown table → 2-D array ----

def test_parse_markdown_table_basic():
    md = (
        "| 부문 | 매출액 | 비중 |\n"
        "|:-----|------:|:----:|\n"
        "| DX   | 1,824 | 60.4% |\n"
        "| DS   | 984   | 32.6% |"
    )
    table = parse_markdown_table(md)
    assert table[0] == ["부문", "매출액", "비중"]   # header kept
    assert table[1] == ["DX", "1,824", "60.4%"]     # separator row dropped
    assert table[2] == ["DS", "984", "32.6%"]
    assert {len(r) for r in table} == {3}            # rectangular


def test_parse_markdown_table_pads_ragged_rows():
    md = "| a | b | c |\n|---|---|---|\n| 1 | 2 |"
    table = parse_markdown_table(md)
    assert table[1] == ["1", "2", ""]               # padded to width 3


def test_parse_markdown_table_empty():
    assert parse_markdown_table("just prose, no pipes") == []


# ---- segmenting text vs table ----

def test_segment_blocks_separates_text_and_table():
    text = (
        "서두 문장입니다.\n\n"
        "| col0 | col1 |\n|:-----|:-----|\n| a | b |\n\n"
        "마무리 문장입니다."
    )
    blocks = segment_blocks(text)
    kinds = [k for k, _ in blocks]
    assert kinds == ["text", "table", "text"]
    assert "서두" in blocks[0][1]
    assert blocks[1][1].startswith("|")
    assert "마무리" in blocks[2][1]


def test_split_sentences():
    s = split_sentences("첫째 문장. 둘째 문장! 셋째 문장?")
    assert len(s) == 3


# ---- gold_inds key convention ----

def test_gold_inds_table_vs_text_keys():
    quotes = ["종속회사수 | 100", "본문 근거 문장"]
    gi = build_gold_inds(quotes, table=[["h"]], pre=[], post=[])
    assert gi["table_0"] == "종속회사수 | 100"
    assert gi["text_0"] == "본문 근거 문장"


def test_gold_inds_fallback_to_first_table_row():
    gi = build_gold_inds([], table=[["h1", "h2"], ["v1", "v2"]], pre=["p"], post=[])
    assert gi == {"table_0": "v1 | v2"}


def test_gold_inds_fallback_to_first_sentence():
    gi = build_gold_inds([], table=[], pre=["첫 문장", "둘째"], post=[])
    assert gi == {"text_0": "첫 문장"}


# ---- exe_ans parsing ----

def test_compute_exe_ans_numeric():
    assert compute_exe_ans("1,234억원") == 1234.0
    assert compute_exe_ans("12.3%") == 12.3
    assert compute_exe_ans(100) == 100.0


def test_compute_exe_ans_non_numeric_keeps_string():
    assert compute_exe_ans("반도체 사업부") == "반도체 사업부"


# ---- full item conversion ----

def _chunk(cid, text):
    return {"chunk_id": cid, "text": text}


def test_convert_item_with_table():
    chunk_text = (
        "당사의 2022년 부문별 매출입니다.\n\n"
        "| 부문 | 매출액 |\n|:-----|------:|\n| DX | 1,824,897 |\n| 총계 | 3,022,314 |\n\n"
        "이상 매출 현황이었습니다."
    )
    item = {
        "id": "DART_T1_001",
        "question": "2022년 총 매출액은?",
        "gold_answer": "3,022,314",
        "gold_answer_unit": "억원",
        "evidence_chunks": ["삼성전자_2022_매출_000"],
        "evidence_quotes": ["총계 | 3,022,314"],
        "reasoning_steps": ["1. 표를 찾는다.", "2. 총계 행을 읽는다."],
        "reasoning_hops": 2,
        "task_type": "T1",
        "source_company": "삼성전자",
        "source_report": "2022_11011",
        "source_section": "매출",
    }
    lookup = {"삼성전자_2022_매출_000": _chunk("삼성전자_2022_매출_000", chunk_text)}
    out = convert_item(item, lookup)

    assert out["id"] == "DART_T1_001"
    assert out["pre_text"] and "2022년 부문별 매출" in out["pre_text"][0]
    assert out["post_text"] and "매출 현황" in out["post_text"][0]
    assert out["table"][0] == ["부문", "매출액"]
    assert ["총계", "3,022,314"] in out["table"]
    assert out["qa"]["program"] == ""
    assert out["qa"]["exe_ans"] == 3022314.0
    assert out["qa"]["answer"] == "3,022,314"
    assert out["qa"]["gold_inds"] == {"table_0": "총계 | 3,022,314"}
    assert "1. 표를 찾는다." in out["qa"]["explanation"]
    # original metadata preserved
    assert out["kdart_meta"]["task_type"] == "T1"
    assert out["kdart_meta"]["source_company"] == "삼성전자"


def test_convert_item_no_table_all_pretext():
    item = {
        "id": "X1",
        "question": "q?",
        "gold_answer": "없음",
        "evidence_chunks": ["c0"],
        "evidence_quotes": ["근거 문장입니다"],
        "reasoning_steps": [],
        "source_company": "X",
        "source_report": "2023_11011",
    }
    lookup = {"c0": _chunk("c0", "첫 문장입니다. 둘째 문장입니다.")}
    out = convert_item(item, lookup)
    assert out["table"] == []
    assert len(out["pre_text"]) == 2
    assert out["post_text"] == []
    # non-numeric gold → exe_ans keeps string
    assert out["qa"]["exe_ans"] == "없음"


def test_validate_reports_clean_dataset():
    item = {
        "id": "X1", "question": "q?", "gold_answer": "100",
        "evidence_chunks": ["c0"], "evidence_quotes": ["a | 100"],
        "reasoning_steps": ["s"], "source_company": "X", "source_report": "2023_11011",
    }
    lookup = {"c0": _chunk("c0", "| a | b |\n|---|---|\n| 100 | 200 |")}
    converted = [convert_item(item, lookup)]
    report = validate(converted)
    assert report["n"] == 1
    assert report["non_rectangular_tables"] == 0
    assert report["missing_exe_ans"] == 0
    assert report["issues"] == []
