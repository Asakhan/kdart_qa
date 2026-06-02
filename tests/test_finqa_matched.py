"""Unit tests for the FinQA difficulty-matched subset (Work 3)."""
from __future__ import annotations

from src.exporter import build_finqa_matched, finqa_difficulty_group


def test_difficulty_group_mapping():
    assert finqa_difficulty_group(1) == "1-2_step"
    assert finqa_difficulty_group(2) == "1-2_step"
    assert finqa_difficulty_group(3) == "3_step"
    assert finqa_difficulty_group(4) == "4plus_step"
    assert finqa_difficulty_group(None) == "unknown"


def test_build_finqa_matched_filters_and_labels():
    items = [
        {"id": "a", "reasoning_hops": 2},
        {"id": "b", "reasoning_hops": 3},
        {"id": "c", "reasoning_hops": 4},
        {"id": "d", "reasoning_hops": 1},
    ]
    subset = build_finqa_matched(items, max_hops=2)
    assert [i["id"] for i in subset] == ["a", "d"]
    assert all(i["finqa_matched"] for i in subset)
    assert subset[0]["difficulty_group"] == "1-2_step"


def test_build_finqa_matched_does_not_mutate_input():
    items = [{"id": "a", "reasoning_hops": 2}]
    build_finqa_matched(items)
    assert "difficulty_group" not in items[0]  # original untouched
    assert "finqa_matched" not in items[0]


def test_max_hops_override():
    items = [{"id": "x", "reasoning_hops": 3}]
    assert build_finqa_matched(items, max_hops=2) == []
    assert len(build_finqa_matched(items, max_hops=3)) == 1
