from pathlib import Path

from src.exporter import edit_rate_from_log


def test_edit_rate_parses_log(tmp_path: Path):
    log = tmp_path / "changes.log"
    log.write_text(
        "2026-06-01 14:23:01 | DART_T4_001 | ACCEPT | -\n"
        "2026-06-01 14:25:30 | DART_T4_002 | EDIT | gold_answer 수정\n"
        "2026-06-01 14:28:14 | DART_T4_003 | REGENERATE | reasoning 단순\n"
        "2026-06-01 14:35:02 | DART_T4_005 | DISCARD | evidence 부족\n",
        encoding="utf-8",
    )
    s = edit_rate_from_log(log, n_total=4)
    assert s["accept"] == 1
    assert s["edit"] == 1
    assert s["regenerate"] == 1
    assert s["discard"] == 1
    assert s["edit_rate"] == 0.25


def test_edit_rate_missing_log_returns_zeros(tmp_path: Path):
    s = edit_rate_from_log(tmp_path / "nope.log", n_total=10)
    assert s["edit_rate"] == 0.0
