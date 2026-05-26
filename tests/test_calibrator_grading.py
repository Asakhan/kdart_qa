from src.calibrator import grade, parse_number


def test_parse_number_with_comma_and_unit():
    assert parse_number("1,234억원") == 1234.0
    assert parse_number("12.3%") == 12.3
    assert parse_number("없음") is None


def test_grade_numeric_within_tolerance():
    assert grade("1,235", "1234", rel_tol=0.01) is True
    assert grade("1300", "1234", rel_tol=0.01) is False


def test_grade_string_exact_after_normalize():
    assert grade("반도체 사업부", "반도체사업부", rel_tol=0.01) is True
    assert grade("디스플레이", "반도체", rel_tol=0.01) is False
