from src.chunker import Chunker, chunk_section_html


def test_chunk_short_text_yields_one_chunk():
    chunker = Chunker(chunk_tokens=512, overlap_tokens=50)
    html = "<p>" + "삼성전자 2024년 3분기 매출액은 1조원이다. " * 5 + "</p>"
    chunks = chunk_section_html(
        html, company="삼성전자", year=2024,
        section="재무_1_요약", report_code="11013", chunker=chunker,
    )
    assert len(chunks) == 1
    assert chunks[0].company == "삼성전자"
    assert chunks[0].token_count <= 512


def test_chunk_long_text_splits_with_overlap():
    chunker = Chunker(chunk_tokens=64, overlap_tokens=10)
    sentence = "테스트 문장입니다. " * 200
    html = f"<p>{sentence}</p>"
    chunks = chunk_section_html(
        html, company="X", year=2024, section="s", report_code=None, chunker=chunker,
    )
    assert len(chunks) >= 2
    for c in chunks:
        assert c.token_count <= 64 + 10  # may include overlap


def test_table_kept_intact():
    chunker = Chunker(chunk_tokens=64, overlap_tokens=10)
    html = (
        "<p>도입.</p>"
        "<table>"
        + "".join(f"<tr><td>{i}</td><td>값{i}</td></tr>" for i in range(20))
        + "</table>"
        "<p>마무리.</p>"
    )
    chunks = chunk_section_html(
        html, company="X", year=2024, section="s", report_code=None, chunker=chunker,
    )
    table_chunks = [c for c in chunks if c.has_table]
    assert len(table_chunks) >= 1
    # The table sits in a single chunk — none of the table rows are missing.
    table_text = "\n".join(c.text for c in table_chunks)
    for i in range(20):
        assert f"값{i}" in table_text
