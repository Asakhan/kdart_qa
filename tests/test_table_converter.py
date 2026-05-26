from src.table_converter import html_table_to_markdown, split_html_into_blocks


def test_html_table_to_markdown_basic():
    html = """
    <table>
      <tr><th>부문</th><th>매출(억원)</th></tr>
      <tr><td>반도체</td><td>1,000</td></tr>
      <tr><td>모바일</td><td>500</td></tr>
    </table>
    """
    md = html_table_to_markdown(html)
    assert "|" in md
    assert "반도체" in md
    assert "1,000" in md


def test_split_html_into_blocks_preserves_table_block():
    html = """
    <p>도입 문장.</p>
    <table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table>
    <p>마무리 문장.</p>
    """
    blocks = split_html_into_blocks(html)
    kinds = [b[0] for b in blocks]
    assert "text" in kinds and "table" in kinds
    assert kinds.index("table") > kinds.index("text")  # order preserved
    # Table comes back as markdown, not raw HTML.
    table_payload = [p for k, p in blocks if k == "table"][0]
    assert "<table" not in table_payload.lower()
    assert "|" in table_payload
