from src.data.ingestion.sec_content import (
    build_filing_url,
    html_to_text,
)


def test_build_filing_url():
    url = build_filing_url(
        cik="0000320193",
        accession_number="0000320193-26-000020",
        primary_document="aapl-20260627.htm",
    )

    assert (
        url
        == "https://www.sec.gov/Archives/edgar/data/"
        "320193/000032019326000020/aapl-20260627.htm"
    )


def test_build_filing_url_requires_fields():
    try:
        build_filing_url(
            cik="",
            accession_number="123",
            primary_document="filing.htm",
        )
    except Exception as exc:
        assert "CIK" in str(exc)
    else:
        raise AssertionError("Expected missing CIK to fail")


def test_html_to_text_removes_markup():
    html = """
    <html>
      <head><style>hidden {}</style></head>
      <body>
        <h1>Risk Factors</h1>
        <p>The company faces supply chain risks.</p>
        <script>alert('ignore')</script>
      </body>
    </html>
    """

    text = html_to_text(html)

    assert "Risk Factors" in text
    assert "supply chain risks" in text
    assert "<p>" not in text
    assert "alert" not in text


def test_html_to_text_decodes_entities():
    text = html_to_text(
        "<p>Revenue &amp; operating income&nbsp;increased.</p>"
    )

    assert "Revenue & operating income increased." in text
