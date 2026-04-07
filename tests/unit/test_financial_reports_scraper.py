"""Unit tests for unified financial reports scraper helpers."""

from backend.quant_pro.data_scrapers.financial_reports import (
    classify_report_type,
    parse_report_period,
)


def test_classify_report_type_variants():
    assert classify_report_type(
        "Nabil Bank Limited has published its provisional financial statement for the second quarter of the fiscal year 2082/83"
    ) == "quarterly"
    assert classify_report_type(
        "Upper Lohore Khola Hydropower Company Limited has published its annual report for the fiscal year 2081/82"
    ) == "annual"
    assert classify_report_type(
        "Sample Company has published its monthly financial statement for Ashad 2082"
    ) == "other_financial"
    assert classify_report_type("Bonus shares approved at AGM") is None


def test_parse_report_period_quarterly_and_annual():
    quarterly = parse_report_period(
        "Nabil Bank Limited has published its provisional financial statement for the second quarter of the fiscal year 2082/83"
    )
    assert quarterly["fiscal_year"] == "082-083"
    assert quarterly["quarter"] == 2
    assert quarterly["period_label"] == "Q2"

    annual = parse_report_period(
        "Upper Lohore Khola Hydropower Company Limited has published its annual report for the fiscal year 2081/82"
    )
    assert annual["fiscal_year"] == "081-082"
    assert annual["quarter"] is None
    assert annual["period_label"] == "FY"


def test_parse_report_period_handles_q_format():
    result = parse_report_period("ABC Microfinance has published Q3 report for fiscal year 082-083")
    assert result["fiscal_year"] == "082-083"
    assert result["quarter"] == 3
    assert result["period_label"] == "Q3"
