"""Unit tests for local OCR parsing helpers."""

from backend.quant_pro.local_financial_ocr import (
    _extract_from_lines,
    _merge_extractions,
    _normalize_number_token,
)


def test_normalize_number_token_handles_devanagari_digits_and_negatives():
    assert _normalize_number_token("१,२३४.५०") == 1234.50
    assert _normalize_number_token("(५,६७८.९०)") == -5678.90


def test_extract_from_lines_pulls_core_fields_from_statement_text():
    text = """
    Revenue from sale of Electricity 6,076,217,538.90 3,855,149,069.40
    Net Profit 996,539,153.34 1,674,689,000.47
    Earnings per share 5.79 6.91
    Book value per share 340.88 409.36
    """

    values, scores, matched_fields, flags = _extract_from_lines(text)

    assert values["income_statement"]["total_revenue"] == 6_076_217_538.90
    assert values["income_statement"]["net_profit"] == 996_539_153.34
    assert values["per_share"]["eps"] == 5.79
    assert values["per_share"]["book_value"] == 340.88
    assert "income_statement.total_revenue" in matched_fields
    assert "per_share.eps" in matched_fields
    assert scores["income_statement.total_revenue"] >= 0.72
    assert flags == []


def test_merge_extractions_prefers_row_values_and_flags_large_gaps():
    line_values = {
        "balance_sheet": {"total_assets": 0, "total_liabilities": 0, "shareholders_equity": 0, "share_capital": 0, "retained_earnings": 0, "total_deposits": 0, "total_loans": 0},
        "income_statement": {"total_revenue": 1000, "operating_profit": 0, "net_profit": 100, "interest_income": 0, "interest_expense": 0},
        "per_share": {"eps": 10, "book_value": 200},
        "ratios": {"npl_pct": 0, "capital_adequacy_pct": 0, "cost_income_ratio": 0},
    }
    row_values = {
        "balance_sheet": {"total_assets": 0, "total_liabilities": 0, "shareholders_equity": 0, "share_capital": 0, "retained_earnings": 0, "total_deposits": 0, "total_loans": 0},
        "income_statement": {"total_revenue": 1030, "operating_profit": 0, "net_profit": 140, "interest_income": 0, "interest_expense": 0},
        "per_share": {"eps": 9.8, "book_value": 205},
        "ratios": {"npl_pct": 0, "capital_adequacy_pct": 0, "cost_income_ratio": 0},
    }

    merged, flags, chosen_source = _merge_extractions(
        line_values,
        {
            "income_statement.total_revenue": 0.80,
            "income_statement.net_profit": 0.78,
            "per_share.eps": 0.90,
            "per_share.book_value": 0.88,
        },
        row_values,
        {
            "income_statement.total_revenue": 0.86,
            "income_statement.net_profit": 0.82,
            "per_share.eps": 0.91,
            "per_share.book_value": 0.90,
        },
    )

    assert merged["income_statement"]["total_revenue"] == 1030
    assert merged["income_statement"]["net_profit"] == 140
    assert merged["per_share"]["book_value"] == 205
    assert chosen_source["income_statement.total_revenue"] == "row"
    assert "inconsistent_value:income_statement.net_profit" in flags
