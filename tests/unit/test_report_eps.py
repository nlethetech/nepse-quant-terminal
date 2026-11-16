"""Quarterly report EPS is annualized exactly once (see PR #32)."""

from backend.quant_pro.stock_report import annualized_report_eps


def test_printed_annualized_eps_is_kept():
    # Q1: net profit Rs 2.56 arba YTD on Rs 30.2 arba capital → YTD EPS 8.48; the report prints 33.9 (annualized)
    assert annualized_report_eps(33.9, 1, 2.56e9, 30.2e9) == 33.9


def test_year_to_date_eps_is_annualized():
    # the same report printing the year-to-date figure instead
    assert round(annualized_report_eps(8.48, 1, 2.56e9, 30.2e9), 2) == 33.92
    # Q2: Rs 5.13 arba net profit for the half year → YTD EPS 16.99; printed 17.0 (YTD) → 34.0
    assert round(annualized_report_eps(17.0, 2, 5.13e9, 30.2e9), 2) == 34.0


def test_without_profit_or_capital_the_printed_figure_stands():
    assert annualized_report_eps(33.9, 1) == 33.9
    assert annualized_report_eps(33.9, 4, 1e9, 1e10) == 33.9
    assert annualized_report_eps(None, 2) is None
