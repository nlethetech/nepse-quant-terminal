from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from validation import quick_chart


def test_quick_chart_date_format_is_cross_platform():
    assert "%-" not in quick_chart._MONTH_YEAR_LABEL
    assert "%#" not in quick_chart._MONTH_YEAR_LABEL


def test_generate_quick_chart_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")

    start = date(2026, 1, 1)
    daily_nav = [
        ((start + timedelta(days=i)).isoformat(), 1_000_000 + i * 1_000)
        for i in range(80)
    ]
    result = {
        "summary": {
            "daily_nav": daily_nav,
            "total_return_pct": 7.9,
            "sharpe_ratio": 1.2,
            "max_drawdown_pct": -2.1,
            "trade_count": 4,
            "win_rate_pct": 50.0,
        },
        "nepse": {"return_pct": 2.5},
    }

    path = quick_chart.generate_quick_chart(
        result,
        strategy_name="Test Strategy",
        start_date=daily_nav[0][0],
        end_date=daily_nav[-1][0],
        output_dir=tmp_path,
        auto_open=False,
    )

    assert path is not None
    assert path.endswith(".png")
    assert Path(path).exists()
