from apps.tui import dashboard_tui


def test_dedupe_symbol_rows_keeps_last_duplicate():
    rows = [
        ("AAA", 100.0),
        ("BBB", 200.0),
        ("AAA", 101.0),
    ]

    assert dashboard_tui._dedupe_symbol_rows(rows) == [
        ("AAA", 101.0),
        ("BBB", 200.0),
    ]


def test_dedupe_symbol_rows_normalizes_symbol_case():
    rows = [
        ("aaa", 100.0),
        ("AAA", 101.0),
        (" bbb ", 200.0),
    ]

    assert dashboard_tui._dedupe_symbol_rows(rows) == [
        ("AAA", 101.0),
        (" bbb ", 200.0),
    ]
