from __future__ import annotations

from backend.trading import strategy_registry


def test_list_public_builtin_strategies():
    strategies = strategy_registry.list_strategies()
    ids = {str(item.get("id") or "") for item in strategies}

    assert ids == {"c5", "sat06"}


def test_default_strategy_for_accounts():
    assert strategy_registry.default_strategy_for_account("account_1") == "c5"
    assert strategy_registry.default_strategy_for_account("account_2") == "sat06"
    assert strategy_registry.default_strategy_for_account("account_99") == "c5"


def test_ensure_account_strategy_ids_fills_missing_or_unknown():
    rows = strategy_registry.ensure_account_strategy_ids(
        [
            {"id": "account_1", "name": "Account 1"},
            {"id": "account_2", "name": "Account 2", "strategy_id": "unknown"},
        ]
    )

    assert rows[0]["strategy_id"] == "c5"
    assert rows[1]["strategy_id"] == "sat06"
