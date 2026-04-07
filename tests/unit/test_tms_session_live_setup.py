from __future__ import annotations

from backend.quant_pro.tms_session import load_tms_settings, validate_live_setup


def test_live_setup_loads_credentials_from_secret_file(monkeypatch, tmp_path):
    secret = tmp_path / "broker.env"
    secret.write_text("NEPSE_TMS_USERNAME=user123\nNEPSE_TMS_PASSWORD=pass123\n", encoding="utf-8")
    monkeypatch.delenv("NEPSE_TMS_USERNAME", raising=False)
    monkeypatch.delenv("NEPSE_TMS_PASSWORD", raising=False)
    monkeypatch.setenv("NEPSE_TMS_SECRET_FILE", str(secret))
    monkeypatch.setenv("NEPSE_LIVE_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("NEPSE_LIVE_EXECUTION_MODE", "shadow_live")

    settings = load_tms_settings()

    assert settings.username == "user123"
    assert settings.password == "pass123"
    assert settings.credentials_source.startswith("secret_file:")


def test_live_setup_requires_owner_confirmation(monkeypatch):
    monkeypatch.setenv("NEPSE_LIVE_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("NEPSE_LIVE_EXECUTION_MODE", "live")
    monkeypatch.setenv("NEPSE_LIVE_OWNER_CONFIRM_REQUIRED", "false")

    settings = load_tms_settings()
    errors = validate_live_setup(settings)

    assert any("owner confirmation" in item.lower() for item in errors)
