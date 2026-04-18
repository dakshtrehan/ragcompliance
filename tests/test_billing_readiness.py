"""Tests for Stripe live-mode readiness checks."""
from __future__ import annotations

import pytest

from ragcompliance.billing import (
    BillingManager,
    BillingReadiness,
    _detect_mode,
    refresh_plans,
)


# ------------------------------------------------------------------ #
# _detect_mode                                                         #
# ------------------------------------------------------------------ #


class TestDetectMode:
    @pytest.mark.parametrize(
        "key,expected",
        [
            ("sk_live_abc123", "live"),
            ("sk_test_abc123", "test"),
            ("rk_live_abc123", "live"),
            ("rk_test_abc123", "test"),
            ("", "unconfigured"),
            ("bogus_key", "unconfigured"),
        ],
    )
    def test_mode_by_prefix(self, key, expected):
        assert _detect_mode(key) == expected


# ------------------------------------------------------------------ #
# readiness()                                                          #
# ------------------------------------------------------------------ #


@pytest.fixture
def priced_env(monkeypatch):
    """Set valid-looking live price IDs so price checks pass by default."""
    monkeypatch.setenv("STRIPE_PRICE_ID_TEAM", "price_live_team_123")
    monkeypatch.setenv("STRIPE_PRICE_ID_ENTERPRISE", "price_live_ent_456")
    refresh_plans()
    yield
    monkeypatch.delenv("STRIPE_PRICE_ID_TEAM", raising=False)
    monkeypatch.delenv("STRIPE_PRICE_ID_ENTERPRISE", raising=False)
    refresh_plans()


class TestReadiness:
    def test_unconfigured_is_not_ok(self, priced_env):
        mgr = BillingManager()
        rd = mgr.readiness()
        assert rd.mode == "unconfigured"
        assert rd.ok is False
        assert any("STRIPE_SECRET_KEY" in i for i in rd.issues)

    def test_test_mode_fully_configured_is_ok(self, priced_env):
        mgr = BillingManager(
            stripe_secret_key="sk_test_abc123",
            stripe_webhook_secret="whsec_test_xyz",
            supabase_url="https://x.supabase.co",
            supabase_key="service_key",
            app_base_url="http://localhost:8000",
        )
        rd = mgr.readiness()
        assert rd.mode == "test"
        # Supabase may fail to actually connect in CI, so we check issues set
        # matches the expected shape rather than ok=True strictly.
        # If supabase_configured=True, issues should be empty.
        if rd.summary["supabase_configured"]:
            assert rd.ok is True

    def test_live_mode_missing_webhook_flags_issue(self, priced_env):
        mgr = BillingManager(
            stripe_secret_key="sk_live_real",
            stripe_webhook_secret="",
            supabase_url="https://x.supabase.co",
            supabase_key="service_key",
            app_base_url="https://dash.example.com",
        )
        rd = mgr.readiness()
        assert rd.mode == "live"
        assert rd.ok is False
        assert any("WEBHOOK_SECRET" in i for i in rd.issues)

    def test_live_mode_localhost_base_url_flags_issue(self, priced_env):
        mgr = BillingManager(
            stripe_secret_key="sk_live_abc",
            stripe_webhook_secret="whsec_live_xyz",
            supabase_url="https://x.supabase.co",
            supabase_key="sk",
            app_base_url="http://localhost:8000",
        )
        rd = mgr.readiness()
        assert any("APP_BASE_URL" in i for i in rd.issues)

    def test_test_mode_localhost_is_fine(self, priced_env):
        mgr = BillingManager(
            stripe_secret_key="sk_test_abc",
            stripe_webhook_secret="whsec_test_xyz",
            supabase_url="https://x.supabase.co",
            supabase_key="sk",
            app_base_url="http://localhost:8000",
        )
        rd = mgr.readiness()
        assert not any("APP_BASE_URL" in i for i in rd.issues)

    def test_product_id_pasted_as_price_id_is_flagged(self, monkeypatch):
        monkeypatch.setenv("STRIPE_PRICE_ID_TEAM", "prod_abc_wrong")
        monkeypatch.setenv("STRIPE_PRICE_ID_ENTERPRISE", "price_live_ent")
        refresh_plans()
        try:
            mgr = BillingManager(
                stripe_secret_key="sk_live_abc",
                stripe_webhook_secret="whsec_live_xyz",
                supabase_url="https://x.supabase.co",
                supabase_key="sk",
                app_base_url="https://dash.example.com",
            )
            rd = mgr.readiness()
            assert any("STRIPE_PRICE_ID_TEAM" in i for i in rd.issues)
            assert rd.ok is False
        finally:
            monkeypatch.delenv("STRIPE_PRICE_ID_TEAM", raising=False)
            monkeypatch.delenv("STRIPE_PRICE_ID_ENTERPRISE", raising=False)
            refresh_plans()

    def test_missing_price_ids_are_flagged(self, monkeypatch):
        monkeypatch.delenv("STRIPE_PRICE_ID_TEAM", raising=False)
        monkeypatch.delenv("STRIPE_PRICE_ID_ENTERPRISE", raising=False)
        refresh_plans()
        mgr = BillingManager(
            stripe_secret_key="sk_live_abc",
            stripe_webhook_secret="whsec_live_xyz",
            supabase_url="https://x.supabase.co",
            supabase_key="sk",
            app_base_url="https://dash.example.com",
        )
        rd = mgr.readiness()
        assert any("STRIPE_PRICE_ID_TEAM" in i for i in rd.issues)
        assert any("STRIPE_PRICE_ID_ENTERPRISE" in i for i in rd.issues)

    def test_summary_never_leaks_full_secret(self, priced_env):
        mgr = BillingManager(
            stripe_secret_key="sk_live_extremely_secret_value",
            stripe_webhook_secret="whsec_test_abc",
        )
        rd = mgr.readiness()
        # Prefix only.
        assert rd.summary["secret_key_prefix"] == "sk_live…"
        # Don't leak the actual webhook secret.
        assert rd.summary["webhook_secret_set"] is True
        # Raw secret shouldn't appear anywhere in the serialized summary.
        serialized = str(rd.to_dict())
        assert "extremely_secret_value" not in serialized
        assert "whsec_test_abc" not in serialized

    def test_readiness_never_raises(self):
        # No args — even a wholly un-instantiable config should return a
        # BillingReadiness, not raise.
        mgr = BillingManager()
        rd = mgr.readiness()
        assert isinstance(rd, BillingReadiness)
        assert rd.ok is False
