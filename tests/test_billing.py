"""Tests for ragcompliance.billing — no live Stripe or Supabase calls."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ragcompliance.billing import (
    PLANS,
    BillingManager,
    WorkspaceSubscription,
    refresh_plans,
)


# --------------------------------------------------------------------------- #
# PLANS                                                                        #
# --------------------------------------------------------------------------- #

class TestPlans:
    def test_plans_has_team_and_enterprise(self):
        assert "team" in PLANS
        assert "enterprise" in PLANS

    def test_team_is_49_dollars(self):
        assert PLANS["team"]["price_usd"] == 49
        assert PLANS["team"]["query_limit"] == 10_000

    def test_enterprise_is_199_dollars_unlimited(self):
        assert PLANS["enterprise"]["price_usd"] == 199
        assert PLANS["enterprise"]["query_limit"] is None

    def test_refresh_plans_picks_up_env_change(self, monkeypatch):
        monkeypatch.setenv("STRIPE_PRICE_ID_TEAM", "price_team_xyz")
        refresh_plans()
        from ragcompliance.billing import PLANS as refreshed
        assert refreshed["team"]["stripe_price_id"] == "price_team_xyz"


# --------------------------------------------------------------------------- #
# WorkspaceSubscription                                                        #
# --------------------------------------------------------------------------- #

class TestWorkspaceSubscription:
    def test_from_row_parses_iso_timestamp(self):
        row = {
            "workspace_id": "w1",
            "tier": "team",
            "status": "active",
            "current_period_end": "2026-05-01T00:00:00+00:00",
            "query_count_current_period": 42,
        }
        sub = WorkspaceSubscription.from_row(row)
        assert sub.workspace_id == "w1"
        assert sub.tier == "team"
        assert sub.status == "active"
        assert sub.current_period_end.year == 2026
        assert sub.query_count_current_period == 42

    def test_to_row_round_trip(self):
        sub = WorkspaceSubscription(
            workspace_id="w2",
            tier="enterprise",
            status="active",
            query_count_current_period=1,
        )
        row = sub.to_row()
        assert row["workspace_id"] == "w2"
        assert row["tier"] == "enterprise"
        assert "updated_at" in row


# --------------------------------------------------------------------------- #
# BillingManager.from_env                                                      #
# --------------------------------------------------------------------------- #

class TestFromEnv:
    def test_from_env_reads_all_vars(self, monkeypatch):
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_xyz")
        monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_xyz")
        monkeypatch.setenv("RAGCOMPLIANCE_SUPABASE_URL", "")
        monkeypatch.setenv("RAGCOMPLIANCE_SUPABASE_KEY", "")
        monkeypatch.setenv("APP_BASE_URL", "https://audit.example.com/")
        m = BillingManager.from_env()
        assert m.stripe_secret_key == "sk_test_xyz"
        assert m.stripe_webhook_secret == "whsec_xyz"
        assert m.app_base_url == "https://audit.example.com"  # trailing slash stripped


# --------------------------------------------------------------------------- #
# create_checkout_session                                                      #
# --------------------------------------------------------------------------- #

class TestCheckout:
    def test_raises_without_stripe(self):
        m = BillingManager()
        with pytest.raises(RuntimeError, match="Stripe not configured"):
            m.create_checkout_session("w1", "team")

    def test_rejects_unknown_tier(self, monkeypatch):
        monkeypatch.setenv("STRIPE_PRICE_ID_TEAM", "price_team")
        monkeypatch.setenv("STRIPE_PRICE_ID_ENTERPRISE", "price_ent")
        refresh_plans()
        m = BillingManager(stripe_secret_key="sk_test_dummy")
        # Force a stripe module on the instance so we bypass the "not configured" guard
        m._stripe = MagicMock()
        with pytest.raises(ValueError, match="Unknown tier"):
            m.create_checkout_session("w1", "platinum")

    def test_rejects_tier_without_price_id(self, monkeypatch):
        monkeypatch.delenv("STRIPE_PRICE_ID_TEAM", raising=False)
        refresh_plans()
        m = BillingManager(stripe_secret_key="sk_test_dummy")
        m._stripe = MagicMock()
        with pytest.raises(RuntimeError, match="STRIPE_PRICE_ID_TEAM"):
            m.create_checkout_session("w1", "team")

    def test_returns_checkout_url(self, monkeypatch):
        monkeypatch.setenv("STRIPE_PRICE_ID_TEAM", "price_team")
        refresh_plans()
        m = BillingManager(
            stripe_secret_key="sk_test_dummy",
            app_base_url="https://example.com",
        )
        fake_session = MagicMock(url="https://stripe.test/checkout/xyz")
        fake_stripe = MagicMock()
        fake_stripe.checkout.Session.create.return_value = fake_session
        m._stripe = fake_stripe

        url = m.create_checkout_session("wabc", "team")

        assert url == "https://stripe.test/checkout/xyz"
        call = fake_stripe.checkout.Session.create.call_args.kwargs
        assert call["mode"] == "subscription"
        assert call["client_reference_id"] == "wabc"
        assert call["metadata"]["workspace_id"] == "wabc"
        assert call["line_items"][0]["price"] == "price_team"


# --------------------------------------------------------------------------- #
# handle_webhook                                                               #
# --------------------------------------------------------------------------- #

class TestWebhook:
    def _make_manager(self):
        m = BillingManager(
            stripe_secret_key="sk_test_dummy",
            stripe_webhook_secret="whsec_dummy",
        )
        m._stripe = MagicMock()
        # Capture upserts so we can assert on them without touching Supabase.
        m._upsert = MagicMock()
        m.get_workspace_subscription = MagicMock(return_value=None)
        return m

    def test_raises_without_webhook_secret(self):
        m = BillingManager(stripe_secret_key="sk_test_dummy")
        m._stripe = MagicMock()
        with pytest.raises(RuntimeError, match="STRIPE_WEBHOOK_SECRET"):
            m.handle_webhook(b"{}", "sig")

    def test_checkout_completed_creates_active_subscription(self):
        m = self._make_manager()
        m._stripe.Webhook.construct_event.return_value = {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "client_reference_id": "wabc",
                    "customer": "cus_123",
                    "subscription": "sub_123",
                    "metadata": {"tier": "team", "workspace_id": "wabc"},
                }
            },
        }
        result = m.handle_webhook(b"{}", "sig")
        assert result == {
            "handled": True,
            "workspace_id": "wabc",
            "tier": "team",
            "status": "active",
        }
        upserted = m._upsert.call_args.args[0]
        assert upserted.tier == "team"
        assert upserted.status == "active"
        assert upserted.stripe_customer_id == "cus_123"

    def test_subscription_deleted_downgrades_to_free(self):
        m = self._make_manager()
        m._stripe.Webhook.construct_event.return_value = {
            "type": "customer.subscription.deleted",
            "data": {
                "object": {
                    "id": "sub_123",
                    "metadata": {"workspace_id": "wabc"},
                }
            },
        }
        result = m.handle_webhook(b"{}", "sig")
        assert result["status"] == "canceled"
        upserted = m._upsert.call_args.args[0]
        assert upserted.tier == "free"
        assert upserted.status == "canceled"

    def test_unhandled_event_type_returns_false(self):
        m = self._make_manager()
        m._stripe.Webhook.construct_event.return_value = {
            "type": "customer.created",
            "data": {"object": {}},
        }
        result = m.handle_webhook(b"{}", "sig")
        assert result == {"handled": False, "event_type": "customer.created"}


# --------------------------------------------------------------------------- #
# check_query_quota                                                            #
# --------------------------------------------------------------------------- #

class TestQuota:
    def test_allows_when_no_subscription_row(self):
        m = BillingManager()
        m.get_workspace_subscription = MagicMock(return_value=None)
        assert m.check_query_quota("wabc") is True

    def test_allows_for_unlimited_enterprise(self, monkeypatch):
        monkeypatch.setenv("STRIPE_PRICE_ID_ENTERPRISE", "price_ent")
        refresh_plans()
        m = BillingManager()
        m.get_workspace_subscription = MagicMock(return_value={
            "workspace_id": "wabc",
            "tier": "enterprise",
            "status": "active",
            "query_count_current_period": 10_000_000,
        })
        assert m.check_query_quota("wabc") is True

    def test_blocks_active_team_over_limit(self):
        m = BillingManager()
        m.get_workspace_subscription = MagicMock(return_value={
            "workspace_id": "wabc",
            "tier": "team",
            "status": "active",
            "query_count_current_period": 10_000,
        })
        assert m.check_query_quota("wabc") is False

    def test_allows_active_team_at_limit_minus_one(self):
        m = BillingManager()
        m.get_workspace_subscription = MagicMock(return_value={
            "workspace_id": "wabc",
            "tier": "team",
            "status": "active",
            "query_count_current_period": 9_999,
        })
        assert m.check_query_quota("wabc") is True

    def test_allows_inactive_subscription(self):
        m = BillingManager()
        m.get_workspace_subscription = MagicMock(return_value={
            "workspace_id": "wabc",
            "tier": "team",
            "status": "canceled",
            "query_count_current_period": 99999,
        })
        assert m.check_query_quota("wabc") is True


# --------------------------------------------------------------------------- #
# increment_usage                                                              #
# --------------------------------------------------------------------------- #

class TestIncrement:
    def test_returns_none_without_supabase(self):
        m = BillingManager()
        assert m.increment_usage("wabc") is None

    def test_calls_rpc_and_returns_count(self):
        m = BillingManager()
        fake_client = MagicMock()
        fake_client.rpc.return_value.execute.return_value = MagicMock(
            data=[{"query_count_current_period": 7}]
        )
        m._supabase = fake_client
        assert m.increment_usage("wabc") == 7
        fake_client.rpc.assert_called_once_with(
            "increment_workspace_usage", {"p_workspace_id": "wabc"}
        )
