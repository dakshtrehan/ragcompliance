"""Alert payload privacy and billing webhook dispatch with a fake Stripe."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from ragcompliance import AuditRecord, RetrievedChunk, SlackAlerter
from ragcompliance.billing import BillingManager, WorkspaceSubscription, refresh_plans


class TestAlertPrivacy:
    def _alerter(self, **kw):
        a = SlackAlerter(webhook_url="http://127.0.0.1:9/never", **kw)
        a._stop.set()  # no network in these tests
        return a

    def test_query_omitted_by_default(self):
        rec = AuditRecord(id="rid", workspace_id="w", session_id="s", query="very private question",
                          actor={"id": "alice"}, retrieved_chunks=[])
        a = self._alerter()
        payload = a._build_payload(rec, a.evaluate(rec))
        text = payload["text"]
        assert "very private" not in text and "record: `rid`" in text and "actor: `alice`" in text
        assert "retrieval_returned_zero_chunks" in text

    def test_query_included_when_opted_in(self):
        rec = AuditRecord(query="q" * 300)
        a = self._alerter(include_query=True)
        text = a._build_payload(rec, [])["text"]
        assert "query: `" in text and text.count("q") >= 240 and "..." in text

    def test_from_env_include_flag(self, monkeypatch):
        monkeypatch.setenv("RAGCOMPLIANCE_SLACK_WEBHOOK_URL", "http://127.0.0.1:9/x")
        monkeypatch.setenv("RAGCOMPLIANCE_SLACK_INCLUDE_QUERY", "true")
        a = SlackAlerter.from_env()
        assert a is not None and a.include_query is True
        a._stop.set()
        monkeypatch.delenv("RAGCOMPLIANCE_SLACK_WEBHOOK_URL")
        assert SlackAlerter.from_env() is None

    def test_evaluate_error_detail_has_type(self):
        a = self._alerter()
        fired = a.evaluate(AuditRecord(retrieved_chunks=[RetrievedChunk("c", "u", "i", 0.9)]), error=KeyError("k"))
        assert fired[0].name == "chain_errored" and fired[0].detail.startswith("KeyError")

    def test_queue_full_drops(self, caplog):
        a = self._alerter(max_queue=1)
        rec = AuditRecord(retrieved_chunks=[])
        assert a.maybe_alert(rec) is True
        with caplog.at_level(logging.WARNING, logger="ragcompliance.alerts"):
            assert a.maybe_alert(rec) is False
        assert "queue full" in caplog.text

    def test_evaluate_exception_contained(self):
        a = self._alerter()

        class Weird:
            @property
            def retrieved_chunks(self):
                raise RuntimeError("x")

        assert a.maybe_alert(Weird()) is False

    def test_post_failure_logged(self, caplog):
        a = SlackAlerter(webhook_url="http://127.0.0.1:9/never")
        with caplog.at_level(logging.WARNING, logger="ragcompliance.alerts"):
            a.maybe_alert(AuditRecord(retrieved_chunks=[]))
            a.flush(timeout=3)
            import time

            for _ in range(50):
                if "alert post failed" in caplog.text:
                    break
                time.sleep(0.05)
        a._stop.set()
        assert "alert post failed" in caplog.text


class FakeStripeObject(dict):
    def to_dict_recursive(self):
        return dict(self)


class FakeStripe:
    def __init__(self, event):
        self._event = event
        self.api_key = None
        outer = self

        class Webhook:
            @staticmethod
            def construct_event(payload, sig, secret):
                if sig == "bad":
                    raise ValueError("signature failed")
                return outer._event

        class Session:
            @staticmethod
            def create(**kw):
                outer.created = kw
                return SimpleNamespace(url="https://checkout/x")

        self.Webhook = Webhook
        self.checkout = SimpleNamespace(Session=Session)


class FakeSupabase:
    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.rpcs: list = []

    def table(self, name):
        outer = self

        class Q:
            def __init__(self):
                self._filters = []

            def select(self, *a):
                return self

            def eq(self, k, v):
                self._filters.append((k, v))
                return self

            def limit(self, n):
                return self

            def upsert(self, row, on_conflict=None):
                outer.rows[row["workspace_id"]] = {**outer.rows.get(row["workspace_id"], {}), **row}
                return self

            def execute(self):
                rows = list(outer.rows.values())
                for k, v in self._filters:
                    rows = [r for r in rows if r.get(k) == v]
                return SimpleNamespace(data=rows)

        return Q()

    def rpc(self, name, params):
        self.rpcs.append((name, params))
        outer = self

        class R:
            def execute(self):
                if name == "increment_workspace_usage":
                    row = outer.rows.setdefault(params["p_workspace_id"], {"workspace_id": params["p_workspace_id"], "query_count_current_period": 0})
                    row["query_count_current_period"] = row.get("query_count_current_period", 0) + 1
                    return SimpleNamespace(data=[{"query_count_current_period": row["query_count_current_period"]}])
                return SimpleNamespace(data=None)

        return R()


def _manager(event=None, supabase=True):
    mgr = BillingManager(stripe_secret_key="sk_test_x", stripe_webhook_secret="whsec_x", app_base_url="https://app/")
    mgr._stripe = FakeStripe(event or {})
    mgr._supabase = FakeSupabase() if supabase else None
    return mgr


class TestBillingWebhooks:
    def test_checkout_completed(self, monkeypatch):
        monkeypatch.setenv("STRIPE_PRICE_ID_TEAM", "price_team")
        refresh_plans()
        event = {"type": "checkout.session.completed", "data": {"object": FakeStripeObject(
            client_reference_id="ws1", metadata={"tier": "team"}, customer="cus_1", subscription="sub_1")}}
        mgr = _manager(event)
        out = mgr.handle_webhook(b"{}", "ok")
        assert out["handled"] and out["tier"] == "team"
        row = mgr.get_workspace_subscription("ws1")
        assert row["status"] == "active" and row["stripe_customer_id"] == "cus_1"
        assert mgr.check_query_quota("ws1") is True
        assert mgr.increment_usage("ws1") == 1

    def test_checkout_without_workspace(self):
        mgr = _manager({"type": "checkout.session.completed", "data": {"object": {"metadata": {}}}})
        assert mgr.handle_webhook(b"{}", "ok")["handled"] is False

    def test_subscription_updated_rollover_and_tier_change(self, monkeypatch):
        monkeypatch.setenv("STRIPE_PRICE_ID_ENTERPRISE", "price_ent")
        refresh_plans()
        mgr = _manager({})
        old_end = datetime.now(timezone.utc) - timedelta(days=1)
        mgr._upsert(WorkspaceSubscription(workspace_id="ws1", tier="team", status="active",
                                          current_period_end=old_end, query_count_current_period=500))
        new_end = int((datetime.now(timezone.utc) + timedelta(days=30)).timestamp())
        mgr._stripe._event = {"type": "customer.subscription.updated", "data": {"object": {
            "id": "sub_2", "status": "active", "metadata": {"workspace_id": "ws1"}, "current_period_end": new_end,
            "items": {"data": [{"price": {"id": "price_ent"}}]}}}}
        out = mgr.handle_webhook(b"{}", "ok")
        assert out["period_rolled_over"] is True and out["tier"] == "enterprise"
        assert mgr.get_workspace_subscription("ws1")["query_count_current_period"] == 0
        assert mgr.handle_webhook.__self__._on_subscription_updated({"metadata": {}})["handled"] is False

    def test_subscription_deleted_and_payment_failed(self):
        mgr = _manager({})
        mgr._upsert(WorkspaceSubscription(workspace_id="ws1", tier="team", status="active", stripe_customer_id="cus_9"))
        mgr._stripe._event = {"type": "customer.subscription.deleted", "data": {"object": {"metadata": {"workspace_id": "ws1"}}}}
        assert mgr.handle_webhook(b"{}", "ok")["status"] == "canceled"
        assert mgr._on_subscription_deleted({"metadata": {}})["handled"] is False
        mgr._stripe._event = {"type": "invoice.payment_failed", "data": {"object": {"customer": "cus_9"}}}
        assert mgr.handle_webhook(b"{}", "ok")["event"] == "payment_failed"
        assert mgr.get_workspace_subscription("ws1")["status"] == "past_due"
        mgr._stripe._event = {"type": "something.else", "data": {"object": {}}}
        assert mgr.handle_webhook(b"{}", "ok")["handled"] is False

    def test_bad_signature_and_unconfigured(self):
        mgr = _manager({})
        with pytest.raises(ValueError):
            mgr.handle_webhook(b"{}", "bad")
        bare = BillingManager()
        with pytest.raises(RuntimeError):
            bare.handle_webhook(b"{}", "x")
        nosecret = BillingManager(stripe_secret_key="sk_test_x")
        nosecret._stripe = FakeStripe({})
        with pytest.raises(RuntimeError, match="WEBHOOK"):
            nosecret.handle_webhook(b"{}", "x")

    def test_checkout_session(self, monkeypatch):
        monkeypatch.setenv("STRIPE_PRICE_ID_TEAM", "price_team")
        refresh_plans()
        mgr = _manager({})
        url = mgr.create_checkout_session("ws1", "team")
        assert url == "https://checkout/x" and mgr._stripe.created["client_reference_id"] == "ws1"
        with pytest.raises(ValueError):
            mgr.create_checkout_session("ws1", "nope")
        monkeypatch.setenv("STRIPE_PRICE_ID_TEAM", "")
        refresh_plans()
        with pytest.raises(RuntimeError):
            mgr.create_checkout_session("ws1", "team")
        with pytest.raises(RuntimeError):
            BillingManager().create_checkout_session("ws1", "team")

    def test_quota_self_heal_and_limits(self, monkeypatch):
        monkeypatch.setenv("STRIPE_PRICE_ID_TEAM", "price_team")
        refresh_plans()
        mgr = _manager({})
        past = datetime.now(timezone.utc) - timedelta(days=2)
        mgr._upsert(WorkspaceSubscription(workspace_id="ws1", tier="team", status="active",
                                          current_period_end=past, query_count_current_period=99_999))
        assert mgr.check_query_quota("ws1") is True
        assert ("reset_workspace_usage", {"p_workspace_id": "ws1"}) in mgr._supabase.rpcs
        future = datetime.now(timezone.utc) + timedelta(days=2)
        mgr._upsert(WorkspaceSubscription(workspace_id="ws2", tier="team", status="active",
                                          current_period_end=future, query_count_current_period=99_999))
        assert mgr.check_query_quota("ws2") is False
        mgr._upsert(WorkspaceSubscription(workspace_id="ws3", tier="enterprise", status="active", current_period_end=future))
        assert mgr.check_query_quota("ws3") is True
        mgr._upsert(WorkspaceSubscription(workspace_id="ws4", tier="team", status="canceled"))
        assert mgr.check_query_quota("ws4") is True
        assert mgr.check_query_quota("unknown") is True

    def test_no_supabase_paths(self, caplog):
        mgr = _manager({}, supabase=False)
        assert mgr.get_workspace_subscription("x") is None and mgr.increment_usage("x") is None
        with caplog.at_level(logging.DEBUG, logger="ragcompliance.billing"):
            mgr._upsert(WorkspaceSubscription(workspace_id="w", tier="team"))
        assert "would upsert" in caplog.text
        assert mgr._on_payment_failed({"customer": "c"})["handled"] is True

    def test_supabase_errors_are_contained(self):
        mgr = _manager({})

        class Boom:
            def table(self, *a):
                raise RuntimeError("down")

            def rpc(self, *a):
                raise RuntimeError("down")

        mgr._supabase = Boom()
        assert mgr.get_workspace_subscription("x") is None
        assert mgr.increment_usage("x") is None
        mgr._upsert(WorkspaceSubscription(workspace_id="w"))  # logged, not raised
        assert mgr._on_payment_failed({"customer": "c"})["handled"] is True

    def test_from_row_parsing_edge_cases(self):
        sub = WorkspaceSubscription.from_row({"workspace_id": "w", "current_period_end": "2026-01-01T00:00:00.12345Z"})
        assert sub.current_period_end.year == 2026
        assert WorkspaceSubscription.from_row({"workspace_id": "w", "current_period_end": "garbage"}).current_period_end is None
        now = datetime.now(timezone.utc)
        assert WorkspaceSubscription.from_row({"workspace_id": "w", "created_at": now}).created_at == now

    def test_readiness_live_localhost_issue(self, monkeypatch):
        monkeypatch.setenv("STRIPE_PRICE_ID_TEAM", "prod_wrong")
        monkeypatch.setenv("STRIPE_PRICE_ID_ENTERPRISE", "price_ok")
        mgr = BillingManager(stripe_secret_key="sk_live_x", stripe_webhook_secret="w")
        rd = mgr.readiness()
        assert rd.mode == "live" and not rd.ok
        assert any("looks wrong" in i for i in rd.issues) and any("localhost" in i for i in rd.issues)
        assert rd.summary["price_ids_set"]["team"] is True
        assert "prod_wrong" not in str(rd.to_dict())

    def test_import_warnings(self, monkeypatch, caplog):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name in ("stripe", "supabase"):
                raise ImportError(name)
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with caplog.at_level(logging.WARNING, logger="ragcompliance.billing"):
            mgr = BillingManager(stripe_secret_key="sk_test", supabase_url="u", supabase_key="k")
        assert mgr._stripe is None and mgr._supabase is None
        assert "ragcompliance[billing]" in caplog.text and "ragcompliance[supabase]" in caplog.text
