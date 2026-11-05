"""Dashboard tests with the FastAPI TestClient and an in-memory sink."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from ragcompliance import Auditor, AuditStorage, RAGComplianceConfig, RetrievedChunk
from ragcompliance.app import _csv_safe, create_app


def _setup(**cfg):
    config = RAGComplianceConfig(sink_url="memory://", async_writes=False, workspace_id="acme", **cfg)
    storage = AuditStorage(config)
    auditor = Auditor(config, storage=storage, session_id="s1", actor={"id": "alice"}, billing=None, alerter=None)
    with auditor.trace('<img src=x onerror="alert(1)"> what?') as t:
        t.add_chunk(RetrievedChunk("c", "doc.pdf", "c1", 0.9))
        t.set_answer("=HYPERLINK(evil)", model_name="m")
    other = Auditor(RAGComplianceConfig(sink_url="memory://", async_writes=False, workspace_id="other"),
                    storage=storage, session_id="s2", billing=None, alerter=None)
    with other.trace("secret tenant question") as t:
        t.set_answer("secret")
    app = create_app(config, storage=storage, billing=None, enable_sso=False)
    return TestClient(app), storage


class TestReadEndpoints:
    def test_health(self):
        client, _ = _setup()
        body = client.get("/health").json()
        assert body["status"] == "ok" and "version" in body

    def test_logs_pinned_to_configured_workspace(self):
        client, _ = _setup()
        body = client.get("/api/logs?workspace_id=other").json()
        assert body["count"] == 1 and body["logs"][0]["workspace_id"] == "acme"
        body = client.get("/api/logs?session_id=s1").json()
        assert body["count"] == 1
        csv_text = client.get("/api/logs/export.csv?workspace_id=other").text
        assert "secret tenant" not in csv_text and "alice" in csv_text
        js = client.get("/api/logs/export.json?workspace_id=other").json()
        assert js["count"] == 1 and js["logs"][0]["workspace_id"] == "acme"

    def test_any_workspace_opt_in(self):
        client, _ = _setup(dashboard_allow_any_workspace=True)
        body = client.get("/api/logs?workspace_id=other").json()
        assert body["logs"][0]["workspace_id"] == "other"

    def test_summary_detail_verify(self):
        client, storage = _setup()
        summary = client.get("/api/summary").json()
        assert summary["total_queries"] == 1 and summary["avg_chunks_retrieved"] == 1.0 and summary["dropped_count"] == 0
        rec = storage.query(workspace_id="acme")[0]
        assert client.get(f"/api/logs/detail/{rec['id']}").json()["id"] == rec["id"]
        assert client.get("/api/logs/detail/nope").status_code == 404
        other = storage.query(workspace_id="other")[0]
        assert client.get(f"/api/logs/detail/{other['id']}").status_code == 404
        ver = client.get("/api/verify").json()
        assert ver["ok"] is True and ver["records_checked"] == 1

    def test_empty_summary(self):
        config = RAGComplianceConfig(sink_url="memory://", async_writes=False, workspace_id="empty")
        app = create_app(config, storage=AuditStorage(config), billing=None, enable_sso=False)
        assert TestClient(app).get("/api/summary").json() == {"total_queries": 0, "workspace_id": "empty"}

    def test_html_escapes_everything(self):
        client, _ = _setup()
        html = client.get("/").text
        assert "function esc(" in html and "esc(r.query)" in html
        assert "${r.query}" not in html and "Verify chain" in html

    def test_csv_formula_injection_neutralised(self):
        client, _ = _setup()
        text = client.get("/api/logs/export.csv").text
        assert "'=HYPERLINK(evil)" in text
        assert _csv_safe("+1") == "'+1" and _csv_safe("safe") == "safe" and _csv_safe(None) == ""

    def test_cors_configurable(self):
        client, _ = _setup(dashboard_cors_origins=["https://ok.example"])
        r = client.options("/api/logs", headers={"Origin": "https://ok.example", "Access-Control-Request-Method": "GET"})
        assert r.headers.get("access-control-allow-origin") == "https://ok.example"
        client2, _ = _setup()
        r2 = client2.get("/api/logs", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in r2.headers


class TestBillingEndpointsWithoutBilling:
    def test_billing_disabled_paths(self):
        client, _ = _setup()
        assert client.get("/health/billing").status_code == 503
        assert client.get("/api/plans").json() == {"plans": {}}
        assert client.post("/billing/checkout", json={"tier": "team"}).status_code == 503
        assert client.post("/stripe/webhook", content=b"{}", headers={"Stripe-Signature": "x"}).status_code == 503
        sub = client.get("/billing/subscription/other").json()
        assert sub["workspace_id"] == "acme" and sub["tier"] == "free"


class TestBillingEndpointsWithBilling:
    class FakeBilling:
        def __init__(self):
            self.checkout_calls = []

        def readiness(self):
            from ragcompliance.billing import BillingReadiness

            return BillingReadiness(mode="test", ok=True)

        def create_checkout_session(self, workspace_id, tier, success_url=None, cancel_url=None):
            self.checkout_calls.append((workspace_id, tier))
            if tier == "bad":
                raise ValueError("Unknown tier")
            if tier == "down":
                raise RuntimeError("Stripe not configured")
            if tier == "boom":
                raise Exception("stripe exploded")
            return "https://checkout.stripe.com/x"

        def handle_webhook(self, payload, signature):
            if signature == "bad":
                raise ValueError("bad sig")
            if signature == "unconfigured":
                raise RuntimeError("not configured")
            return {"handled": True}

        def get_workspace_subscription(self, workspace_id):
            return {"workspace_id": workspace_id, "tier": "team", "status": "active"} if workspace_id == "acme" else None

    def _client(self):
        config = RAGComplianceConfig(sink_url="memory://", async_writes=False, workspace_id="acme")
        billing = self.FakeBilling()
        app = create_app(config, storage=AuditStorage(config), billing=billing, enable_sso=False)
        return TestClient(app), billing

    def test_billing_paths(self):
        client, billing = self._client()
        assert client.get("/health/billing").status_code == 200
        assert "team" in client.get("/api/plans").json()["plans"]
        r = client.post("/billing/checkout", json={"workspace_id": "other", "tier": "team"})
        assert r.json()["checkout_url"].startswith("https://") and billing.checkout_calls == [("acme", "team")]
        assert client.post("/billing/checkout", json={"tier": "bad"}).status_code == 400
        assert client.post("/billing/checkout", json={"tier": "down"}).status_code == 503
        assert client.post("/billing/checkout", json={"tier": "boom"}).status_code == 502
        assert client.post("/stripe/webhook", content=b"{}").status_code == 400
        assert client.post("/stripe/webhook", content=b"{}", headers={"Stripe-Signature": "ok"}).json() == {"handled": True}
        assert client.post("/stripe/webhook", content=b"{}", headers={"Stripe-Signature": "bad"}).status_code == 400
        assert client.post("/stripe/webhook", content=b"{}", headers={"Stripe-Signature": "unconfigured"}).status_code == 503
        assert client.get("/billing/subscription/acme").json()["tier"] == "team"


def test_module_level_app_is_lazy(monkeypatch):
    import ragcompliance.app as mod

    monkeypatch.setenv("RAGCOMPLIANCE_DEV_MODE", "true")
    monkeypatch.setattr(mod, "_app_instance", None)
    app = mod.app
    assert app.title == "RAGCompliance Dashboard" and mod.app is app
    with pytest.raises(AttributeError):
        mod.nope  # noqa: B018


def test_create_app_with_sso_disabled_env(monkeypatch):
    for k in ("RAGCOMPLIANCE_OIDC_ISSUER", "RAGCOMPLIANCE_OIDC_CLIENT_ID", "RAGCOMPLIANCE_OIDC_CLIENT_SECRET", "RAGCOMPLIANCE_OIDC_REDIRECT_URI"):
        monkeypatch.delenv(k, raising=False)
    config = RAGComplianceConfig(sink_url="memory://", async_writes=False)
    app = create_app(config, storage=AuditStorage(config), billing=None)
    assert app.state.sso.enabled is False
