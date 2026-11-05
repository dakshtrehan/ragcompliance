"""FastAPI dashboard (extra ``dashboard``).

Run with::

    uvicorn ragcompliance.app:app --reload

or build your own instance::

    from ragcompliance.app import create_app
    app = create_app(config, storage=my_storage)

Tenant scoping: every read endpoint is pinned to the configured
``workspace_id``. The ``workspace_id`` query parameter is honoured only
when ``RAGCOMPLIANCE_DASHBOARD_ALLOW_ANY_WORKSPACE=true`` (AUDIT.md B05);
turn SSO on before doing that. All record fields are HTML escaped in the
dashboard page (AUDIT.md B06). CORS is off unless
``RAGCOMPLIANCE_DASHBOARD_CORS_ORIGINS`` lists origins.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from ._version import __version__
from .config import RAGComplianceConfig
from .storage import AuditStorage

logger = logging.getLogger(__name__)


def create_app(
    config: RAGComplianceConfig | None = None,
    *,
    storage: AuditStorage | None = None,
    billing: Any | None = None,
    enable_sso: bool = True,
) -> FastAPI:
    """Build the dashboard application. Nothing touches the network until
    the first request."""
    config = config or RAGComplianceConfig.from_env()
    storage = storage or AuditStorage(config)
    if billing is None:
        try:
            from .billing import BillingManager

            billing = BillingManager.from_env()
        except Exception as exc:
            logger.info("ragcompliance dashboard: billing disabled (%s)", exc)
            billing = None

    app = FastAPI(
        title="RAGCompliance Dashboard",
        description="Audit trail viewer for RAG pipelines in regulated industries",
        version=__version__,
    )
    if config.dashboard_cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.dashboard_cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
    app.state.config = config
    app.state.storage = storage
    app.state.billing = billing

    if enable_sso:
        from .auth import init_sso

        app.state.sso = init_sso(app)

    def _scope(workspace_id: str | None) -> str:
        if workspace_id and config.dashboard_allow_any_workspace:
            return workspace_id
        return config.workspace_id

    # ---------------------------------------------------------------- #
    # Health + audit read API                                          #
    # ---------------------------------------------------------------- #

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "version": __version__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/health/billing")
    def billing_health() -> JSONResponse:
        if billing is None:
            return JSONResponse(
                {"mode": "unconfigured", "ok": False, "issues": ["billing extra not installed"]},
                status_code=503,
            )
        rd = billing.readiness()
        return JSONResponse(rd.to_dict(), status_code=200 if rd.ok else 503)

    @app.get("/api/logs")
    def get_logs(
        workspace_id: str | None = Query(None, description="Honoured only when allowed"),
        session_id: str | None = Query(None, description="Filter by session"),
        limit: int = Query(50, ge=1, le=500),
    ) -> dict[str, Any]:
        records = storage.query(
            workspace_id=_scope(workspace_id), session_id=session_id, limit=limit
        )
        return {"count": len(records), "logs": records}

    @app.get("/api/summary")
    def get_summary() -> dict[str, Any]:
        records = storage.query(workspace_id=config.workspace_id, limit=500)
        if not records:
            return {"total_queries": 0, "workspace_id": config.workspace_id}
        latencies = [int(r.get("latency_ms") or 0) for r in records]
        chunk_counts = [len(r.get("retrieved_chunks") or []) for r in records]
        return {
            "workspace_id": config.workspace_id,
            "total_queries": len(records),
            "avg_latency_ms": round(sum(latencies) / len(latencies)),
            "avg_chunks_retrieved": round(sum(chunk_counts) / len(chunk_counts), 1),
            "last_query_at": records[0].get("timestamp"),
            "dropped_count": storage.dropped_count,
        }

    @app.get("/api/verify")
    def verify(
        workspace_id: str | None = Query(None),
        chain_id: str | None = Query(None),
    ) -> dict[str, Any]:
        return storage.verify(workspace_id=_scope(workspace_id), chain_id=chain_id).to_dict()

    @app.get("/api/logs/detail/{record_id}")
    def get_log(record_id: str) -> dict[str, Any]:
        record = storage.get_by_id(record_id, workspace_id=config.workspace_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Audit record not found")
        return record

    # ---------------------------------------------------------------- #
    # Export                                                           #
    # ---------------------------------------------------------------- #

    def _filtered(workspace_id: str | None, session_id: str | None, limit: int) -> list[dict[str, Any]]:
        return storage.query(workspace_id=_scope(workspace_id), session_id=session_id, limit=limit)

    @app.get("/api/logs/export.csv")
    def export_csv(
        workspace_id: str | None = Query(None),
        session_id: str | None = Query(None),
        limit: int = Query(500, ge=1, le=10_000),
    ) -> StreamingResponse:
        records = _filtered(workspace_id, session_id, limit)
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "id", "timestamp", "session_id", "workspace_id", "actor_id", "query",
                "num_chunks", "chunk_sources", "llm_answer", "model_name",
                "chain_signature", "sequence", "record_hash", "latency_ms",
            ]
        )
        for r in records:
            chunks = r.get("retrieved_chunks") or []
            sources = ";".join(
                str(c.get("source_url", "")) if isinstance(c, dict) else "" for c in chunks
            )
            writer.writerow(
                [
                    r.get("id", ""), r.get("timestamp", ""), r.get("session_id", ""),
                    r.get("workspace_id", ""), (r.get("actor") or {}).get("id", ""),
                    _csv_safe(r.get("query", "")), len(chunks), sources,
                    _csv_safe(r.get("llm_answer", "")), r.get("model_name", ""),
                    r.get("chain_signature", ""), r.get("sequence", ""),
                    r.get("record_hash", ""), r.get("latency_ms", ""),
                ]
            )
        buf.seek(0)
        filename = f"ragcompliance-logs-{_stamp()}.csv"
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/logs/export.json")
    def export_json(
        workspace_id: str | None = Query(None),
        session_id: str | None = Query(None),
        limit: int = Query(500, ge=1, le=10_000),
    ) -> StreamingResponse:
        records = _filtered(workspace_id, session_id, limit)
        payload = json.dumps({"count": len(records), "logs": records}, default=str, indent=2)
        filename = f"ragcompliance-logs-{_stamp()}.json"
        return StreamingResponse(
            iter([payload]),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # ---------------------------------------------------------------- #
    # Billing (reference implementation)                               #
    # ---------------------------------------------------------------- #

    @app.get("/api/plans")
    def get_plans() -> dict[str, Any]:
        if billing is None:
            return {"plans": {}}
        from .billing import PLANS

        return {"plans": PLANS}

    @app.post("/billing/checkout")
    async def billing_checkout(payload: dict[str, Any]) -> dict[str, str]:
        if billing is None:
            raise HTTPException(status_code=503, detail="billing not configured")
        workspace_id = _scope(payload.get("workspace_id"))
        tier = payload.get("tier", "team")
        try:
            url = billing.create_checkout_session(
                workspace_id=workspace_id,
                tier=tier,
                success_url=payload.get("success_url"),
                cancel_url=payload.get("cancel_url"),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except Exception as e:
            logger.error("Checkout session error: %s", e)
            raise HTTPException(status_code=502, detail="Stripe error") from e
        return {"checkout_url": url}

    @app.post("/stripe/webhook")
    async def stripe_webhook(
        request: Request,
        stripe_signature: str | None = Header(default=None, alias="Stripe-Signature"),
    ) -> JSONResponse:
        if billing is None:
            raise HTTPException(status_code=503, detail="billing not configured")
        if not stripe_signature:
            raise HTTPException(status_code=400, detail="Missing Stripe-Signature header")
        body = await request.body()
        try:
            result = billing.handle_webhook(body, stripe_signature)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=400, detail="Webhook error") from e
        return JSONResponse(result)

    @app.get("/billing/subscription/{workspace_id}")
    def get_subscription(workspace_id: str) -> dict[str, Any]:
        scoped = _scope(workspace_id)
        if billing is None:
            return {"workspace_id": scoped, "tier": "free", "status": "inactive"}
        row = billing.get_workspace_subscription(scoped)
        if not row:
            return {"workspace_id": scoped, "tier": "free", "status": "inactive"}
        return row

    # ---------------------------------------------------------------- #
    # HTML dashboard                                                   #
    # ---------------------------------------------------------------- #

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return _HTML

    return app


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _csv_safe(value: Any) -> str:
    """Neutralise spreadsheet formula injection in exported text."""
    text = str(value or "")
    if text[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + text
    return text


_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>RAGCompliance Dashboard</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #0f1117; color: #e2e8f0; }
    header { padding: 24px 40px; border-bottom: 1px solid #1e2535; display: flex; align-items: center; gap: 12px; }
    header h1 { font-size: 20px; font-weight: 600; }
    header span { font-size: 12px; background: #1e3a5f; color: #60a5fa; padding: 2px 8px; border-radius: 12px; }
    header .actions { margin-left: auto; display: flex; gap: 8px; }
    .btn { background: #1e3a5f; color: #60a5fa; padding: 8px 14px; border-radius: 6px; text-decoration: none; font-size: 13px; border: 1px solid #1e4b7a; cursor: pointer; }
    .btn:hover { background: #264b73; }
    .stats { display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; padding: 32px 40px 0; }
    .stat { background: #1a1f2e; border: 1px solid #1e2535; border-radius: 8px; padding: 20px; }
    .stat label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; }
    .stat output { display: block; font-size: 28px; font-weight: 700; margin-top: 6px; color: #60a5fa; }
    .logs { padding: 32px 40px; }
    .logs h2 { font-size: 15px; margin-bottom: 16px; color: #94a3b8; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th { text-align: left; padding: 10px 14px; background: #1a1f2e; color: #64748b; font-weight: 500;
         font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; }
    td { padding: 12px 14px; border-bottom: 1px solid #1e2535; vertical-align: top; max-width: 320px;
         overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    tr:hover td { background: #1a1f2e; }
    .sig { font-family: monospace; font-size: 11px; color: #475569; }
    .ms { color: #34d399; }
    .ok { color: #34d399; } .bad { color: #f87171; }
    .empty { text-align: center; padding: 48px; color: #475569; }
  </style>
</head>
<body>
  <header>
    <h1>RAGCompliance</h1>
    <span>Audit Dashboard</span>
    <div class="actions">
      <a class="btn" href="/api/logs/export.csv">Export CSV</a>
      <a class="btn" href="/api/logs/export.json">Export JSON</a>
      <a class="btn" href="/api/verify">Verify chain</a>
    </div>
  </header>

  <div class="stats">
    <div class="stat"><label>Total Queries</label><output id="total">--</output></div>
    <div class="stat"><label>Avg Latency</label><output id="latency">--</output></div>
    <div class="stat"><label>Avg Chunks</label><output id="chunks">--</output></div>
    <div class="stat"><label>Chain</label><output id="chain">--</output></div>
    <div class="stat"><label>Last Query</label><output id="last" style="font-size:13px;margin-top:10px">--</output></div>
  </div>

  <div class="logs">
    <h2>Recent Audit Logs</h2>
    <table>
      <thead>
        <tr>
          <th>Timestamp</th><th>Actor</th><th>Query</th><th>Chunks</th>
          <th>Latency</th><th>Model</th><th>Seq</th><th>Signature</th>
        </tr>
      </thead>
      <tbody id="tbody"><tr><td colspan="8" class="empty">Loading...</td></tr></tbody>
    </table>
  </div>

  <script>
    function esc(v) {
      return String(v == null ? '' : v)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }
    async function load() {
      const [sumRes, logsRes, verRes] = await Promise.all([
        fetch('/api/summary'), fetch('/api/logs?limit=50'), fetch('/api/verify')
      ]);
      const sum = await sumRes.json();
      const logs = await logsRes.json();
      const ver = await verRes.json();

      document.getElementById('total').textContent = sum.total_queries ?? 0;
      document.getElementById('latency').textContent = sum.avg_latency_ms ? sum.avg_latency_ms + 'ms' : '--';
      document.getElementById('chunks').textContent = sum.avg_chunks_retrieved ?? '--';
      const chainEl = document.getElementById('chain');
      chainEl.textContent = ver.ok ? 'intact' : 'BROKEN';
      chainEl.className = ver.ok ? 'ok' : 'bad';
      document.getElementById('last').textContent = sum.last_query_at
        ? new Date(sum.last_query_at).toLocaleString() : '--';

      const tbody = document.getElementById('tbody');
      if (!logs.logs.length) {
        tbody.innerHTML = '<tr><td colspan="8" class="empty">No audit records yet.</td></tr>';
        return;
      }
      tbody.innerHTML = logs.logs.map(r => `
        <tr>
          <td>${esc(new Date(r.timestamp).toLocaleString())}</td>
          <td>${esc((r.actor || {}).id || '')}</td>
          <td title="${esc(r.query)}">${esc(r.query)}</td>
          <td>${(r.retrieved_chunks || []).length}</td>
          <td class="ms">${esc(r.latency_ms)}ms</td>
          <td>${esc(r.model_name || '')}</td>
          <td>${esc(r.sequence ?? '')}</td>
          <td class="sig">${esc((r.record_hash || r.chain_signature || '').slice(0, 16))}</td>
        </tr>`).join('');
    }
    load();
    setInterval(load, 30000);
  </script>
</body>
</html>
"""

_app_instance: FastAPI | None = None


def __getattr__(name: str) -> Any:
    # ``uvicorn ragcompliance.app:app`` resolves ``app`` lazily so importing
    # this module has no side effects (AUDIT.md A07).
    if name == "app":
        global _app_instance
        if _app_instance is None:
            _app_instance = create_app()
        return _app_instance
    raise AttributeError(name)
