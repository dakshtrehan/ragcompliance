"""
RAGCompliance FastAPI Dashboard
--------------------------------
Run with: uvicorn ragcompliance.dashboard.app:app --reload
"""

from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from ragcompliance.config import RAGComplianceConfig
from ragcompliance.storage import AuditStorage

app = FastAPI(
    title="RAGCompliance Dashboard",
    description="Audit trail viewer for RAG pipelines in regulated industries",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

config = RAGComplianceConfig.from_env()
storage = AuditStorage(config)


# ------------------------------------------------------------------ #
# API routes                                                          #
# ------------------------------------------------------------------ #

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/logs")
def get_logs(
    workspace_id: str | None = Query(None, description="Filter by workspace"),
    session_id: str | None = Query(None, description="Filter by session"),
    limit: int = Query(50, ge=1, le=500),
) -> dict[str, Any]:
    records = storage.query(
        workspace_id=workspace_id or config.workspace_id,
        session_id=session_id,
        limit=limit,
    )
    return {"count": len(records), "logs": records}


@app.get("/api/logs/{record_id}")
def get_log(record_id: str) -> dict[str, Any]:
    records = storage.query(workspace_id=config.workspace_id, limit=500)
    match = next((r for r in records if r.get("id") == record_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Audit record not found")
    return match


@app.get("/api/summary")
def get_summary() -> dict[str, Any]:
    records = storage.query(workspace_id=config.workspace_id, limit=500)
    if not records:
        return {"total_queries": 0}

    latencies = [r.get("latency_ms", 0) for r in records]
    chunk_counts = [len(r.get("retrieved_chunks", [])) for r in records]

    return {
        "workspace_id": config.workspace_id,
        "total_queries": len(records),
        "avg_latency_ms": round(sum(latencies) / len(latencies)),
        "avg_chunks_retrieved": round(sum(chunk_counts) / len(chunk_counts), 1),
        "last_query_at": records[0].get("timestamp") if records else None,
    }


# ------------------------------------------------------------------ #
# Minimal HTML dashboard (no frontend framework needed)              #
# ------------------------------------------------------------------ #

@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return """
<!DOCTYPE html>
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
    .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; padding: 32px 40px 0; }
    .stat { background: #1a1f2e; border: 1px solid #1e2535; border-radius: 8px; padding: 20px; }
    .stat label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; }
    .stat value { display: block; font-size: 28px; font-weight: 700; margin-top: 6px; color: #60a5fa; }
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
    .empty { text-align: center; padding: 48px; color: #475569; }
  </style>
</head>
<body>
  <header>
    <h1>RAGCompliance</h1>
    <span>Audit Dashboard</span>
  </header>

  <div class="stats">
    <div class="stat"><label>Total Queries</label><value id="total">--</value></div>
    <div class="stat"><label>Avg Latency</label><value id="latency">--</value></div>
    <div class="stat"><label>Avg Chunks</label><value id="chunks">--</value></div>
    <div class="stat"><label>Last Query</label><value id="last" style="font-size:13px;margin-top:10px">--</value></div>
  </div>

  <div class="logs">
    <h2>Recent Audit Logs</h2>
    <table>
      <thead>
        <tr>
          <th>Timestamp</th><th>Query</th><th>Chunks</th>
          <th>Latency</th><th>Model</th><th>Signature</th>
        </tr>
      </thead>
      <tbody id="tbody"><tr><td colspan="6" class="empty">Loading...</td></tr></tbody>
    </table>
  </div>

  <script>
    async function load() {
      const [sumRes, logsRes] = await Promise.all([
        fetch('/api/summary'), fetch('/api/logs?limit=50')
      ]);
      const sum = await sumRes.json();
      const logs = await logsRes.json();

      document.getElementById('total').textContent = sum.total_queries ?? 0;
      document.getElementById('latency').textContent = sum.avg_latency_ms ? sum.avg_latency_ms + 'ms' : '--';
      document.getElementById('chunks').textContent = sum.avg_chunks_retrieved ?? '--';
      document.getElementById('last').textContent = sum.last_query_at
        ? new Date(sum.last_query_at).toLocaleString() : '--';

      const tbody = document.getElementById('tbody');
      if (!logs.logs.length) {
        tbody.innerHTML = '<tr><td colspan="6" class="empty">No audit records yet.</td></tr>';
        return;
      }
      tbody.innerHTML = logs.logs.map(r => `
        <tr>
          <td>${new Date(r.timestamp).toLocaleString()}</td>
          <td title="${r.query}">${r.query}</td>
          <td>${(r.retrieved_chunks || []).length}</td>
          <td class="ms">${r.latency_ms}ms</td>
          <td>${r.model_name || '—'}</td>
          <td class="sig">${r.chain_signature?.slice(0, 16)}…</td>
        </tr>`).join('');
    }
    load();
    setInterval(load, 30000);
  </script>
</body>
</html>
"""
