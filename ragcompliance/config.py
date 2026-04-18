import os
from dataclasses import dataclass


@dataclass
class RAGComplianceConfig:
    supabase_url: str = ""
    supabase_key: str = ""
    workspace_id: str = "default"
    table_name: str = "rag_audit_logs"
    enabled: bool = True
    # If True, logs to stdout when Supabase is not configured (dev mode)
    dev_mode: bool = False
    # If True, hard-blocks the chain when the workspace is over its plan quota.
    # Default is soft-warn only so quota misconfiguration never bricks a prod chain.
    enforce_quota: bool = False
    # Async write path: when True, save() enqueues the record for a background
    # worker thread and returns immediately so the chain's hot path is never
    # blocked on Supabase latency. Turn off only if you need a strictly
    # synchronous audit write (e.g. tests inspecting storage mid-chain).
    async_writes: bool = True
    # Bounded in-memory queue. If Supabase stalls or is unreachable, records
    # pile up here. Beyond this, save() logs a warning and drops the record
    # rather than leaking memory. 1000 = roughly 1MB per 1KB record, which is
    # plenty of buffer for any realistic outage.
    async_max_queue: int = 1000
    # Seconds to wait on process shutdown for the worker to finish draining.
    async_shutdown_timeout: float = 5.0

    @classmethod
    def from_env(cls) -> "RAGComplianceConfig":
        return cls(
            supabase_url=os.getenv("RAGCOMPLIANCE_SUPABASE_URL", ""),
            supabase_key=os.getenv("RAGCOMPLIANCE_SUPABASE_KEY", ""),
            workspace_id=os.getenv("RAGCOMPLIANCE_WORKSPACE_ID", "default"),
            table_name=os.getenv("RAGCOMPLIANCE_TABLE_NAME", "rag_audit_logs"),
            enabled=os.getenv("RAGCOMPLIANCE_ENABLED", "true").lower() == "true",
            dev_mode=os.getenv("RAGCOMPLIANCE_DEV_MODE", "false").lower() == "true",
            enforce_quota=os.getenv("RAGCOMPLIANCE_ENFORCE_QUOTA", "false").lower() == "true",
            async_writes=os.getenv("RAGCOMPLIANCE_ASYNC_WRITES", "true").lower() == "true",
            async_max_queue=int(os.getenv("RAGCOMPLIANCE_ASYNC_MAX_QUEUE", "1000")),
            async_shutdown_timeout=float(
                os.getenv("RAGCOMPLIANCE_ASYNC_SHUTDOWN_TIMEOUT", "5.0")
            ),
        )
