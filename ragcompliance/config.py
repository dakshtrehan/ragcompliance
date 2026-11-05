"""Configuration for the audit middleware.

Every field has an environment variable twin read by ``from_env``. The
0.1.x fields and their variables are unchanged; the 1.0 additions are
grouped at the bottom.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_list(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [p.strip() for p in raw.split(",") if p.strip()]


@dataclass
class RAGComplianceConfig:
    # ---- storage (0.1.x) ---------------------------------------------- #
    supabase_url: str = ""
    supabase_key: str = ""
    workspace_id: str = "default"
    table_name: str = "rag_audit_logs"
    enabled: bool = True
    # If True, logs to stdout when no sink is configured (dev mode)
    dev_mode: bool = False
    # If True, hard-blocks the chain when the workspace is over its plan quota.
    enforce_quota: bool = False
    # Async write path: save() enqueues the record for a background worker.
    async_writes: bool = True
    # Bounded in-memory queue; beyond this records are dropped (or the
    # caller blocks, see async_overflow).
    async_max_queue: int = 1000
    # Seconds to wait on process shutdown for the worker to finish draining.
    async_shutdown_timeout: float = 5.0

    # ---- PII / PHI redaction (0.1.8) --------------------------------- #
    redact_pii: bool = False
    redaction_patterns: list[str] | None = None
    redaction_replacement: str = "[REDACTED:{name}]"

    # ---- sinks and hash chain (1.0) ---------------------------------- #
    # Sink URL. Examples: jsonl:///var/log/rag-audit.jsonl,
    # sqlite:///var/lib/ragcompliance/audit.db, postgresql://user:pw@host/db,
    # s3://bucket/prefix, supabase://, memory://. Empty means: use Supabase
    # when credentials are set, otherwise dev mode / warning.
    sink_url: str = ""
    # Chain identity. One chain per (workspace_id, chain_id). Set a distinct
    # value per writer process when several processes share a sink that
    # cannot serialise writers (Supabase, S3).
    chain_id: str = "default"
    # Optional HMAC key (string). When set, every sealed record carries
    # record_hmac so a store-level attacker cannot forge a consistent chain.
    chain_hmac_key: str = ""
    # Retention in days. None disables purging. Purges are explicit
    # (storage.purge_expired() or the CLI); nothing is deleted automatically.
    retention_days: int | None = None
    # What to do when the async queue is full: "drop" (log and count) or
    # "block" (wait for space, trading latency for completeness).
    async_overflow: str = "drop"
    # Supabase table layout: 1 = 0.1.x columns, v2 fields nested in extra;
    # 2 = dedicated columns after running supabase_migration_v1.sql.
    supabase_schema_version: int = 1

    # ---- capture (1.0) ------------------------------------------------ #
    # Persist an error record when a chain fails (off keeps 0.1.x behaviour).
    record_errors: bool = False
    # Store the prompt text in lineage. The prompt SHA-256 is always stored.
    capture_prompts: bool = False
    # Truncation limits for coerced query / answer. Truncation is recorded
    # in lineage so a signed record never hides it.
    max_query_chars: int = 2000
    max_answer_chars: int = 4000
    # Default actor attached to every record (per invocation actors override).
    actor_id: str = ""
    actor_type: str = "service"
    # Soft cap on in-flight run state held by a handler.
    max_pending_runs: int = 10000

    # ---- dashboard (1.0) ---------------------------------------------- #
    dashboard_allow_any_workspace: bool = False
    dashboard_cors_origins: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #

    @property
    def default_actor(self) -> dict[str, Any]:
        if not self.actor_id:
            return {}
        return {"id": self.actor_id, "type": self.actor_type}

    @property
    def hmac_key_bytes(self) -> bytes | None:
        return self.chain_hmac_key.encode("utf-8") if self.chain_hmac_key else None

    @classmethod
    def from_env(cls) -> RAGComplianceConfig:
        patterns_env = os.getenv("RAGCOMPLIANCE_REDACTION_PATTERNS", "").strip()
        patterns_list = (
            [p.strip() for p in patterns_env.split(",") if p.strip()]
            if patterns_env
            else None
        )
        retention_raw = os.getenv("RAGCOMPLIANCE_RETENTION_DAYS", "").strip()
        retention_days: int | None = None
        if retention_raw:
            try:
                retention_days = int(retention_raw)
            except ValueError:
                retention_days = None
        return cls(
            supabase_url=os.getenv("RAGCOMPLIANCE_SUPABASE_URL", ""),
            supabase_key=os.getenv("RAGCOMPLIANCE_SUPABASE_KEY", ""),
            workspace_id=os.getenv("RAGCOMPLIANCE_WORKSPACE_ID", "default"),
            table_name=os.getenv("RAGCOMPLIANCE_TABLE_NAME", "rag_audit_logs"),
            enabled=_env_bool("RAGCOMPLIANCE_ENABLED", True),
            dev_mode=_env_bool("RAGCOMPLIANCE_DEV_MODE", False),
            enforce_quota=_env_bool("RAGCOMPLIANCE_ENFORCE_QUOTA", False),
            async_writes=_env_bool("RAGCOMPLIANCE_ASYNC_WRITES", True),
            async_max_queue=_env_int("RAGCOMPLIANCE_ASYNC_MAX_QUEUE", 1000),
            async_shutdown_timeout=_env_float("RAGCOMPLIANCE_ASYNC_SHUTDOWN_TIMEOUT", 5.0),
            redact_pii=_env_bool("RAGCOMPLIANCE_REDACT_PII", False),
            redaction_patterns=patterns_list,
            redaction_replacement=os.getenv(
                "RAGCOMPLIANCE_REDACTION_REPLACEMENT", "[REDACTED:{name}]"
            ),
            sink_url=os.getenv("RAGCOMPLIANCE_SINK", "").strip(),
            chain_id=os.getenv("RAGCOMPLIANCE_CHAIN_ID", "default").strip() or "default",
            chain_hmac_key=os.getenv("RAGCOMPLIANCE_CHAIN_HMAC_KEY", ""),
            retention_days=retention_days,
            async_overflow=os.getenv("RAGCOMPLIANCE_ASYNC_OVERFLOW", "drop").strip().lower()
            or "drop",
            supabase_schema_version=_env_int("RAGCOMPLIANCE_SUPABASE_SCHEMA_VERSION", 1),
            record_errors=_env_bool("RAGCOMPLIANCE_RECORD_ERRORS", False),
            capture_prompts=_env_bool("RAGCOMPLIANCE_CAPTURE_PROMPTS", False),
            max_query_chars=_env_int("RAGCOMPLIANCE_MAX_QUERY_CHARS", 2000),
            max_answer_chars=_env_int("RAGCOMPLIANCE_MAX_ANSWER_CHARS", 4000),
            actor_id=os.getenv("RAGCOMPLIANCE_ACTOR_ID", "").strip(),
            actor_type=os.getenv("RAGCOMPLIANCE_ACTOR_TYPE", "service").strip() or "service",
            max_pending_runs=max(1, _env_int("RAGCOMPLIANCE_MAX_PENDING_RUNS", 10000)),
            dashboard_allow_any_workspace=_env_bool(
                "RAGCOMPLIANCE_DASHBOARD_ALLOW_ANY_WORKSPACE", False
            ),
            dashboard_cors_origins=_env_list("RAGCOMPLIANCE_DASHBOARD_CORS_ORIGINS"),
        )
