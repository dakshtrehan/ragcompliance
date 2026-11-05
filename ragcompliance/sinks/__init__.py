"""Pluggable audit sinks.

Select a sink with a URL (``RAGCOMPLIANCE_SINK`` or
``RAGComplianceConfig.sink_url``)::

    memory://
    jsonl:///var/log/ragcompliance/audit.jsonl?fsync=1
    sqlite:///var/lib/ragcompliance/audit.db?table=rag_audit_records
    postgresql://user:password@host:5432/db?table=rag_audit_records
    s3://bucket/prefix?endpoint_url=https://minio.local&object_lock_mode=COMPLIANCE&object_lock_days=365
    supabase://            (credentials from RAGCOMPLIANCE_SUPABASE_URL / _KEY)

Heavy dependencies (boto3, psycopg, supabase) are imported only when
their sink is constructed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, unquote, urlparse

from .base import AuditSink
from .memory import MemorySink

if TYPE_CHECKING:  # pragma: no cover
    from ..config import RAGComplianceConfig

__all__ = [
    "AuditSink",
    "JSONLSink",
    "MemorySink",
    "PostgresSink",
    "S3Sink",
    "SQLiteSink",
    "SupabaseSink",
    "make_sink",
]


def __getattr__(name: str) -> Any:
    # Lazy attribute access so ``from ragcompliance.sinks import S3Sink``
    # works without importing every backend module at package import.
    if name == "JSONLSink":
        from .jsonl import JSONLSink

        return JSONLSink
    if name == "SQLiteSink":
        from .sqlite import SQLiteSink

        return SQLiteSink
    if name == "PostgresSink":
        from .postgres import PostgresSink

        return PostgresSink
    if name == "S3Sink":
        from .s3 import S3Sink

        return S3Sink
    if name == "SupabaseSink":
        from .supabase import SupabaseSink

        return SupabaseSink
    raise AttributeError(name)


def _flag(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def make_sink(url: str, config: RAGComplianceConfig | None = None, **overrides: Any) -> AuditSink:
    """Build a sink from a URL. ``config`` supplies chain id, HMAC key and
    Supabase credentials; keyword overrides win over both."""
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    query = {k: v[-1] for k, v in parse_qs(parsed.query, keep_blank_values=True).items()}

    common: dict[str, Any] = {}
    if config is not None:
        common["chain_id"] = config.chain_id
        common["hmac_key"] = config.hmac_key_bytes
    if "chain_id" in query:
        common["chain_id"] = query["chain_id"]
    common.update({k: v for k, v in overrides.items() if k in ("chain_id", "hmac_key")})
    extra = {k: v for k, v in overrides.items() if k not in ("chain_id", "hmac_key")}

    if scheme == "memory":
        return MemorySink(**common)

    if scheme == "jsonl":
        from .jsonl import JSONLSink

        path = _path_from(parsed)
        return JSONLSink(path, fsync=_flag(query.get("fsync")), **common, **extra)

    if scheme == "sqlite":
        from .sqlite import SQLiteSink

        path = _path_from(parsed)
        kwargs: dict[str, Any] = {}
        if "table" in query:
            kwargs["table"] = query["table"]
        kwargs.update(extra)
        return SQLiteSink(path, **common, **kwargs)

    if scheme in ("postgres", "postgresql"):
        from .postgres import PostgresSink

        kwargs = {}
        if "table" in query:
            kwargs["table"] = query["table"]
        if "create_schema" in query:
            kwargs["create_schema"] = _flag(query["create_schema"], True)
        kwargs.update(extra)
        # Strip our own options from the DSN before handing it to psycopg.
        clean_query = "&".join(
            f"{k}={v}"
            for k, v in query.items()
            if k not in ("table", "create_schema", "chain_id")
        )
        dsn = parsed._replace(query=clean_query).geturl()
        return PostgresSink(dsn, **common, **kwargs)

    if scheme == "s3":
        from .s3 import S3Sink

        bucket = parsed.netloc
        prefix = unquote(parsed.path).strip("/") or "ragcompliance"
        kwargs = {}
        for opt in ("endpoint_url", "region_name", "object_lock_mode"):
            if opt in query:
                kwargs[opt] = query[opt]
        if "object_lock_days" in query:
            kwargs["object_lock_days"] = int(query["object_lock_days"])
        kwargs.update(extra)
        return S3Sink(bucket, prefix, **common, **kwargs)

    if scheme == "supabase":
        from .supabase import SupabaseSink

        if config is None:
            raise ValueError("supabase:// sink needs a RAGComplianceConfig for credentials")
        kwargs = {
            "table": config.table_name,
            "schema_version": config.supabase_schema_version,
        }
        kwargs.update(extra)
        return SupabaseSink(config.supabase_url, config.supabase_key, **common, **kwargs)

    raise ValueError(
        f"unknown sink URL {url!r}; expected one of memory://, jsonl://, sqlite://, "
        "postgresql://, s3://, supabase://"
    )


def _path_from(parsed: Any) -> str:
    """``jsonl:///abs/path`` -> ``/abs/path``; ``jsonl://rel/path`` -> ``rel/path``."""
    path = unquote(parsed.path or "")
    if parsed.netloc:
        path = parsed.netloc + path
    elif path.startswith("//"):
        path = path[1:]
    return path or ":memory:"
