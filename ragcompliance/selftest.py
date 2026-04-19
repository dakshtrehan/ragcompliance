"""
ragcompliance-selftest: verify that an install is wired correctly.

Runs a series of sanity checks against the current environment and reports
red/yellow/green per check. Designed to be copy-pasteable into a CI job or a
fresh-venv smoke test so operators can confirm a deployment before pointing
production chains at it.

Checks (in order):
    1. ragcompliance import + version readback
    2. Optional extras available on this install (supabase, langchain_core,
       llama_index_core, fastapi, authlib)
    3. Required env vars for non-dev-mode operation
    4. RAGComplianceConfig.from_env() loads without raising
    5. AuditStorage constructs against the resolved config
    6. Round-trip save of a synthetic record (dev_mode path or Supabase)
    7. Signature recomputation on the synthetic record matches what was stored

Exit code 0 means every red-level check passed. Yellow checks (missing
optional extras) are informational and do not fail.

Usage:
    $ ragcompliance-selftest
    $ ragcompliance-selftest --dev-mode        # force stdout path
    $ ragcompliance-selftest --json            # machine-readable output
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from ragcompliance import __version__

logger = logging.getLogger(__name__)

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_DIM = "\033[2m"
_RESET = "\033[0m"

# Severity: red = failure, yellow = informational, green = ok
SEVERITY_RED = "red"
SEVERITY_YELLOW = "yellow"
SEVERITY_GREEN = "green"


@dataclass
class CheckResult:
    name: str
    ok: bool
    severity: str  # "red" | "yellow" | "green"
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "severity": self.severity,
            "detail": self.detail,
            "data": self.data,
        }


def _check_version() -> CheckResult:
    return CheckResult(
        name="ragcompliance.import",
        ok=True,
        severity=SEVERITY_GREEN,
        detail=f"ragcompliance {__version__} imported",
        data={"version": __version__},
    )


def _check_optional_extras() -> list[CheckResult]:
    """Yellow: missing extras are informational, not failures."""
    extras = {
        "supabase": "supabase (pip install 'ragcompliance[supabase]')",
        "langchain_core": "langchain (pip install langchain-core)",
        "llama_index.core": "llama-index (pip install 'ragcompliance[llamaindex]')",
        "fastapi": "dashboard (pip install 'ragcompliance[dashboard]')",
        "authlib": "sso (pip install 'ragcompliance[sso]')",
    }
    results: list[CheckResult] = []
    for module, hint in extras.items():
        try:
            importlib.import_module(module)
            results.append(
                CheckResult(
                    name=f"extras.{module}",
                    ok=True,
                    severity=SEVERITY_GREEN,
                    detail=f"{module} available",
                )
            )
        except ImportError:
            results.append(
                CheckResult(
                    name=f"extras.{module}",
                    ok=False,
                    severity=SEVERITY_YELLOW,
                    detail=f"not installed: {hint}",
                )
            )
    return results


def _check_env_vars(dev_mode_override: bool) -> CheckResult:
    """Red if non-dev-mode and required Supabase env vars are missing."""
    dev_mode = (
        dev_mode_override
        or os.getenv("RAGCOMPLIANCE_DEV_MODE", "false").lower() == "true"
    )
    if dev_mode:
        return CheckResult(
            name="env.vars",
            ok=True,
            severity=SEVERITY_GREEN,
            detail="dev_mode=true, Supabase env vars not required",
            data={"dev_mode": True},
        )
    required = ["RAGCOMPLIANCE_SUPABASE_URL", "RAGCOMPLIANCE_SUPABASE_KEY"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        return CheckResult(
            name="env.vars",
            ok=False,
            severity=SEVERITY_RED,
            detail=(
                f"missing: {', '.join(missing)}. Set RAGCOMPLIANCE_DEV_MODE=true "
                "for stdout-only dev, or fill these in for Supabase writes."
            ),
            data={"missing": missing},
        )
    return CheckResult(
        name="env.vars",
        ok=True,
        severity=SEVERITY_GREEN,
        detail="required env vars present",
    )


def _check_config() -> CheckResult:
    try:
        from ragcompliance import RAGComplianceConfig

        config = RAGComplianceConfig.from_env()
        return CheckResult(
            name="config.load",
            ok=True,
            severity=SEVERITY_GREEN,
            detail=(
                f"workspace_id={config.workspace_id} dev_mode={config.dev_mode} "
                f"async_writes={config.async_writes}"
            ),
            data={
                "workspace_id": config.workspace_id,
                "dev_mode": config.dev_mode,
                "async_writes": config.async_writes,
            },
        )
    except Exception as exc:  # pragma: no cover - defensive
        return CheckResult(
            name="config.load",
            ok=False,
            severity=SEVERITY_RED,
            detail=f"RAGComplianceConfig.from_env() raised: {exc!r}",
        )


def _check_storage_roundtrip(dev_mode_override: bool) -> CheckResult:
    """Dev-mode-only smoke test: build a storage, save a synthetic record,
    confirm the handler's SHA-256 signature matches on re-computation.
    """
    import hashlib
    import json as _json

    try:
        from ragcompliance import AuditRecord, AuditStorage, RAGComplianceConfig, RetrievedChunk
    except ImportError as exc:
        return CheckResult(
            name="storage.roundtrip",
            ok=False,
            severity=SEVERITY_RED,
            detail=f"could not import storage layer: {exc!r}",
        )

    def _sign(query: str, chunks: list, answer: str) -> str:
        """Mirror of RAGComplianceHandler._sign_chain so selftest stays
        self-contained and doesn't require instantiating a handler."""
        payload = {
            "query": query,
            "chunks": [
                {"content": c.content, "source_url": c.source_url, "chunk_id": c.chunk_id}
                for c in chunks
            ],
            "answer": answer,
        }
        raw = _json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    config = RAGComplianceConfig.from_env()
    if dev_mode_override:
        config.dev_mode = True
    # Only exercise the dev-mode stdout path from selftest.
    # Supabase round-trip is left to a live deployment: the selftest should
    # not silently write rows to anyone's production audit table.
    config.dev_mode = True

    storage = AuditStorage(config)
    chunks = [
        RetrievedChunk(
            content="example chunk",
            source_url="example.pdf",
            chunk_id="selftest-0",
            similarity_score=0.99,
        )
    ]
    record = AuditRecord(
        session_id="selftest",
        workspace_id=config.workspace_id,
        query="what does selftest look like?",
        retrieved_chunks=chunks,
        llm_answer="selftest works.",
        model_name="selftest-model",
        latency_ms=1,
    )
    record.chain_signature = _sign(
        record.query,
        record.retrieved_chunks,
        record.llm_answer,
    )

    try:
        ok = storage.save(record)
    except Exception as exc:
        return CheckResult(
            name="storage.roundtrip",
            ok=False,
            severity=SEVERITY_RED,
            detail=f"storage.save raised: {exc!r}",
        )

    if not ok:
        return CheckResult(
            name="storage.roundtrip",
            ok=False,
            severity=SEVERITY_RED,
            detail="storage.save returned False in dev_mode path",
        )

    recomputed = _sign(
        record.query, record.retrieved_chunks, record.llm_answer
    )
    if recomputed != record.chain_signature:
        return CheckResult(
            name="storage.roundtrip",
            ok=False,
            severity=SEVERITY_RED,
            detail="signature recomputation mismatch",
        )

    return CheckResult(
        name="storage.roundtrip",
        ok=True,
        severity=SEVERITY_GREEN,
        detail="dev-mode storage save + signature recomputation match",
    )


def run_selftest(dev_mode: bool = False) -> list[CheckResult]:
    """Run every check in order, return results in list order."""
    results: list[CheckResult] = []
    results.append(_check_version())
    results.extend(_check_optional_extras())
    results.append(_check_env_vars(dev_mode_override=dev_mode))
    results.append(_check_config())
    results.append(_check_storage_roundtrip(dev_mode_override=dev_mode))
    return results


def _format_human(results: list[CheckResult]) -> str:
    stdout_tty = sys.stdout.isatty()

    def style(color: str, s: str) -> str:
        if not stdout_tty:
            return s
        return f"{color}{s}{_RESET}"

    lines: list[str] = []
    lines.append("")
    lines.append(style(_DIM, "ragcompliance-selftest"))
    lines.append(style(_DIM, "-" * 22))
    for r in results:
        if r.severity == SEVERITY_GREEN:
            mark = style(_GREEN, "  ok ")
        elif r.severity == SEVERITY_YELLOW:
            mark = style(_YELLOW, "info ")
        else:
            mark = style(_RED, "FAIL ")
        name = r.name.ljust(24)
        lines.append(f"{mark} {name} {r.detail}")
    lines.append("")

    red_failures = [r for r in results if r.severity == SEVERITY_RED and not r.ok]
    if red_failures:
        lines.append(
            style(
                _RED,
                f"{len(red_failures)} blocking check(s) failed. "
                "Fix these before pointing production chains at this install.",
            )
        )
    else:
        lines.append(style(_GREEN, "All blocking checks passed."))
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ragcompliance-selftest",
        description=(
            "Verify that the current ragcompliance install is wired correctly. "
            "Exit 0 on all blocking checks passing, non-zero otherwise."
        ),
    )
    parser.add_argument(
        "--dev-mode",
        action="store_true",
        help="Force dev-mode checks (skip required Supabase env-var check).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the default human output.",
    )
    args = parser.parse_args(argv)

    results = run_selftest(dev_mode=args.dev_mode)
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        print(_format_human(results))

    # Exit non-zero only on red failures. Yellow (missing optional extras) is
    # informational: a minimal install intentionally doesn't pull every extra.
    red_failures = [r for r in results if r.severity == SEVERITY_RED and not r.ok]
    return 0 if not red_failures else 1


if __name__ == "__main__":
    sys.exit(main())
