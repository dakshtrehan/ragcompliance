"""ragcompliance-selftest: verify that an install is wired correctly.

Checks (in order):

1. ragcompliance import and version readback
2. optional extras present on this install (informational)
3. a sink is configured for non dev mode operation
4. ``RAGComplianceConfig.from_env()`` loads
5. round trip: write two records to an in-memory sink, verify the hash
   chain, verify the content signature, and confirm tampering is detected

Exit code 0 means every red level check passed. Yellow checks (missing
optional extras) are informational.

Usage::

    ragcompliance selftest
    ragcompliance selftest --dev-mode     # skip the sink check
    ragcompliance selftest --json
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

from ._version import __version__

logger = logging.getLogger(__name__)

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_DIM = "\033[2m"
_RESET = "\033[0m"

SEVERITY_RED = "red"
SEVERITY_YELLOW = "yellow"
SEVERITY_GREEN = "green"


@dataclass
class CheckResult:
    name: str
    ok: bool
    severity: str
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


EXTRAS = {
    "langchain_core": "langchain (pip install 'ragcompliance[langchain]')",
    "llama_index.core": "llamaindex (pip install 'ragcompliance[llamaindex]')",
    "openai": "openai (pip install 'ragcompliance[openai]')",
    "anthropic": "anthropic (pip install 'ragcompliance[anthropic]')",
    "supabase": "supabase (pip install 'ragcompliance[supabase]')",
    "psycopg": "postgres (pip install 'ragcompliance[postgres]')",
    "boto3": "s3 (pip install 'ragcompliance[s3]')",
    "fastapi": "dashboard (pip install 'ragcompliance[dashboard]')",
    "authlib": "sso (pip install 'ragcompliance[sso]')",
    "stripe": "billing (pip install 'ragcompliance[billing]')",
}


def _check_optional_extras() -> list[CheckResult]:
    results: list[CheckResult] = []
    for module, hint in EXTRAS.items():
        try:
            importlib.import_module(module)
            results.append(
                CheckResult(f"extras.{module}", True, SEVERITY_GREEN, f"{module} available")
            )
        except ImportError:
            results.append(
                CheckResult(f"extras.{module}", False, SEVERITY_YELLOW, f"not installed: {hint}")
            )
    return results


def _check_sink_configured(dev_mode_override: bool) -> CheckResult:
    dev_mode = dev_mode_override or os.getenv("RAGCOMPLIANCE_DEV_MODE", "false").lower() == "true"
    if dev_mode:
        return CheckResult(
            "sink.configured", True, SEVERITY_GREEN,
            "dev_mode=true, no sink required (in-memory sink + stdout)", {"dev_mode": True},
        )
    sink = os.getenv("RAGCOMPLIANCE_SINK", "").strip()
    if sink:
        return CheckResult(
            "sink.configured", True, SEVERITY_GREEN, f"RAGCOMPLIANCE_SINK={sink.split('://')[0]}://...",
        )
    if os.getenv("RAGCOMPLIANCE_SUPABASE_URL") and os.getenv("RAGCOMPLIANCE_SUPABASE_KEY"):
        return CheckResult("sink.configured", True, SEVERITY_GREEN, "Supabase credentials present")
    return CheckResult(
        "sink.configured", False, SEVERITY_RED,
        "no sink: set RAGCOMPLIANCE_SINK (jsonl://, sqlite://, postgresql://, s3://) or "
        "RAGCOMPLIANCE_SUPABASE_URL/_KEY, or RAGCOMPLIANCE_DEV_MODE=true for local use",
    )


def _check_config() -> CheckResult:
    try:
        from .config import RAGComplianceConfig

        config = RAGComplianceConfig.from_env()
        return CheckResult(
            "config.load", True, SEVERITY_GREEN,
            f"workspace_id={config.workspace_id} dev_mode={config.dev_mode} "
            f"async_writes={config.async_writes} chain_id={config.chain_id} "
            f"hmac={'on' if config.chain_hmac_key else 'off'} "
            f"redact_pii={config.redact_pii}",
            {
                "workspace_id": config.workspace_id,
                "dev_mode": config.dev_mode,
                "async_writes": config.async_writes,
                "chain_id": config.chain_id,
                "hmac": bool(config.chain_hmac_key),
                "redact_pii": config.redact_pii,
            },
        )
    except Exception as exc:
        return CheckResult(
            "config.load", False, SEVERITY_RED, f"RAGComplianceConfig.from_env() raised: {exc!r}"
        )


def _check_chain_roundtrip() -> CheckResult:
    """Write two records to an in-memory sink, verify, tamper, verify again."""
    try:
        from .auditor import Auditor
        from .chain import verify_records
        from .config import RAGComplianceConfig
        from .models import RetrievedChunk
        from .sinks import MemorySink
        from .storage import AuditStorage
    except ImportError as exc:  # pragma: no cover
        return CheckResult("chain.roundtrip", False, SEVERITY_RED, f"import failed: {exc!r}")

    config = RAGComplianceConfig.from_env()
    config.dev_mode = False
    config.async_writes = False
    config.enabled = True
    sink = MemorySink(chain_id=config.chain_id, hmac_key=config.hmac_key_bytes)
    storage = AuditStorage(config, sink=sink)
    auditor = Auditor(config, session_id="selftest", storage=storage, billing=None, alerter=None)
    for i in range(2):
        with auditor.trace(f"selftest query {i}") as t:
            t.add_chunk(RetrievedChunk("example chunk", "example.pdf", f"selftest-{i}", 0.99))
            t.set_answer("selftest works.", model_name="selftest-model")

    result = verify_records(sink.records, hmac_key=config.hmac_key_bytes)
    if not result.ok or result.records_checked != 2:
        return CheckResult(
            "chain.roundtrip", False, SEVERITY_RED,
            f"fresh chain did not verify: {[p.code for p in result.problems]}",
        )
    tampered = [dict(r) for r in sink.records]
    tampered[0]["llm_answer"] = "tampered"
    detect = verify_records(tampered, hmac_key=config.hmac_key_bytes)
    codes = {p.code for p in detect.problems}
    if detect.ok or "content_signature_mismatch" not in codes or "record_hash_mismatch" not in codes:
        return CheckResult(
            "chain.roundtrip", False, SEVERITY_RED, f"tampering was not detected: {sorted(codes)}"
        )
    return CheckResult(
        "chain.roundtrip", True, SEVERITY_GREEN,
        "2 records sealed, chain verified, tampering detected",
        {"head_hash": result.chains[0].head_hash if result.chains else ""},
    )


def run_selftest(dev_mode: bool = False) -> list[CheckResult]:
    results: list[CheckResult] = [_check_version()]
    results.extend(_check_optional_extras())
    results.append(_check_sink_configured(dev_mode_override=dev_mode))
    results.append(_check_config())
    results.append(_check_chain_roundtrip())
    return results


def _format_human(results: list[CheckResult]) -> str:
    stdout_tty = sys.stdout.isatty()

    def style(color: str, s: str) -> str:
        return f"{color}{s}{_RESET}" if stdout_tty else s

    lines: list[str] = ["", style(_DIM, "ragcompliance-selftest"), style(_DIM, "-" * 22)]
    for r in results:
        if r.severity == SEVERITY_GREEN:
            mark = style(_GREEN, "  ok ")
        elif r.severity == SEVERITY_YELLOW:
            mark = style(_YELLOW, "info ")
        else:
            mark = style(_RED, "FAIL ")
        lines.append(f"{mark} {r.name.ljust(24)} {r.detail}")
    lines.append("")
    red_failures = [r for r in results if r.severity == SEVERITY_RED and not r.ok]
    if red_failures:
        lines.append(
            style(
                _RED,
                f"{len(red_failures)} blocking check(s) failed. Fix these before pointing "
                "production chains at this install.",
            )
        )
    else:
        lines.append(style(_GREEN, "All blocking checks passed."))
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ragcompliance-selftest",
        description="Verify that the current ragcompliance install is wired correctly.",
    )
    parser.add_argument("--dev-mode", action="store_true", help="Skip the sink check.")
    parser.add_argument("--json", action="store_true", help="Machine readable output.")
    args = parser.parse_args(argv)

    results = run_selftest(dev_mode=args.dev_mode)
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        print(_format_human(results))
    red_failures = [r for r in results if r.severity == SEVERITY_RED and not r.ok]
    return 0 if not red_failures else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
