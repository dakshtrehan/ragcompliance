"""Verifier CLI: ``ragcompliance verify`` / ``ragcompliance-verify``.

Verifies hash chains and content signatures from one of three sources:

``--sink URL``       any sink URL (or RAGCOMPLIANCE_SINK / Supabase env)
``--file PATH``      a JSONL file of records (for example records.jsonl)
``--evidence PATH``  an evidence pack zip (also checks the manifest hashes)

Exit codes: 0 intact, 1 problems found, 2 usage or I/O error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from .chain import VerificationResult, verify_records
from .config import RAGComplianceConfig


def verify_jsonl_file(
    path: str,
    *,
    hmac_key: bytes | str | None = None,
    workspace_id: str | None = None,
    chain_id: str | None = None,
    trust_anchors: list[str] | None = None,
    require_chain: bool = False,
) -> VerificationResult:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if workspace_id is not None and rec.get("workspace_id") != workspace_id:
                continue
            if chain_id is not None and rec.get("chain_id") != chain_id:
                continue
            records.append(rec)
    return verify_records(
        records, hmac_key=hmac_key, trust_anchors=trust_anchors or (), require_chain=require_chain
    )


def _open_storage(sink_url: str | None) -> Any:
    from .storage import AuditStorage

    cfg = RAGComplianceConfig.from_env()
    cfg.async_writes = False
    if sink_url:
        cfg.sink_url = sink_url
    if not cfg.sink_url and not (cfg.supabase_url and cfg.supabase_key) and not cfg.dev_mode:
        raise SystemExit(
            "error: no sink. Pass --sink URL or set RAGCOMPLIANCE_SINK / Supabase credentials."
        )
    return AuditStorage(cfg)


def _format_human(result: VerificationResult, extra_problems: list[str] | None = None) -> str:
    lines: list[str] = []
    ok = result.ok and not extra_problems
    lines.append("ragcompliance verify")
    lines.append("--------------------")
    lines.append(f"records checked      : {result.records_checked}")
    lines.append(f"legacy unchained     : {result.legacy_unchained}")
    lines.append(f"hmac checked         : {'yes' if result.hmac_checked else 'no'}")
    lines.append(f"chains               : {len(result.chains)}")
    for c in result.chains:
        state = "intact" if c.ok else "BROKEN"
        anchor = " (anchored by checkpoint)" if c.anchored_by_checkpoint else ""
        lines.append(
            f"  - {c.workspace_id}/{c.chain_id}: seq {c.first_sequence}..{c.last_sequence}, "
            f"{c.records} records, head {c.head_hash[:16]}, {state}{anchor}"
        )
    for text in extra_problems or []:
        lines.append(f"  ! pack: {text}")
    for problem in result.problems:
        lines.append(
            f"  ! seq {problem.sequence} {problem.record_id}: {problem.code}: {problem.detail}"
        )
    lines.append("")
    lines.append("RESULT: INTACT" if ok else "RESULT: TAMPER EVIDENCE OR BROKEN CHAIN")
    return "\n".join(lines)


def build_parser(prog: str = "ragcompliance verify") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog, description="Verify the hash chain and content signatures of an audit log."
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--sink", help="Sink URL (default: RAGCOMPLIANCE_SINK or Supabase env)")
    src.add_argument("--file", help="JSONL file of records")
    src.add_argument("--evidence", help="Evidence pack zip")
    parser.add_argument("--workspace", help="Only this workspace")
    parser.add_argument("--chain-id", help="Only this chain id")
    parser.add_argument(
        "--hmac-key-env",
        help="Environment variable holding the HMAC key (default RAGCOMPLIANCE_CHAIN_HMAC_KEY)",
    )
    parser.add_argument(
        "--trust-anchor",
        action="append",
        default=[],
        help="record_hash of a purged record the chain may start after (repeatable)",
    )
    parser.add_argument(
        "--require-chain",
        action="store_true",
        help="Treat records without chain fields (0.1.x) as failures",
    )
    parser.add_argument("--json", action="store_true", help="Machine readable output")
    return parser


def main(argv: list[str] | None = None, prog: str = "ragcompliance verify") -> int:
    parser = build_parser(prog)
    args = parser.parse_args(argv)

    key_env = args.hmac_key_env or "RAGCOMPLIANCE_CHAIN_HMAC_KEY"
    hmac_key: str | None = os.getenv(key_env) or None

    extra_problems: list[str] = []
    try:
        if args.evidence:
            from .evidence import verify_evidence_pack

            extra_problems, result = verify_evidence_pack(args.evidence, hmac_key=hmac_key)
        elif args.file:
            result = verify_jsonl_file(
                args.file,
                hmac_key=hmac_key,
                workspace_id=args.workspace,
                chain_id=args.chain_id,
                trust_anchors=args.trust_anchor,
                require_chain=args.require_chain,
            )
        else:
            storage = _open_storage(args.sink)
            result = verify_records(
                storage.iter_records(workspace_id=args.workspace, chain_id=args.chain_id),
                hmac_key=hmac_key,
                trust_anchors=args.trust_anchor,
                require_chain=args.require_chain,
            )
    except SystemExit as exc:
        if exc.code and isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            return 2
        raise
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        payload = result.to_dict()
        payload["pack_problems"] = extra_problems
        payload["ok"] = result.ok and not extra_problems
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_format_human(result, extra_problems))
    return 0 if (result.ok and not extra_problems) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(prog="python -m ragcompliance.verify"))
