"""Unified command line: ``ragcompliance <subcommand>``.

Subcommands:

``verify``     verify a hash chain (sink, JSONL file or evidence pack)
``evidence``   export a zipped evidence pack for an auditor
``soc2``       render the Markdown evidence summary only
``purge``      apply the retention policy (explicit, never automatic)
``show``       print per chain counts and head hashes
``selftest``   check that an install is wired correctly

The 0.1.x console scripts (``ragcompliance-selftest``,
``ragcompliance-soc2``) keep working and ``ragcompliance-verify`` is an
alias for ``ragcompliance verify``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any

from ._version import __version__
from .config import RAGComplianceConfig


def _open_storage(sink_url: str | None, workspace: str | None = None) -> Any:
    from .storage import AuditStorage

    cfg = RAGComplianceConfig.from_env()
    cfg.async_writes = False
    if sink_url:
        cfg.sink_url = sink_url
    if workspace:
        cfg.workspace_id = workspace
    if not cfg.sink_url and not (cfg.supabase_url and cfg.supabase_key) and not cfg.dev_mode:
        print(
            "error: no sink. Pass --sink URL or set RAGCOMPLIANCE_SINK / Supabase credentials.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return AuditStorage(cfg)


def _cmd_evidence(args: argparse.Namespace) -> int:
    from .evidence import export_evidence_pack

    storage = _open_storage(args.sink, args.workspace)
    try:
        pack = export_evidence_pack(
            storage,
            workspace_id=args.workspace,
            start=args.start,
            end=args.end,
            out_path=args.out,
            config=storage.config,
            sample_size=args.sample,
            seed=args.seed,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary = {
        "path": pack.path,
        "records": pack.manifest["record_count"],
        "chain_ok": pack.manifest["chain_ok"],
        "pack_hash": pack.pack_hash,
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"wrote {pack.path}")
        print(f"records   : {summary['records']}")
        print(f"chain ok  : {'yes' if summary['chain_ok'] else 'NO'}")
        print(f"pack hash : {summary['pack_hash']}")
    return 0 if pack.manifest["chain_ok"] else 1


def _cmd_purge(args: argparse.Namespace) -> int:
    from .retention import PurgeNotSupported, RetentionPolicy

    storage = _open_storage(args.sink, args.workspace)
    days = args.days if args.days is not None else storage.config.retention_days
    if days is None:
        print("error: no retention period. Pass --days or set RAGCOMPLIANCE_RETENTION_DAYS.", file=sys.stderr)
        return 2
    policy = RetentionPolicy(days=days)
    cutoff = policy.cutoff(datetime.now(timezone.utc))
    assert cutoff is not None
    if args.dry_run:
        victims = list(
            storage.iter_records(workspace_id=args.workspace, end=cutoff, chain_id=args.chain_id)
        )
        print(f"dry run: {len(victims)} record(s) older than {cutoff.isoformat()} would be purged")
        return 0
    try:
        result = storage.purge_expired(policy, workspace_id=args.workspace, chain_id=args.chain_id)
    except PurgeNotSupported as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if result is None:
        print("nothing to do")
        return 0
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"purged {result.purged} record(s) before {result.cutoff.isoformat()}")
        if result.checkpoint:
            print(f"checkpoint record {result.checkpoint['id']} appended at sequence {result.checkpoint['sequence']}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    storage = _open_storage(args.sink, args.workspace)
    chains: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in storage.iter_records(workspace_id=args.workspace):
        key = (str(rec.get("workspace_id", "")), str(rec.get("chain_id", "")))
        info = chains.setdefault(key, {"records": 0, "first": None, "last": None, "head": ""})
        info["records"] += 1
        seq = rec.get("sequence")
        if seq is not None:
            if info["first"] is None or seq < info["first"]:
                info["first"] = seq
            if info["last"] is None or seq >= info["last"]:
                info["last"] = seq
                info["head"] = rec.get("record_hash", "")
    if args.json:
        print(json.dumps([{"workspace_id": k[0], "chain_id": k[1], **v} for k, v in chains.items()], indent=2))
    else:
        print(f"ragcompliance {__version__}: {storage.describe()}")
        if not chains:
            print("no records")
        for (ws, cid), info in sorted(chains.items()):
            print(
                f"{ws}/{cid}: {info['records']} records, seq {info['first']}..{info['last']}, "
                f"head {str(info['head'])[:16]}"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragcompliance", description="Tamper evident audit trails for RAG pipelines."
    )
    parser.add_argument("--version", action="version", version=f"ragcompliance {__version__}")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("verify", help="verify a hash chain", add_help=False)
    p.add_argument("args", nargs=argparse.REMAINDER)

    p = sub.add_parser("soc2", help="render the Markdown evidence summary", add_help=False)
    p.add_argument("args", nargs=argparse.REMAINDER)

    p = sub.add_parser("selftest", help="check the install", add_help=False)
    p.add_argument("args", nargs=argparse.REMAINDER)

    p = sub.add_parser("evidence", help="export a zipped evidence pack")
    p.add_argument("--workspace", required=True)
    p.add_argument("--start", required=True, help="ISO date")
    p.add_argument("--end", required=True, help="ISO date")
    p.add_argument("--out", required=True, help="output .zip path")
    p.add_argument("--sink", default=None)
    p.add_argument("--sample", type=int, default=25)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_evidence)

    p = sub.add_parser("purge", help="apply the retention policy")
    p.add_argument("--workspace", required=True)
    p.add_argument("--sink", default=None)
    p.add_argument("--chain-id", default=None)
    p.add_argument("--days", type=int, default=None, help="override RAGCOMPLIANCE_RETENTION_DAYS")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_purge)

    p = sub.add_parser("show", help="per chain counts and head hashes")
    p.add_argument("--workspace", default=None)
    p.add_argument("--sink", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_show)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Delegated subcommands own their argument parsing, so hand the rest of
    # the line over before our parser can object to their options.
    if argv and argv[0] == "verify":
        from .verify import main as verify_main

        return verify_main(argv[1:], prog="ragcompliance verify")
    if argv and argv[0] == "soc2":
        from .soc2 import _main as soc2_main

        return soc2_main(argv[1:])
    if argv and argv[0] == "selftest":
        from .selftest import main as selftest_main

        return selftest_main(argv[1:])

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    try:
        return int(args.func(args))
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2


def verify_entry() -> int:
    """Console script ``ragcompliance-verify``."""
    from .verify import main as verify_main

    return verify_main(prog="ragcompliance-verify")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
