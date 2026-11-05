"""Evidence pack exporter.

An evidence pack is a zip file an auditor can open with standard tools
and re-verify with ``ragcompliance verify --evidence pack.zip``::

    manifest.json        format, tool version, period, counts, sink and retention
                         description, SHA-256 of every other file, pack_hash
    records.jsonl        every record in scope, one canonical JSON object per line
    verification.json    hash chain and content signature verification result
    report.md            the SOC 2 / ISO 27001 evidence summary (soc2.generate_report)
    controls.json        the control mapping with status and caveats
    controls.md          the same as a Markdown table
    retention.json       configured retention policy and purge checkpoints in scope
    redaction.json       enabled redaction patterns and hooks (never the salt)
    README.txt           how to verify the pack

``pack_hash`` is the SHA-256 over the sorted ``name:sha256`` lines of the
file table, so the manifest commits to the contents and a single hash
can be quoted in an audit workpaper.
"""

from __future__ import annotations

import json
import os
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ._version import __version__
from .chain import (
    RECORD_KIND_RETENTION_CHECKPOINT,
    VerificationResult,
    canonical_json,
    sha256_hex,
    verify_records,
)
from .config import RAGComplianceConfig
from .controls import controls_json, controls_markdown
from .retention import RetentionPolicy
from .soc2 import _compute_stats, _format_md, _parse_date, _sample_records

EVIDENCE_FORMAT = "ragcompliance-evidence-pack"
EVIDENCE_FORMAT_VERSION = 1

README_TEXT = """ragcompliance evidence pack
===========================

This archive was produced by ragcompliance {version}. It contains every audit
record for one workspace in the reporting period, the result of verifying the
hash chain and content signatures over those records, a Markdown summary,
and the control mapping the summary refers to.

To re-verify independently:

    pip install ragcompliance
    ragcompliance verify --evidence <this file>

The command recomputes the SHA-256 of every file against manifest.json,
recomputes pack_hash, and re-runs the chain verification over records.jsonl.
If an HMAC key was configured when the records were written, pass
--hmac-key-env NAME_OF_ENV_VAR to also check record_hmac.

Nothing in this archive is a SOC 2 attestation or an ISO 27001 certificate.
"""


@dataclass
class EvidencePack:
    path: str
    manifest: dict[str, Any]
    verification: VerificationResult

    @property
    def pack_hash(self) -> str:
        return str(self.manifest.get("pack_hash", ""))


def _records_from(storage: Any, workspace_id: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
    iterator = getattr(storage, "iter_records", None)
    if not callable(iterator):
        raise TypeError("storage must expose iter_records(); pass an AuditStorage or a sink")
    return list(iterator(workspace_id=workspace_id, start=start, end=end))


def build_manifest(
    *,
    workspace_id: str,
    start: datetime,
    end: datetime,
    records: list[dict[str, Any]],
    verification: VerificationResult,
    files: dict[str, bytes],
    storage_description: dict[str, Any] | None,
    retention: RetentionPolicy,
    redaction: dict[str, Any] | None,
) -> dict[str, Any]:
    file_hashes = {name: sha256_hex(data) for name, data in sorted(files.items())}
    pack_hash = sha256_hex("\n".join(f"{n}:{h}" for n, h in sorted(file_hashes.items())))
    checkpoints = [r for r in records if r.get("kind") == RECORD_KIND_RETENTION_CHECKPOINT]
    return {
        "format": EVIDENCE_FORMAT,
        "format_version": EVIDENCE_FORMAT_VERSION,
        "tool_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "workspace_id": workspace_id,
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "record_count": len(records),
        "invocation_count": len(records) - len(checkpoints),
        "retention_checkpoint_count": len(checkpoints),
        "chain_ok": verification.ok,
        "chain_problem_count": len(verification.problems),
        "chains": [c.to_dict() for c in verification.chains],
        "sink": (storage_description or {}).get("sink") if storage_description else None,
        "dropped_count": (storage_description or {}).get("dropped_count", 0),
        "retention": retention.to_dict(),
        "redaction": redaction,
        "files": file_hashes,
        "pack_hash": pack_hash,
    }


def export_evidence_pack(
    storage: Any,
    *,
    workspace_id: str,
    start: str | datetime,
    end: str | datetime,
    out_path: str | os.PathLike[str],
    config: RAGComplianceConfig | None = None,
    redactor: Any | None = None,
    sample_size: int = 25,
    seed: int | None = None,
    records: Iterable[dict[str, Any]] | None = None,
) -> EvidencePack:
    """Write a zipped evidence pack and return its manifest.

    ``storage`` is an ``AuditStorage`` or any sink. ``records`` may be
    supplied directly (then ``storage`` may be ``None``).
    """
    cfg = config or (getattr(storage, "config", None) if storage is not None else None) or RAGComplianceConfig.from_env()
    start_dt, end_dt = _parse_date(start), _parse_date(end)
    if end_dt <= start_dt:
        raise ValueError("end must be strictly after start")

    if records is not None:
        loaded = list(records)
    else:
        loaded = _records_from(storage, workspace_id, start_dt, end_dt)
    loaded.sort(
        key=lambda r: (str(r.get("chain_id", "")), int(r.get("sequence") or 0), str(r.get("timestamp", "")))
    )

    hmac_key = cfg.hmac_key_bytes
    verification = verify_records(loaded, hmac_key=hmac_key)
    stats = _compute_stats(loaded, hmac_key=hmac_key)
    sampled = _sample_records(loaded, n=sample_size, seed=seed)

    describe = getattr(storage, "describe", None)
    storage_description = describe() if callable(describe) else None
    if storage_description is not None and "sink" not in storage_description:
        storage_description = {"sink": storage_description, "dropped_count": 0}

    retention = RetentionPolicy(days=cfg.retention_days)
    redaction_desc: dict[str, Any] | None = None
    if redactor is not None and hasattr(redactor, "describe"):
        redaction_desc = redactor.describe()
    elif cfg.redact_pii:
        from .redaction import DEFAULT_PATTERN_ORDER

        redaction_desc = {
            "patterns": list(cfg.redaction_patterns or DEFAULT_PATTERN_ORDER),
            "hooks": [],
            "replacement": cfg.redaction_replacement,
        }

    source = f"{((storage_description or {}).get('sink') or {}).get('type', 'audit')} sink"
    report_md = _format_md(workspace_id, start_dt, end_dt, stats, sampled, source=source)

    files: dict[str, bytes] = {
        "records.jsonl": ("\n".join(canonical_json(r) for r in loaded) + ("\n" if loaded else "")).encode("utf-8"),
        "verification.json": json.dumps(verification.to_dict(), indent=2, sort_keys=True).encode("utf-8"),
        "report.md": report_md.encode("utf-8"),
        "controls.json": json.dumps(controls_json(), indent=2).encode("utf-8"),
        "controls.md": controls_markdown().encode("utf-8"),
        "retention.json": json.dumps(
            {
                "policy": retention.to_dict(),
                "checkpoints": [
                    {
                        "id": r.get("id"),
                        "timestamp": r.get("timestamp"),
                        "sequence": r.get("sequence"),
                        **(r.get("extra") or {}),
                    }
                    for r in loaded
                    if r.get("kind") == RECORD_KIND_RETENTION_CHECKPOINT
                ],
            },
            indent=2,
            default=str,
        ).encode("utf-8"),
        "redaction.json": json.dumps(redaction_desc, indent=2, default=str).encode("utf-8"),
        "README.txt": README_TEXT.format(version=__version__).encode("utf-8"),
    }
    manifest = build_manifest(
        workspace_id=workspace_id,
        start=start_dt,
        end=end_dt,
        records=loaded,
        verification=verification,
        files=files,
        storage_description=storage_description,
        retention=retention,
        redaction=redaction_desc,
    )

    out = os.fspath(out_path)
    parent = os.path.dirname(os.path.abspath(out))
    os.makedirs(parent, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        for name, data in sorted(files.items()):
            zf.writestr(name, data)
    return EvidencePack(path=out, manifest=manifest, verification=verification)


def read_evidence_pack(path: str | os.PathLike[str]) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Return ``(manifest, files)`` from a pack without verifying it."""
    with zipfile.ZipFile(os.fspath(path)) as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        files = {name: zf.read(name) for name in zf.namelist() if name != "manifest.json"}
    return manifest, files


def verify_evidence_pack(
    path: str | os.PathLike[str], *, hmac_key: bytes | str | None = None
) -> tuple[list[str], VerificationResult]:
    """Check file hashes, ``pack_hash`` and re-run chain verification.

    Returns ``(file_problems, verification)``. The pack is sound when
    ``file_problems`` is empty and ``verification.ok`` is true.
    """
    manifest, files = read_evidence_pack(path)
    problems: list[str] = []
    if manifest.get("format") != EVIDENCE_FORMAT:
        problems.append("manifest.format is not a ragcompliance evidence pack")
    expected = manifest.get("files") or {}
    for name, digest in expected.items():
        data = files.get(name)
        if data is None:
            problems.append(f"missing file: {name}")
        elif sha256_hex(data) != digest:
            problems.append(f"hash mismatch: {name}")
    for name in files:
        if name not in expected:
            problems.append(f"unexpected file not in manifest: {name}")
    pack_hash = sha256_hex("\n".join(f"{n}:{h}" for n, h in sorted(expected.items())))
    if pack_hash != manifest.get("pack_hash"):
        problems.append("pack_hash does not match the file table")

    records: list[dict[str, Any]] = []
    raw = files.get("records.jsonl", b"")
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    verification = verify_records(records, hmac_key=hmac_key)
    return problems, verification
