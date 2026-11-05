"""Hash chain primitives for tamper evident audit logs.

This module is the only place that knows how a record is hashed. Every
sink, the verifier CLI, the evidence pack exporter and the legacy
``chain_signature`` all call into these functions, so there is exactly
one definition of "intact".

Two layers of integrity
-----------------------
``content_signature``
    SHA-256 over the canonical JSON of ``{"query", "chunks", "answer"}``
    where each chunk contributes ``content``, ``source_url`` and
    ``chunk_id``. This is the 0.1.x ``chain_signature`` and is kept
    byte for byte compatible so records written by 0.1.x still verify.
    It answers "has this record been edited?".

``record_hash`` / ``prev_hash`` / ``sequence``
    Each sealed record carries the hash of the record before it in the
    same chain (``prev_hash``), its own position (``sequence``) and the
    SHA-256 over the canonical JSON of the whole record including those
    two fields (``record_hash``). Editing, deleting, inserting or
    reordering any record changes a hash that the next record commits
    to, so the break is detectable. A chain is identified by the pair
    ``(workspace_id, chain_id)``; see DECISIONS.md D01 and D02.

``record_hmac`` (optional)
    HMAC-SHA256 over ``record_hash`` with a key the store never sees.
    A hash chain is tamper evident, not tamper proof: anyone who can
    rewrite the whole store can recompute every hash. The HMAC closes
    that gap for anyone without the key.

Canonical form
--------------
``canonical_json`` uses sorted keys, no whitespace, UTF-8 (``ensure_ascii``
off) and ``default=str`` for anything the ``json`` module cannot encode.
Floats are serialised by ``json`` which is stable across CPython
versions for finite values. ``NaN`` and infinity are rejected because
their serialisation is not canonical.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

GENESIS_HASH = "0" * 64
"""``prev_hash`` of the first record in a chain."""

RECORD_KIND_INVOCATION = "rag_invocation"
RECORD_KIND_RETENTION_CHECKPOINT = "retention_checkpoint"

CHAIN_FIELDS = ("chain_id", "sequence", "prev_hash", "record_hash", "record_hmac")
"""Fields assigned by the sink when a record is sealed."""

_HASH_EXCLUDED = ("record_hash", "record_hmac")


class ChainError(ValueError):
    """Raised when a record cannot be sealed or canonicalised."""


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def canonical_json(obj: Any) -> str:
    """Deterministic JSON used for every hash in the system."""
    try:
        return json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        )
    except ValueError as exc:  # allow_nan=False raises ValueError
        raise ChainError(f"record is not canonicalisable: {exc}") from exc


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------- #
# Content signature (legacy chain_signature)                            #
# --------------------------------------------------------------------- #


def _chunk_field(chunk: Any, name: str) -> Any:
    if isinstance(chunk, Mapping):
        return chunk.get(name, "")
    return getattr(chunk, name, "")


def content_signature(query: str, chunks: Iterable[Any], answer: str) -> str:
    """SHA-256 over ``(query, chunks[i].{content,source_url,chunk_id}, answer)``.

    Compatible with the 0.1.x ``chain_signature``. ``chunks`` may be
    ``RetrievedChunk`` objects or dicts.
    """
    payload = {
        "query": query,
        "chunks": [
            {
                "content": _chunk_field(c, "content"),
                "source_url": _chunk_field(c, "source_url"),
                "chunk_id": _chunk_field(c, "chunk_id"),
            }
            for c in chunks
        ],
        "answer": answer,
    }
    # NOTE: the legacy signature used json.dumps(sort_keys=True, default=str)
    # with default separators. Keep that exact form for compatibility.
    raw = json.dumps(payload, sort_keys=True, default=str)
    return sha256_hex(raw)


def verify_content_signature(record: Mapping[str, Any]) -> bool:
    """Recompute ``chain_signature`` from a record dict and compare."""
    try:
        expected = content_signature(
            record.get("query", "") or "",
            record.get("retrieved_chunks", []) or [],
            record.get("llm_answer", "") or "",
        )
    except Exception:
        return False
    return hmac.compare_digest(expected, str(record.get("chain_signature") or ""))


# --------------------------------------------------------------------- #
# Record hash                                                           #
# --------------------------------------------------------------------- #


def hashable_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """The record minus the fields that are derived from the hash."""
    return {k: v for k, v in record.items() if k not in _HASH_EXCLUDED}


def compute_record_hash(record: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical JSON of the record without ``record_hash``
    and ``record_hmac``. ``prev_hash``, ``sequence`` and ``chain_id`` must
    already be present."""
    return sha256_hex(canonical_json(hashable_payload(record)))


def compute_record_hmac(record_hash: str, key: bytes | str) -> str:
    if isinstance(key, str):
        key = key.encode("utf-8")
    return hmac.new(key, record_hash.encode("utf-8"), hashlib.sha256).hexdigest()


def seal(
    record: Mapping[str, Any],
    *,
    prev_hash: str,
    sequence: int,
    chain_id: str,
    hmac_key: bytes | str | None = None,
) -> dict[str, Any]:
    """Return a new dict with the chain fields filled in.

    The input is not mutated. ``sequence`` must be a positive integer and
    ``prev_hash`` a 64 character hex string (``GENESIS_HASH`` for the
    first record).
    """
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
        raise ChainError(f"sequence must be a positive int, got {sequence!r}")
    if not _is_hex64(prev_hash):
        raise ChainError("prev_hash must be a 64 character hex string")
    if not chain_id:
        raise ChainError("chain_id must be a non-empty string")

    sealed = dict(record)
    sealed["chain_id"] = chain_id
    sealed["sequence"] = sequence
    sealed["prev_hash"] = prev_hash
    sealed.pop("record_hash", None)
    sealed.pop("record_hmac", None)
    sealed["record_hash"] = compute_record_hash(sealed)
    sealed["record_hmac"] = (
        compute_record_hmac(sealed["record_hash"], hmac_key) if hmac_key else ""
    )
    return sealed


def _is_hex64(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


# --------------------------------------------------------------------- #
# Verification                                                          #
# --------------------------------------------------------------------- #


@dataclass
class Problem:
    """One verification failure."""

    code: str
    detail: str
    sequence: int | None = None
    record_id: str = ""
    workspace_id: str = ""
    chain_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "detail": self.detail,
            "sequence": self.sequence,
            "record_id": self.record_id,
            "workspace_id": self.workspace_id,
            "chain_id": self.chain_id,
        }


@dataclass
class ChainSummary:
    workspace_id: str
    chain_id: str
    records: int = 0
    first_sequence: int | None = None
    last_sequence: int | None = None
    head_hash: str = ""
    anchor: str = GENESIS_HASH
    anchored_by_checkpoint: bool = False
    problems: list[Problem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "chain_id": self.chain_id,
            "records": self.records,
            "first_sequence": self.first_sequence,
            "last_sequence": self.last_sequence,
            "head_hash": self.head_hash,
            "anchor": self.anchor,
            "anchored_by_checkpoint": self.anchored_by_checkpoint,
            "ok": self.ok,
            "problems": [p.to_dict() for p in self.problems],
        }


@dataclass
class VerificationResult:
    """Outcome of verifying a set of records."""

    records_checked: int = 0
    chains: list[ChainSummary] = field(default_factory=list)
    problems: list[Problem] = field(default_factory=list)
    legacy_unchained: int = 0
    content_signature_failures: int = 0
    hmac_checked: bool = False

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "records_checked": self.records_checked,
            "legacy_unchained": self.legacy_unchained,
            "content_signature_failures": self.content_signature_failures,
            "hmac_checked": self.hmac_checked,
            "chains": [c.to_dict() for c in self.chains],
            "problems": [p.to_dict() for p in self.problems],
        }


def _has_chain_fields(record: Mapping[str, Any]) -> bool:
    return bool(record.get("record_hash")) and record.get("sequence") is not None


def verify_records(
    records: Iterable[Mapping[str, Any]],
    *,
    hmac_key: bytes | str | None = None,
    trust_anchors: Iterable[str] = (),
    require_chain: bool = False,
) -> VerificationResult:
    """Verify content signatures, hash links, ordering and (optionally) HMACs.

    Records may belong to several ``(workspace_id, chain_id)`` chains; each
    is verified independently. Records are sorted by ``sequence`` within a
    chain, so the input order does not matter.

    ``trust_anchors`` are ``record_hash`` values the caller knows were the
    last record before a purge. A chain whose first ``prev_hash`` matches
    an anchor, or matches the ``last_purged_hash`` of a retention
    checkpoint found later in the same chain, is reported as anchored
    rather than broken.

    When ``require_chain`` is true, records without chain fields are a
    failure. Otherwise they are counted in ``legacy_unchained`` and only
    their content signature is checked, which is the right behaviour for
    stores that still hold 0.1.x records.
    """
    result = VerificationResult(hmac_checked=hmac_key is not None)
    anchors = set(trust_anchors)
    by_chain: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for rec in records:
        rec = dict(rec)
        result.records_checked += 1
        kind = rec.get("kind") or RECORD_KIND_INVOCATION

        if kind == RECORD_KIND_INVOCATION and not verify_content_signature(rec):
            result.content_signature_failures += 1
            result.problems.append(
                Problem(
                    "content_signature_mismatch",
                    "chain_signature does not match query, chunks and answer",
                    sequence=rec.get("sequence"),
                    record_id=str(rec.get("id", "")),
                    workspace_id=str(rec.get("workspace_id", "")),
                    chain_id=str(rec.get("chain_id", "")),
                )
            )

        if not _has_chain_fields(rec):
            if require_chain:
                result.problems.append(
                    Problem(
                        "missing_chain_fields",
                        "record has no record_hash or sequence",
                        record_id=str(rec.get("id", "")),
                        workspace_id=str(rec.get("workspace_id", "")),
                    )
                )
            else:
                result.legacy_unchained += 1
            continue

        key = (str(rec.get("workspace_id", "")), str(rec.get("chain_id", "")))
        by_chain.setdefault(key, []).append(rec)

    for (workspace_id, chain_id), chain_records in sorted(by_chain.items()):
        summary = _verify_one_chain(workspace_id, chain_id, chain_records, hmac_key, anchors)
        result.chains.append(summary)
        result.problems.extend(summary.problems)

    return result


def _verify_one_chain(
    workspace_id: str,
    chain_id: str,
    records: list[dict[str, Any]],
    hmac_key: bytes | str | None,
    anchors: set[str],
) -> ChainSummary:
    summary = ChainSummary(workspace_id=workspace_id, chain_id=chain_id)

    def problem(code: str, detail: str, rec: Mapping[str, Any]) -> None:
        summary.problems.append(
            Problem(
                code,
                detail,
                sequence=rec.get("sequence"),
                record_id=str(rec.get("id", "")),
                workspace_id=workspace_id,
                chain_id=chain_id,
            )
        )

    try:
        records.sort(key=lambda r: int(r["sequence"]))
    except (TypeError, ValueError):
        summary.problems.append(
            Problem(
                "bad_sequence",
                "a record has a non-integer sequence",
                workspace_id=workspace_id,
                chain_id=chain_id,
            )
        )
        return summary

    checkpoint_anchors = {
        str((r.get("extra") or {}).get("last_purged_hash", ""))
        for r in records
        if (r.get("kind") == RECORD_KIND_RETENTION_CHECKPOINT)
    }
    checkpoint_anchors.discard("")

    summary.records = len(records)
    summary.first_sequence = int(records[0]["sequence"])
    summary.last_sequence = int(records[-1]["sequence"])

    first_prev = str(records[0].get("prev_hash", ""))
    summary.anchor = first_prev
    if first_prev != GENESIS_HASH:
        if first_prev in anchors or first_prev in checkpoint_anchors:
            summary.anchored_by_checkpoint = True
        else:
            problem(
                "unanchored_start",
                "first record does not start at genesis and no retention "
                "checkpoint or trust anchor explains the gap",
                records[0],
            )

    prev_hash = first_prev
    prev_seq = int(records[0]["sequence"]) - 1
    seen_ids: set[str] = set()
    for rec in records:
        seq = int(rec["sequence"])
        if seq != prev_seq + 1:
            if seq == prev_seq:
                problem("fork", f"duplicate sequence {seq}; two writers shared this chain", rec)
            else:
                problem("sequence_gap", f"expected sequence {prev_seq + 1}, found {seq}", rec)
        prev_seq = seq

        rid = str(rec.get("id", ""))
        if rid and rid in seen_ids:
            problem("duplicate_id", f"record id {rid} appears more than once", rec)
        seen_ids.add(rid)

        if str(rec.get("prev_hash", "")) != prev_hash:
            problem(
                "broken_link",
                "prev_hash does not match the previous record's record_hash",
                rec,
            )

        try:
            expected_hash = compute_record_hash(rec)
        except ChainError as exc:
            problem("uncanonical_record", str(exc), rec)
            expected_hash = ""
        stored_hash = str(rec.get("record_hash", ""))
        if not expected_hash or not hmac.compare_digest(expected_hash, stored_hash):
            problem("record_hash_mismatch", "record_hash does not match record content", rec)

        if hmac_key is not None:
            expected_mac = compute_record_hmac(stored_hash, hmac_key)
            if not hmac.compare_digest(expected_mac, str(rec.get("record_hmac", ""))):
                problem("hmac_mismatch", "record_hmac does not match the configured key", rec)

        prev_hash = stored_hash

    summary.head_hash = prev_hash
    return summary
