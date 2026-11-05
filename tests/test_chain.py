"""Unit tests for ragcompliance.chain."""

from __future__ import annotations

import hashlib
import json

import pytest

from ragcompliance.chain import (
    GENESIS_HASH,
    ChainError,
    canonical_json,
    compute_record_hash,
    compute_record_hmac,
    content_signature,
    hashable_payload,
    seal,
    verify_content_signature,
    verify_records,
)
from ragcompliance.models import AuditRecord, RetrievedChunk


def _rec(i: int, workspace: str = "ws") -> dict:
    r = AuditRecord(
        session_id="s",
        workspace_id=workspace,
        query=f"q{i}",
        retrieved_chunks=[RetrievedChunk(f"c{i}", "src", f"id{i}", 0.5)],
        llm_answer=f"a{i}",
    )
    r.chain_signature = content_signature(r.query, r.retrieved_chunks, r.llm_answer)
    return r.to_dict()


def _chain(n: int, hmac_key: bytes | None = None, workspace: str = "ws") -> list[dict]:
    out: list[dict] = []
    prev = GENESIS_HASH
    for i in range(1, n + 1):
        sealed = seal(_rec(i, workspace), prev_hash=prev, sequence=i, chain_id="default", hmac_key=hmac_key)
        out.append(sealed)
        prev = sealed["record_hash"]
    return out


class TestCanonicalJson:
    def test_key_order_independent(self):
        assert canonical_json({"b": 1, "a": [1, {"d": 2, "c": 3}]}) == canonical_json(
            {"a": [1, {"c": 3, "d": 2}], "b": 1}
        )

    def test_no_whitespace_and_unicode_preserved(self):
        assert canonical_json({"k": "é"}) == '{"k":"é"}'

    def test_rejects_nan(self):
        with pytest.raises(ChainError):
            canonical_json({"x": float("nan")})

    def test_datetime_and_objects_fall_back(self):
        from datetime import datetime, timezone

        dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert canonical_json({"t": dt}) == '{"t":"2026-01-01T00:00:00+00:00"}'
        assert canonical_json({"c": RetrievedChunk("x", "y", "z")}).startswith('{"c":{"chunk_id":"z"')


class TestContentSignature:
    def test_matches_legacy_formula(self):
        chunks = [{"content": "c", "source_url": "u", "chunk_id": "i", "similarity_score": 0.9}]
        payload = {
            "query": "q",
            "chunks": [{"content": "c", "source_url": "u", "chunk_id": "i"}],
            "answer": "a",
        }
        legacy = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        assert content_signature("q", chunks, "a") == legacy
        assert content_signature("q", [RetrievedChunk("c", "u", "i", 0.9)], "a") == legacy

    def test_verify_content_signature(self):
        rec = _rec(1)
        assert verify_content_signature(rec)
        rec["llm_answer"] = "tampered"
        assert not verify_content_signature(rec)
        assert not verify_content_signature({"chain_signature": None})
        assert not verify_content_signature({"retrieved_chunks": 5, "chain_signature": "x"})


class TestSeal:
    def test_seal_sets_fields_and_is_pure(self):
        rec = _rec(1)
        sealed = seal(rec, prev_hash=GENESIS_HASH, sequence=1, chain_id="c")
        assert rec["record_hash"] == ""  # input not mutated
        assert sealed["sequence"] == 1 and sealed["prev_hash"] == GENESIS_HASH
        assert sealed["record_hash"] == compute_record_hash(sealed)
        assert sealed["record_hmac"] == ""
        assert hashable_payload(sealed).keys() == (set(sealed) - {"record_hash", "record_hmac"})

    def test_hmac(self):
        sealed = seal(_rec(1), prev_hash=GENESIS_HASH, sequence=1, chain_id="c", hmac_key="k")
        assert sealed["record_hmac"] == compute_record_hmac(sealed["record_hash"], b"k")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"prev_hash": GENESIS_HASH, "sequence": 0, "chain_id": "c"},
            {"prev_hash": GENESIS_HASH, "sequence": True, "chain_id": "c"},
            {"prev_hash": "zz", "sequence": 1, "chain_id": "c"},
            {"prev_hash": "g" * 64, "sequence": 1, "chain_id": "c"},
            {"prev_hash": GENESIS_HASH, "sequence": 1, "chain_id": ""},
        ],
    )
    def test_seal_validates(self, kwargs):
        with pytest.raises(ChainError):
            seal(_rec(1), **kwargs)


class TestVerify:
    def test_intact(self):
        res = verify_records(_chain(5))
        assert res.ok and res.records_checked == 5 and len(res.chains) == 1
        c = res.chains[0]
        assert c.first_sequence == 1 and c.last_sequence == 5 and c.anchor == GENESIS_HASH
        assert res.to_dict()["ok"] is True

    def test_input_order_does_not_matter(self):
        chain = _chain(4)
        assert verify_records(reversed(chain)).ok

    def test_edit_detected(self):
        chain = _chain(3)
        chain[1]["query"] = "evil"
        codes = {p.code for p in verify_records(chain).problems}
        assert codes == {"content_signature_mismatch", "record_hash_mismatch"}

    def test_edit_with_rehash_breaks_next_link(self):
        chain = _chain(3)
        chain[1]["extra"] = {"x": 1}
        chain[1]["record_hash"] = compute_record_hash(chain[1])
        codes = {p.code for p in verify_records(chain).problems}
        assert codes == {"broken_link"}

    def test_deletion_detected(self):
        chain = _chain(4)
        del chain[1]
        codes = {p.code for p in verify_records(chain).problems}
        assert {"sequence_gap", "broken_link"} == codes

    def test_truncation_of_head_is_undetectable_without_anchor(self):
        # Dropping the newest records leaves a shorter but valid chain. This is
        # the known limit of hash chains and why head hashes should be exported.
        chain = _chain(4)[:2]
        assert verify_records(chain).ok

    def test_insertion_detected(self):
        chain = _chain(3)
        forged = seal(_rec(99), prev_hash=chain[2]["record_hash"], sequence=4, chain_id="default")
        forged["sequence"] = 3  # pretend it replaces record 3
        codes = {p.code for p in verify_records(chain + [forged]).problems}
        assert "fork" in codes

    def test_duplicate_id(self):
        chain = _chain(2)
        chain[1]["id"] = chain[0]["id"]
        chain[1]["record_hash"] = compute_record_hash(chain[1])
        codes = {p.code for p in verify_records(chain).problems}
        assert "duplicate_id" in codes

    def test_hmac_checked_only_with_key(self):
        chain = _chain(2, hmac_key=b"k")
        assert verify_records(chain).ok
        assert verify_records(chain, hmac_key=b"k").ok
        bad = verify_records(chain, hmac_key=b"other")
        assert {p.code for p in bad.problems} == {"hmac_mismatch"}
        assert bad.hmac_checked

    def test_unanchored_start_and_trust_anchor(self):
        chain = _chain(4)[2:]
        res = verify_records(chain)
        assert {p.code for p in res.problems} == {"unanchored_start"}
        anchored = verify_records(chain, trust_anchors=[chain[0]["prev_hash"]])
        assert anchored.ok and anchored.chains[0].anchored_by_checkpoint

    def test_legacy_records_counted_not_failed(self):
        legacy = _rec(1)  # no chain fields
        res = verify_records([legacy])
        assert res.ok and res.legacy_unchained == 1
        strict = verify_records([legacy], require_chain=True)
        assert {p.code for p in strict.problems} == {"missing_chain_fields"}

    def test_multiple_chains_verified_independently(self):
        a = _chain(2, workspace="a")
        b = _chain(3, workspace="b")
        b[0]["llm_answer"] = "x"
        res = verify_records(a + b)
        assert len(res.chains) == 2
        by_ws = {c.workspace_id: c for c in res.chains}
        assert by_ws["a"].ok and not by_ws["b"].ok

    def test_bad_sequence_type(self):
        chain = _chain(1)
        chain[0]["sequence"] = "one"
        res = verify_records(chain)
        assert {p.code for p in res.problems} == {"bad_sequence"}

    def test_uncanonical_record(self):
        chain = _chain(1)
        chain[0]["extra"] = {"nan": float("nan")}
        codes = {p.code for p in verify_records(chain).problems}
        assert "uncanonical_record" in codes and "record_hash_mismatch" in codes

    def test_checkpoint_anchors_chain(self):
        from ragcompliance.retention import build_retention_checkpoint

        full = _chain(4)
        survivors = full[2:]
        checkpoint = build_retention_checkpoint(
            workspace_id="ws",
            cutoff=__import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").timezone.utc),
            purged=2,
            last_purged_sequence=2,
            last_purged_hash=full[1]["record_hash"],
        )
        sealed_cp = seal(checkpoint, prev_hash=survivors[-1]["record_hash"], sequence=5, chain_id="default")
        res = verify_records(survivors + [sealed_cp])
        assert res.ok and res.chains[0].anchored_by_checkpoint
