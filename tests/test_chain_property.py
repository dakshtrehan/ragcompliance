"""Property based tests for the hash chain (hypothesis).

The integrity primitive is where a subtle bug costs the most, so instead
of a handful of examples these tests state the properties:

* any sequence of records sealed in order verifies;
* changing any hashed field of any record is detected;
* deleting or swapping any record is detected;
* canonical JSON does not depend on key order;
* sealing is deterministic.
"""

from __future__ import annotations

import copy
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from ragcompliance.chain import (
    GENESIS_HASH,
    canonical_json,
    compute_record_hash,
    content_signature,
    seal,
    verify_records,
)
from ragcompliance.models import AuditRecord, RetrievedChunk

text = st.text(min_size=0, max_size=40)
scores = st.one_of(st.none(), st.floats(min_value=-1.0, max_value=1.0, allow_nan=False))

chunk_st = st.builds(
    RetrievedChunk,
    content=text,
    source_url=text,
    chunk_id=text,
    similarity_score=scores,
    metadata=st.dictionaries(st.text(max_size=8), st.one_of(text, st.integers(), st.booleans()), max_size=3),
)

record_st = st.builds(
    AuditRecord,
    session_id=text,
    workspace_id=st.just("ws"),
    query=text,
    retrieved_chunks=st.lists(chunk_st, max_size=3),
    llm_answer=text,
    model_name=text,
    latency_ms=st.integers(min_value=0, max_value=10_000),
    extra=st.dictionaries(st.text(max_size=8), st.one_of(text, st.integers()), max_size=3),
    actor=st.dictionaries(st.sampled_from(["id", "type", "ip"]), text, max_size=3),
)


def _sealed_chain(records: list[AuditRecord], key: bytes | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    prev = GENESIS_HASH
    for i, rec in enumerate(records, start=1):
        rec.chain_signature = content_signature(rec.query, rec.retrieved_chunks, rec.llm_answer)
        sealed = seal(rec.to_dict(), prev_hash=prev, sequence=i, chain_id="default", hmac_key=key)
        out.append(sealed)
        prev = sealed["record_hash"]
    return out


HASHED_FIELDS = (
    "query", "llm_answer", "model_name", "session_id", "latency_ms", "timestamp",
    "prev_hash", "sequence", "actor", "extra", "retrieved_chunks", "kind",
)


@settings(max_examples=60, deadline=None)
@given(st.lists(record_st, min_size=1, max_size=6), st.one_of(st.none(), st.binary(min_size=1, max_size=16)))
def test_sealed_chain_verifies(records, key):
    chain = _sealed_chain(records, key)
    res = verify_records(chain, hmac_key=key)
    assert res.ok, res.to_dict()
    assert res.records_checked == len(records)
    assert res.chains[0].head_hash == chain[-1]["record_hash"]


@settings(max_examples=80, deadline=None)
@given(
    st.lists(record_st, min_size=1, max_size=5),
    st.data(),
)
def test_any_field_mutation_is_detected(records, data):
    chain = _sealed_chain(records)
    idx = data.draw(st.integers(min_value=0, max_value=len(chain) - 1))
    field = data.draw(st.sampled_from(HASHED_FIELDS))
    mutated = copy.deepcopy(chain)
    original = mutated[idx][field]
    if isinstance(original, str):
        mutated[idx][field] = original + "x"
    elif isinstance(original, int):
        mutated[idx][field] = original + 1
    elif isinstance(original, list):
        mutated[idx][field] = original + [{"content": "x", "source_url": "y", "chunk_id": "z"}]
    elif isinstance(original, dict):
        mutated[idx][field] = {**original, "__tamper__": 1}
    else:
        mutated[idx][field] = "tampered"
    res = verify_records(mutated)
    assert not res.ok
    codes = {p.code for p in res.problems}
    assert codes & {"record_hash_mismatch", "content_signature_mismatch", "broken_link", "sequence_gap", "fork"}


@settings(max_examples=60, deadline=None)
@given(st.lists(record_st, min_size=2, max_size=6), st.data())
def test_mutation_with_rehash_breaks_a_link_or_signature(records, data):
    chain = _sealed_chain(records)
    idx = data.draw(st.integers(min_value=0, max_value=len(chain) - 1))
    mutated = copy.deepcopy(chain)
    mutated[idx]["extra"] = {**mutated[idx]["extra"], "__tamper__": True}
    mutated[idx]["record_hash"] = compute_record_hash(mutated[idx])
    res = verify_records(mutated)
    if idx == len(chain) - 1:
        # Rewriting the newest record and its hash is only caught by an
        # external anchor (exported head hash) or the HMAC key.
        assert res.ok
        assert res.chains[0].head_hash != chain[-1]["record_hash"]
    else:
        assert {p.code for p in res.problems} == {"broken_link"}


@settings(max_examples=60, deadline=None)
@given(st.lists(record_st, min_size=2, max_size=6), st.data())
def test_deleting_any_interior_record_is_detected(records, data):
    chain = _sealed_chain(records)
    idx = data.draw(st.integers(min_value=0, max_value=len(chain) - 2))
    mutated = chain[:idx] + chain[idx + 1 :]
    res = verify_records(mutated)
    assert not res.ok
    codes = {p.code for p in res.problems}
    assert codes & {"sequence_gap", "broken_link", "unanchored_start"}


@settings(max_examples=40, deadline=None)
@given(st.lists(record_st, min_size=2, max_size=6), st.data())
def test_swapping_sequence_numbers_is_detected(records, data):
    chain = _sealed_chain(records)
    i = data.draw(st.integers(min_value=0, max_value=len(chain) - 1))
    j = data.draw(st.integers(min_value=0, max_value=len(chain) - 1).filter(lambda x: x != i))
    mutated = copy.deepcopy(chain)
    mutated[i]["sequence"], mutated[j]["sequence"] = mutated[j]["sequence"], mutated[i]["sequence"]
    assert not verify_records(mutated).ok


@settings(max_examples=60, deadline=None)
@given(record_st)
def test_sealing_is_deterministic_and_order_independent(record):
    record.chain_signature = content_signature(record.query, record.retrieved_chunks, record.llm_answer)
    d = record.to_dict()
    shuffled = dict(reversed(list(d.items())))
    a = seal(d, prev_hash=GENESIS_HASH, sequence=1, chain_id="c")
    b = seal(shuffled, prev_hash=GENESIS_HASH, sequence=1, chain_id="c")
    assert a["record_hash"] == b["record_hash"]
    assert canonical_json(a) == canonical_json(b)


@settings(max_examples=40, deadline=None)
@given(record_st)
def test_round_trip_through_from_dict_preserves_hash(record):
    record.chain_signature = content_signature(record.query, record.retrieved_chunks, record.llm_answer)
    sealed = seal(record.to_dict(), prev_hash=GENESIS_HASH, sequence=1, chain_id="c")
    rebuilt = AuditRecord.from_dict(sealed).to_dict()
    assert compute_record_hash(rebuilt) == sealed["record_hash"]
