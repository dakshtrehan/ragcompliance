# Hash chain and verification

Every record carries two layers of integrity.

## Content signature (`chain_signature`)

SHA-256 over the canonical JSON of:

```json
{"query": "...", "chunks": [{"content": "...", "source_url": "...", "chunk_id": "..."}], "answer": "..."}
```

This is the 0.1.x signature, kept byte for byte compatible, so records
written by older versions still verify. It answers one question: has this
record's query, retrieved content or answer been edited?

Similarity scores, chunk metadata, model name, latency and timestamps are
not part of it. They are covered by the record hash below.

## Record hash and chain

When a sink stores a record it seals it:

| Field | Meaning |
|-------|---------|
| `chain_id` | Chain name (default `default`). One chain per `(workspace_id, chain_id)`. |
| `sequence` | Position in the chain, starting at 1, no gaps. |
| `prev_hash` | `record_hash` of the previous record in the chain, or 64 zeros for the first. |
| `record_hash` | SHA-256 over the canonical JSON of the whole record (every field except `record_hash` and `record_hmac`), including `prev_hash` and `sequence`. |
| `record_hmac` | HMAC-SHA256 of `record_hash` with `RAGCOMPLIANCE_CHAIN_HMAC_KEY`, when set. |

Because each record commits to the hash of the one before it:

* editing any field of record N changes `record_hash(N)`, which no longer
  matches what record N+1 committed to (`broken_link`), and if the editor
  also recomputes the hash of N, the verifier still flags `broken_link` on
  N+1;
* deleting record N leaves a `sequence_gap` and a `broken_link`;
* inserting a record produces a `fork` (duplicate sequence) or a gap;
* reordering changes the `prev_hash` relationships.

Canonical JSON means sorted keys, no whitespace, UTF-8, and a rejection of
`NaN` and infinity. The implementation is `ragcompliance.chain` and is the
only place hashes are computed.

## What the chain cannot prove on its own

* **Truncation at the head.** Deleting the newest records leaves a shorter
  chain that still verifies. Record the head hash (`ragcompliance show`) in
  a place the writer cannot modify, or export evidence packs regularly;
  each pack commits to the head hash of the period.
* **Rewriting everything.** Someone who can rewrite the whole store can
  recompute every hash. The optional HMAC key defeats this for anyone who
  does not hold the key, which is why the key should live outside the
  store (a secret manager, a KMS). A WORM store (S3 Object Lock) or a
  database the writer cannot `UPDATE` or `DELETE` in (the SQLite and
  PostgreSQL sinks install triggers) achieves the same from the other side.
* **Clock accuracy.** Timestamps come from the writer's clock.

## Retention purges

Deleting old records necessarily breaks the chain where the deletion
happened. A purge therefore appends a `retention_checkpoint` record that
names the `record_hash` and `sequence` of the last purged record. The
verifier treats the first surviving record as anchored when a checkpoint
in the same chain names its `prev_hash`, and reports
`anchored_by_checkpoint: true` rather than a break. See
[Retention](retention.md).

## Verifier output

```
$ ragcompliance verify --sink sqlite:///audit.db
ragcompliance verify
--------------------
records checked      : 1204
legacy unchained     : 0
hmac checked         : yes
chains               : 2
  - acme/default: seq 1..1198, 1198 records, head 90675dacc51cebd6, intact
  - acme/batch-worker: seq 1..6, 6 records, head 027439054a3c0020, intact

RESULT: INTACT
```

Problem codes: `content_signature_mismatch`, `record_hash_mismatch`,
`broken_link`, `sequence_gap`, `fork`, `duplicate_id`, `unanchored_start`,
`hmac_mismatch`, `missing_chain_fields` (only with `--require-chain`),
`bad_sequence`, `uncanonical_record`.

`--json` prints the same as machine readable JSON. Exit code 0 means
intact, 1 means problems were found, 2 means the command could not run.

## Multiple writers

Chains are per `(workspace_id, chain_id)`. The SQLite, PostgreSQL and
JSONL sinks serialise concurrent writers (write lock, advisory lock, file
lock) so any number of processes can share one chain. Supabase with the
layout 2 migration uses a unique index and retries on conflict. Supabase
layout 1 and S3 cannot serialise writers across processes; give each
writer its own `RAGCOMPLIANCE_CHAIN_ID` there. A shared `chain_id` across
unsynchronised writers shows up as `fork` in the verifier, never as a
silently corrupted log.

## Programmatic use

```python
from ragcompliance import verify_records

result = verify_records(storage.iter_records(workspace_id="acme"), hmac_key=b"...")
result.ok, result.records_checked, [p.code for p in result.problems]
```
