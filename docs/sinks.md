# Sinks

A sink stores sealed records and reads them back in chain order. Select one
with `RAGCOMPLIANCE_SINK` (or `RAGComplianceConfig.sink_url`). Options go
in the URL query string. Heavy dependencies are imported only when their
sink is built.

| URL | Extra | Multi process safe chain | Append only enforcement | Purge |
|-----|-------|--------------------------|-------------------------|-------|
| `memory://` | none | n/a | no | yes |
| `jsonl:///path/audit.jsonl?fsync=1` | none | yes (`flock`) | file permissions | yes (atomic rewrite) |
| `sqlite:////path/audit.db?table=rag_audit_records` | none | yes (`BEGIN IMMEDIATE`) | triggers abort UPDATE and DELETE | yes (trigger dropped and recreated inside the transaction) |
| `postgresql://user:pw@host:5432/db?table=rag_audit_records` | `postgres` | yes (advisory lock) | trigger with a `SET LOCAL` escape for purges | yes |
| `s3://bucket/prefix?endpoint_url=...&object_lock_mode=COMPLIANCE&object_lock_days=365` | `s3` | no, use one `chain_id` per writer | Object Lock when enabled | yes, unless Object Lock |
| `supabase://` | `supabase` | layout 2 only | trigger (layout 2 migration) | layout 2 only |

Set `?chain_id=...` on any URL to override `RAGCOMPLIANCE_CHAIN_ID`.

## JSONL

One canonical JSON object per line. A sibling `.lock` file carries the
`flock` so several processes can append. `fsync=1` forces every write to
disk. A purge rewrites the file atomically and then appends the checkpoint.
Good for single hosts, containers with a mounted volume, and shipping to a
log pipeline that preserves lines.

## SQLite

Columns for the fields the dashboard filters on plus the full payload as
JSON. Two triggers reject any `UPDATE` or `DELETE`; the only way to delete
is the purge path, which runs inside the same transaction that recreates
the trigger. WAL mode is enabled. Good for one host or a small service.

## PostgreSQL

Same shape as SQLite. The schema is created on first use unless
`?create_schema=0` is given; `sql/postgres_schema.sql` holds the same DDL
for DBAs. The append only trigger function raises unless the session
setting `ragcompliance.allow_purge` is `on`, which the purge sets with
`SET LOCAL` for its transaction only. Writers take
`pg_advisory_xact_lock(hashtext(workspace_id || ':' || chain_id))`, so
many application instances can append to one chain. Give the application
role `INSERT` and `SELECT` only; the purge needs `DELETE` and can run
under a separate role.

## S3 compatible

Layout:

```
{prefix}/{workspace_id}/{chain_id}/{sequence:012d}-{record_id}.json
{prefix}/{workspace_id}/{chain_id}/_head.json
```

One object per record keeps every record independently addressable and
lets Object Lock make each one WORM. Pass `object_lock_mode` (`GOVERNANCE`
or `COMPLIANCE`) and `object_lock_days` and every record is written with a
retain until date; the bucket must have Object Lock enabled. With Object
Lock the purge refuses to run, which is the point. Works with MinIO, Ceph,
Cloudflare R2 and other S3 compatible services via `endpoint_url`.

## Supabase

The 0.1.x backend. Two table layouts (see [Migration](migration.md)):

* Layout 1 (default): the 0.1.x columns. New fields are nested under
  `extra._ragcompliance`. No migration needed. Chain continuity across
  several processes sharing a `chain_id` is not guaranteed.
* Layout 2: run `supabase_migration_v1.sql` and set
  `RAGCOMPLIANCE_SUPABASE_SCHEMA_VERSION=2`. Dedicated columns, a unique
  index on `(workspace_id, chain_id, sequence)` that makes concurrent
  writers retry instead of forking, an append only trigger, and purge
  support.

The library connects with the service role key, which bypasses row level
security. Tenant isolation at the store depends on who holds that key.

## Writing your own

Subclass `ragcompliance.sinks.base.AuditSink` and implement
`_tail_locked`, `_write_locked` and `_iter_all`; override `_locked` if the
backend offers a transaction or lock, and `_purge_locked` if it supports
deletion. The base class does the sealing, so a custom sink cannot get the
chain wrong by accident.

## Async writes and the dropped counter

`AuditStorage` wraps the sink. By default `save()` enqueues the record and a
daemon thread writes it, so the request path never waits on storage. The
queue is bounded (`RAGCOMPLIANCE_ASYNC_MAX_QUEUE`, default 1000). When it
is full the record is dropped and counted (`storage.dropped_count`, also
in every evidence pack manifest) or, with
`RAGCOMPLIANCE_ASYNC_OVERFLOW=block`, the caller waits. On shutdown an
`atexit` hook drains the queue for `RAGCOMPLIANCE_ASYNC_SHUTDOWN_TIMEOUT`
seconds. Set `RAGCOMPLIANCE_ASYNC_WRITES=false` for strictly synchronous
writes.
