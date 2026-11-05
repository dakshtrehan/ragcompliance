# Retention

A retention policy states how long records are kept. Enforcement is
explicit: nothing is deleted unless you run the purge.

## Configure

```bash
RAGCOMPLIANCE_RETENTION_DAYS=365
```

```python
from ragcompliance import RetentionPolicy
policy = RetentionPolicy(days=365)
policy.cutoff()        # now minus 365 days
```

## Purge

```bash
ragcompliance purge --workspace acme --dry-run       # count only
ragcompliance purge --workspace acme                 # uses RAGCOMPLIANCE_RETENTION_DAYS
ragcompliance purge --workspace acme --days 90       # override
```

```python
result = storage.purge_expired(workspace_id="acme")
result.purged, result.last_purged_sequence, result.checkpoint["id"]
```

Schedule it from cron or your orchestrator. Purges are per workspace and
chain.

## What a purge does to the chain

Only a prefix of a chain can be removed without leaving holes, so the purge
stops at the first record that is newer than the cutoff, even if an older
timestamped record appears later (clock skew). After deleting, it appends a
`retention_checkpoint` record:

```json
{
  "kind": "retention_checkpoint",
  "session_id": "retention",
  "actor": {"id": "ragcompliance-retention", "type": "system"},
  "extra": {"purged": 412, "cutoff": "2025-11-05T00:00:00+00:00",
            "last_purged_sequence": 412, "last_purged_hash": "0e44ba25..."}
}
```

The first surviving record's `prev_hash` equals `last_purged_hash`, and the
verifier reports the chain as `anchored_by_checkpoint` rather than broken.
If every record was purged, the checkpoint itself continues the sequence
from the last purged record so the chain never restarts at genesis.

## Sinks

All sinks support purging except S3 with Object Lock (deletes are refused
by the store) and Supabase layout 1 (migrate to layout 2 first). The
PostgreSQL purge needs `DELETE` on the table; the application role used for
writing can be `INSERT` and `SELECT` only.

## Evidence

The evidence pack includes `retention.json` with the configured policy and
every checkpoint in the period, so an auditor sees both the rule and the
proof it was applied.
