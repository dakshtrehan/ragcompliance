# Evidence packs

An evidence pack is a zip an auditor can open with standard tools and
re-verify with the CLI.

```bash
ragcompliance evidence --sink sqlite:///audit.db --workspace acme \
    --start 2026-01-01 --end 2026-03-31 --out acme-2026-q1.zip
```

```python
from ragcompliance.evidence import export_evidence_pack
pack = export_evidence_pack(storage, workspace_id="acme", start="2026-01-01", end="2026-03-31",
                            out_path="acme-2026-q1.zip", redactor=auditor.redactor)
pack.pack_hash
```

## Contents

| File | Content |
|------|---------|
| `manifest.json` | format and version, tool version, period, record counts, chain summary per `(workspace, chain)`, sink description (no secrets), dropped record counter, retention policy, redaction configuration, SHA-256 of every other file, `pack_hash` |
| `records.jsonl` | every record in scope, one canonical JSON object per line, in chain order |
| `verification.json` | the full verification result: chains, head hashes, anchors, problems |
| `report.md` | the evidence summary (same as `ragcompliance soc2`) |
| `controls.json`, `controls.md` | the control mapping with status and caveats |
| `retention.json` | policy and every retention checkpoint in the period |
| `redaction.json` | enabled patterns and hook names (never a salt) |
| `README.txt` | how to verify |

`pack_hash` is the SHA-256 over the sorted `name:sha256` lines of the file
table. Quote it in the audit workpaper; anyone with the zip can recompute
it.

## Verifying a pack

```bash
ragcompliance verify --evidence acme-2026-q1.zip
ragcompliance verify --evidence acme-2026-q1.zip --hmac-key-env RC_HMAC   # if records carry HMACs
```

The command checks every file hash against the manifest, recomputes
`pack_hash`, and re-runs chain and content signature verification over
`records.jsonl`. It needs nothing but the zip and the `ragcompliance`
package.

## The summary report

`report.md` (also available alone via `ragcompliance soc2`) contains:

* an executive summary: records and invocations in scope, unique sessions,
  distinct actors, content signature integrity (verified, failed, unsigned),
  hash chain state, latency, earliest and latest record, models observed,
  redaction findings;
* the hash chain verification table per chain;
* the control evidence matrix (see [Control mapping](control-mapping.md));
* a random sample of records for manual spot checks, seeded so an auditor
  can reproduce it with `--sample N --seed K`;
* the methodology.

Every record in the period is verified, not only the sample. The sample is
for a human to read.

## What a pack is not

It is not a SOC 2 report or an ISO 27001 certificate. Those are issued by
licensed auditors and accredited bodies after examining your whole system.
The pack is the part of the evidence that this component can produce
about itself, in a form that can be checked independently.
