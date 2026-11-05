# Command line

One entry point, `ragcompliance`, with subcommands. The 0.1.x scripts
`ragcompliance-selftest` and `ragcompliance-soc2` still work, and
`ragcompliance-verify` is an alias for `ragcompliance verify`.

All commands that read a store take `--sink URL`; without it they use
`RAGCOMPLIANCE_SINK`, then the Supabase credentials, then dev mode.

## verify

```bash
ragcompliance verify [--sink URL | --file records.jsonl | --evidence pack.zip]
                     [--workspace W] [--chain-id C] [--hmac-key-env VAR]
                     [--trust-anchor HASH ...] [--require-chain] [--json]
```

Exit 0 intact, 1 problems, 2 error. `--hmac-key-env` names the variable
holding the HMAC key (default `RAGCOMPLIANCE_CHAIN_HMAC_KEY`).
`--trust-anchor` accepts the `record_hash` of a purged record whose
checkpoint is not in the input. `--require-chain` fails on 0.1.x records
that have no chain fields instead of counting them.

## evidence

```bash
ragcompliance evidence --workspace W --start 2026-01-01 --end 2026-03-31 --out pack.zip
                       [--sink URL] [--sample 25] [--seed 42] [--json]
```

Writes the zip and prints the record count, chain state and `pack_hash`.
Exit 1 if the chain is not intact (the pack is still written so the
problems can be examined).

## soc2

```bash
ragcompliance soc2 --workspace W --start 2026-01-01 --end 2026-03-31 [--sink URL] [--sample 25] [--seed K] [--out report.md]
```

The Markdown summary alone, to stdout or a file.

## purge

```bash
ragcompliance purge --workspace W [--sink URL] [--chain-id C] [--days N] [--dry-run] [--json]
```

Applies the retention policy (`--days` or `RAGCOMPLIANCE_RETENTION_DAYS`).
Appends a retention checkpoint. Refuses on sinks that cannot delete.

## show

```bash
ragcompliance show [--sink URL] [--workspace W] [--json]
```

Per chain record count, sequence range and head hash. Record the head hash
somewhere the writer cannot modify.

## selftest

```bash
ragcompliance selftest [--dev-mode] [--json]
```

Checks the install: import, optional extras, sink configuration, config
load, and a round trip that seals two records to an in-memory sink,
verifies the chain and confirms tampering is detected. Exit non-zero on
blocking failures.
