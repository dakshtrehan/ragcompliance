# Contributing to RAGCompliance

Thank you for considering a contribution. RAGCompliance is MIT-licensed middleware for putting RAG chains on an audit trail, and it lives or dies on whether it works quietly and correctly inside other people's systems. Every contribution is welcome, from a one-character typo fix to a new integration.

## How to get set up

```bash
git clone https://github.com/dakshtrehan/ragcompliance
cd ragcompliance
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,supabase,dashboard,llamaindex]"
pytest -v
```

Tests live in `tests/` and expect Python 3.11+. The full suite should pass offline — anything that genuinely needs a live Supabase, Stripe, or OpenAI should be gated behind an env flag and skipped by default.

## Before you open a PR

1. `ruff check ragcompliance tests` — lint must be clean. Run `ruff check --fix` to auto-apply what it can.
2. `pytest -q` — the full suite must pass.
3. If you changed public API, env vars, or the audit record shape, update the README and add a line under `[Unreleased]` in `CHANGELOG.md`.
4. Keep commits small and focused. One logical change per commit, conventional-commit-style prefixes (`feat:`, `fix:`, `docs:`, `chore:`, `perf:`) are preferred.

## What makes a good bug report

The bug template asks for a minimal reproduction, your `ragcompliance` version, and the full traceback. "I get an error" is hard to act on; a ten-line script that reliably reproduces the error is close to a fix. If you can share the audit record that looks wrong (with any sensitive content scrubbed), even better.

## What makes a good feature request

Start from the problem, not the solution. "My compliance team cannot sign off on RAG because we can't prove what the retriever saw" is a great opening. "Add an option to log retriever scores" is a great closing. Both together in one issue usually produces a quick merge.

## Scope of the project

RAGCompliance is deliberately small. It is a callback handler plus an audit store plus a dashboard. It is not:

- A vector database
- A reranker
- A full-fledged observability platform
- A retrieval tuner

Integrations with any of those are welcome as long as they preserve the "drop in, no chain rewrites" promise.

## Paid work and sponsorship

The repo stays MIT-licensed. If you need something built faster than the roadmap allows, or you want an operated dashboard rather than self-hosting, see the "Self-host and optional paid support" section in the README. Paid engagements do not fork the project, gate features, or change the license.

## Code of conduct

Be kind, be direct, and don't be the person who makes an open source maintainer regret opening issues. Disrespectful comments, personal attacks, and spam are removed; repeat offenders are blocked. If something in the project or community feels wrong, please email [daksh.trehan@hotmail.com](mailto:daksh.trehan@hotmail.com) directly.

## License

By contributing, you agree that your contribution will be licensed under the MIT License (the same license as the project).
