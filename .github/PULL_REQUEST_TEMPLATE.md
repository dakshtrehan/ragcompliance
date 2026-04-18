# Pull Request

## What does this change?

<!-- 1-3 sentences on the behavior change. Link to the issue this fixes. -->

Fixes # <!-- or: Related to # -->

## Why

<!-- The motivation. "Because the handler dropped records under concurrent LCEL chains" is better than "improve handler reliability". -->

## Changes

<!-- A short bulleted list, oriented around observable behavior. -->

- ...
- ...

## How has this been tested?

- [ ] Added / updated unit tests
- [ ] `pytest` passes locally
- [ ] `ruff check ragcompliance tests` is clean
- [ ] Manually verified against a real Supabase / Stripe / IdP setup (describe below if yes)

<!-- If manual verification: what did you do, what did you see? -->

## Breaking changes / migration notes

<!-- If this changes public API, env vars, the DB schema, or the audit record shape, call it out here with a migration path. If not, write "None." -->

None.

## Checklist

- [ ] Public API changes are documented in the README and `CHANGELOG.md` under `[Unreleased]`
- [ ] New env vars are added to the env-var reference table in the docs
- [ ] I did not add a paid tier — this project is MIT and stays MIT
