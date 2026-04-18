---
name: Bug report
about: Report something that's broken or behaving unexpectedly
title: "[bug] "
labels: ["bug"]
---

## What happened

<!-- A clear and concise description of what went wrong. -->

## What you expected

<!-- What you thought would happen. -->

## Reproduction

Minimal code / commands to reproduce. Prefer a tiny repro over a full app:

```python
# e.g.
from ragcompliance import RAGComplianceHandler, RAGComplianceConfig

handler = RAGComplianceHandler(config=RAGComplianceConfig.from_env())
# ...
```

## Environment

- `ragcompliance` version: <!-- `pip show ragcompliance | grep Version` -->
- Python version: <!-- `python --version` -->
- OS: <!-- macOS 14.5, Ubuntu 22.04, etc. -->
- LangChain / LlamaIndex version (if relevant):
- Supabase project region (if relevant):

## Logs / traceback

<details>
<summary>Full traceback</summary>

```
paste here
```

</details>

## Anything else

<!-- Links to related issues, workarounds you've tried, hypotheses. -->
