# Redaction

An audit log of a RAG system is a copy of every question, every retrieved
passage and every answer. Redaction runs before hashing, so the stored
record verifies without the raw values and there is no "redact on read"
that a direct database query bypasses.

## Turn it on

```bash
RAGCOMPLIANCE_REDACT_PII=true
RAGCOMPLIANCE_REDACTION_PATTERNS=anthropic_key,openai_key,bearer_token,aws_access_key,credit_card,ssn,email,phone_us,ipv4
RAGCOMPLIANCE_REDACTION_REPLACEMENT=[REDACTED:{name}]
```

Redaction applies in every integration (LangChain, LlamaIndex, the SDK
wrappers and the standalone `Auditor`) to: the query, each chunk's content,
source URL and string metadata values, the answer, and the prompt text
when prompt capture is on. Per pattern hit counts land in
`extra.redaction_findings` so a dashboard or an evidence pack can show what
was removed without storing it.

## Built in patterns

| Name | Notes |
|------|-------|
| `email` | |
| `ssn` | US SSN, excludes area numbers the SSA never issues |
| `credit_card` | 13 to 19 digits, Luhn validated so ids do not false positive |
| `phone_us` | requires at least one separator |
| `ipv4` | octet bounded |
| `aws_access_key` | `AKIA` and `ASIA` prefixes |
| `anthropic_key` | runs before `openai_key` because both start with `sk-` |
| `openai_key` | |
| `bearer_token` | `Authorization: Bearer ...` values |
| `iban` | mod 97 validated; not in the default order, opt in explicitly |

## Custom patterns

```python
import re
from ragcompliance import Pattern, Redactor

case_id = Pattern("case_id", re.compile(r"\bCASE-\d{4,}\b"))
redactor = Redactor(custom_patterns=[case_id])
```

## Hooks

A hook is any callable `(text) -> str | RedactionResult`, run after the
regex pass. Use it for a named entity recogniser, a dictionary of customer
names, or anything a regex cannot express.

```python
from ragcompliance import Redactor
from ragcompliance.redaction import RedactionResult

def ner_redact(text: str) -> RedactionResult:
    spans = my_ner_model(text)              # your model
    for span in reversed(spans):
        text = text[: span.start] + f"[REDACTED:{span.label}]" + text[span.end :]
    return RedactionResult(text, {"ner": len(spans)})

redactor = Redactor(hooks=[ner_redact])
```

A hook that raises is logged and skipped, and the failure is counted as
`hook_errors` in the findings. A broken hook never stops the record from
being written.

## Pseudonymisation

Replace values with a salted hash so analytics can still count distinct
customers without seeing them:

```python
from ragcompliance import Redactor, pseudonymize

redactor = Redactor(replacement=pseudonymize(salt=os.environ["RC_PSEUDO_SALT"]))
# "alice@example.com" -> "[PSEUDO:EMAIL:3f1c9a2b7d0e]" (same value, same token)
```

Keep the salt out of the store and rotate it per workspace. The evidence
pack records that a callable replacement was used, never the salt.

## Wiring a custom redactor

```python
handler = RAGComplianceHandler(redactor=redactor)
auditor = Auditor(config, redactor=redactor)
client = AuditedOpenAI(OpenAI(), redactor=redactor)
```

Any object with a `redact(text)` method returning something with `.text`
and `.findings` works. Objects that also implement
`redact_record(record)` get full control over which fields are touched.

## Limits

Regex patterns miss free text names, addresses and anything context
dependent. Review the pattern set against your own data, add hooks for
what regexes cannot catch, and treat the audit store as sensitive even
with redaction on.
