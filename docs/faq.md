# FAQ

**Is this a SOC 2 certification?**
No. A SOC 2 report is issued by a licensed CPA firm after examining your
whole system. ragcompliance produces the part of the evidence an audit
logging component can produce about itself, in a form auditors can verify
independently. See [Control mapping](control-mapping.md).

**How is this different from LangSmith or other tracing tools?**
Tracing tools are for debugging and observability; they store traces in
their own service and let you edit or delete them. ragcompliance writes a
hash chained record into storage you control, enforces append only
semantics where the store allows it, redacts before hashing, and exports an
auditor facing evidence pack. The two are complementary.

**Does the hash chain prove nothing was deleted?**
It proves no record was removed from inside the chain and nothing was
edited. It cannot by itself prove the newest records were not truncated;
record the head hash somewhere the writer cannot modify (evidence packs do
this per period). With an HMAC key, nobody without the key can rewrite the
chain consistently. See [Threat model](threat-model.md).

**Which sink should I use?**
PostgreSQL if you have it: multi writer safe, append only trigger, purge
support. SQLite for a single host. S3 with Object Lock when you need WORM
storage. JSONL when you ship logs to a pipeline. Supabase if you already
run it (apply the layout 2 migration).

**Does redaction catch names and addresses?**
The built in patterns are structured identifiers (emails, SSNs, cards,
phones, IPs, API keys, IBANs). For free text entities add a hook with an
NER model. See [Redaction](redaction.md).

**Can I store the prompt?**
Yes, `RAGCOMPLIANCE_CAPTURE_PROMPTS=true`. The hash is always stored; the
text is redacted when redaction is on.

**What does the Anthropic wrapper do with `temperature`?**
Anthropic SDK 2.x targets API version 2026-10-21, which rejects
`temperature`, and the SDK no longer accepts it. The wrapper drops it (and
`top_p`, `top_k` when unsupported), logs once, and records the requested
value in `lineage.model_params_dropped`. See
[OpenAI and Anthropic SDKs](integrations/sdk.md).

**Is the hot path slow?**
Capture is a few dictionary operations behind a lock; the default write path
hands the record to a background thread. Hashing happens in the sink's
writer. Redaction is a handful of compiled regexes per field.

**Can several processes write one chain?**
Yes with SQLite, PostgreSQL, JSONL and Supabase layout 2. For S3 and
Supabase layout 1 give each writer its own `RAGCOMPLIANCE_CHAIN_ID`. See
[Hash chain](hash-chain.md).

**Does it work with Haystack, DSPy, or my own framework?**
Use the standalone `Auditor`: open a trace, add chunks, set the answer. The
LangChain and LlamaIndex handlers are thin adapters over the same core.

**Where do I report a security issue?**
See `SECURITY.md`. Anything that breaks a row of the threat model tables is
the highest priority.
