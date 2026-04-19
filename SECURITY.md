# Security Policy

Thanks for helping keep RAGCompliance and its users safe. This document explains which versions get fixes, how to report a vulnerability, and what to expect in return.

## Supported versions

Security fixes land on the latest minor release line. Older releases are not backported.

| Version | Supported |
|---------|-----------|
| 0.1.x (latest) | yes |
| < 0.1.4 | no (known silent audit bug on langchain-core >= 1.3.0, see CHANGELOG 0.1.5) |

If you are running a version older than the latest on PyPI, please upgrade before filing a report so we can reproduce on a supported build.

## Reporting a vulnerability

Please do **not** open a public GitHub issue for security bugs. Email the maintainer directly at **daksh.trehan@hotmail.com** with the subject `ragcompliance security`. PGP is available on request.

In your report, include:

1. A description of the issue and the class of vulnerability (RCE, auth bypass, audit-integrity break, secret leak, etc.).
2. Affected version(s) (output of `pip show ragcompliance`).
3. Reproduction steps or a minimal proof-of-concept. A failing test case is ideal.
4. Impact assessment: what an attacker can do and under what preconditions.
5. Whether the bug is actively being exploited (so we can prioritise).
6. How you would like to be credited in the fix notes (name, handle, or anonymous).

## What to expect

| Step | Target |
|------|--------|
| Initial acknowledgement | within 72 hours |
| Triage + severity assessment | within 7 days |
| Fix landed on `main` + patch release | within 30 days for high / critical severity issues |
| Public disclosure (CHANGELOG + GitHub Security Advisory) | coordinated with reporter, typically on the patch release |

If a fix is going to take longer than 30 days for a high-severity issue, I will tell you why and propose a mitigation for users in the meantime.

## Scope

**In scope:**

- The `ragcompliance` Python package on PyPI.
- The FastAPI dashboard shipped in `ragcompliance.app`.
- The LangChain and LlamaIndex callback handlers.
- The SOC 2 evidence report generator (`ragcompliance.soc2`).
- The Stripe billing reference implementation (`ragcompliance.billing`).

**Out of scope** (these should be reported upstream):

- Vulnerabilities in `langchain-core`, `llama-index-core`, `supabase`, `stripe`, or `authlib`.
- Vulnerabilities in the underlying Python runtime or OS.
- Third-party services (Supabase, Stripe, your IdP) themselves.
- Social engineering, physical attacks, or attacks requiring already-compromised credentials.

## Threat model at a glance

RAGCompliance is middleware that writes one signed audit row per RAG chain invocation to storage you control. The main integrity promise is:

- Given a stored record, an auditor can recompute `sha256(query + chunks + answer)` and detect post-hoc tampering on any of those three fields.

The main confidentiality promises are:

- Row-level security on Supabase isolates rows per `workspace_id`. Cross-workspace reads require a compromised service-role key.
- The dashboard ships wide open by default (for local dev). OIDC SSO is opt-in and documented in the README and docs site.
- The billing readiness probe (`/health/billing`) sanitises all secrets; only prefixes like `sk_live…` ever leak.

If you find a bug that breaks any of the promises above, please report it. That is the highest-priority class.

## Safe harbour

Good-faith security research on RAGCompliance, following this policy, will not be pursued legally by the maintainer. Please do not:

- Run automated scanners that generate meaningful load against the author's infrastructure.
- Access other users' data.
- Modify, degrade, or destroy data that is not yours.
- Publicly disclose a vulnerability before a patch release is available.

Thanks for reading this far. Responsible disclosure keeps real users safe.
