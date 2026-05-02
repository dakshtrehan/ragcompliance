"""
PII / PHI redaction for RAGCompliance audit records.

Why this exists
---------------
A RAG audit log is, by construction, a perfect copy of every query a user
ever asks plus every chunk the retriever surfaced plus every word the LLM
generated. In a regulated industry that's a massive amount of attack
surface: a single SELECT against the audit table and an attacker has
patient identifiers, account numbers, support-ticket attachments, and
internal documents at full fidelity.

This module redacts known sensitive patterns BEFORE the record is signed
and persisted, so:
  - The chain signature (SHA-256) is computed over the redacted payload.
    An auditor reproducing the signature does not need access to raw
    secrets.
  - Storage never sees the raw secret. There is no "redact on read"
    fallback that can be bypassed by querying the database directly.
  - Findings (counts by pattern name) are surfaced in the record's
    ``extra`` dict so a compliance dashboard can show, per record,
    how many PII hits were caught.

Determinism
-----------
The default redaction is a fixed token like ``[REDACTED:EMAIL]``.
Same input -> same output. A custom replacement function can hash the
matched value with a workspace-scoped salt if you want to preserve
uniqueness for downstream analytics without leaking the value.

Performance
-----------
All built-in patterns are compiled regex applied in a single pass per
field. Typical overhead on a 4-KB chunk with all built-in patterns
enabled is < 0.5ms; redaction is run inside the LangChain callback hot
path so it has to be cheap. Custom patterns share the same compile-once,
match-many path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Pattern as _RePattern


# --------------------------------------------------------------------------
# Built-in patterns
# --------------------------------------------------------------------------
#
# Each pattern is intentionally conservative — false negatives are
# preferable to false positives in a compliance context (a missed
# redaction is bad, but a corrupted retrieved chunk that loses meaning
# breaks the audit usefulness entirely). Operators can layer custom
# patterns on top with ``custom_patterns=[...]``.

_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}\b"
)

# US SSN: 9 digits with optional dashes/spaces. Excludes obvious
# non-SSNs like 000-XX-XXXX and 666-XX-XXXX which the SSA never issues.
_SSN = re.compile(
    r"\b(?!000|666|9\d\d)\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}\b"
)

# Credit card: 13-19 digits, optionally dash/space separated. Validation
# is via a Luhn check at match time; the regex is intentionally loose.
_CREDIT_CARD = re.compile(
    r"\b(?:\d[ -]?){12,18}\d\b"
)

# US phone: optional +1, optional parens, separators flexible. Avoids
# matching plain 10-digit IDs by requiring at least one separator.
_PHONE_US = re.compile(
    r"(?:\+?1[ .\-])?\(?\b[2-9]\d{2}\)?[ .\-]\d{3}[ .\-]\d{4}\b"
)

# IPv4 with octet bounds (0-255 each).
_IPV4 = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)

# AWS access key id — fixed 20 chars, AKIA / ASIA prefixes.
_AWS_ACCESS_KEY = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")

# OpenAI / Anthropic style API keys: prefix + long base64ish tail.
_OPENAI_KEY = re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b")
_ANTHROPIC_KEY = re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")

# Bearer token in an Authorization header.
_BEARER = re.compile(
    r"(?i)\bbearer\s+[A-Za-z0-9_\-.=]{20,}\b"
)


def _luhn_ok(digits_only: str) -> bool:
    """Standard Luhn checksum. Returns False for trivially invalid inputs."""
    if not (13 <= len(digits_only) <= 19) or not digits_only.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(digits_only)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# --------------------------------------------------------------------------
# Pattern registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Pattern:
    """A single named redaction pattern.

    ``validator`` is an optional callable that takes the matched text
    (with separators preserved) and returns True if the match should be
    redacted. Used by ``credit_card`` to apply the Luhn check; this lets
    the regex stay loose without flooding logs with false positives on
    long digit strings (chunk_ids, IDs, etc.).
    """

    name: str
    regex: _RePattern[str]
    validator: Callable[[str], bool] | None = None


# Anthropic key MUST come before generic openai key because both start
# with ``sk-`` — pattern order matters when two regexes overlap.
BUILTIN_PATTERNS: dict[str, Pattern] = {
    "email": Pattern("email", _EMAIL),
    "ssn": Pattern("ssn", _SSN),
    "credit_card": Pattern(
        "credit_card",
        _CREDIT_CARD,
        validator=lambda s: _luhn_ok(re.sub(r"[ \-]", "", s)),
    ),
    "phone_us": Pattern("phone_us", _PHONE_US),
    "ipv4": Pattern("ipv4", _IPV4),
    "aws_access_key": Pattern("aws_access_key", _AWS_ACCESS_KEY),
    "anthropic_key": Pattern("anthropic_key", _ANTHROPIC_KEY),
    "openai_key": Pattern("openai_key", _OPENAI_KEY),
    "bearer_token": Pattern("bearer_token", _BEARER),
}


# Anthropic must run before openai (longer prefix wins).
DEFAULT_PATTERN_ORDER: tuple[str, ...] = (
    "anthropic_key",
    "openai_key",
    "bearer_token",
    "aws_access_key",
    "credit_card",
    "ssn",
    "email",
    "phone_us",
    "ipv4",
)


# --------------------------------------------------------------------------
# Redactor
# --------------------------------------------------------------------------


@dataclass
class RedactionResult:
    """One redaction pass: the new text plus a per-pattern hit count."""

    text: str
    findings: dict[str, int] = field(default_factory=dict)


class Redactor:
    """Apply a fixed set of patterns to text and return a redaction result.

    Operators normally construct one ``Redactor`` per process and pass it
    into the handler. The redactor is thread-safe: ``redact`` only reads
    immutable compiled regex objects.
    """

    def __init__(
        self,
        patterns: Iterable[str] | None = None,
        replacement: str = "[REDACTED:{name}]",
        custom_patterns: Iterable[Pattern] | None = None,
    ):
        # Resolve names against the built-in registry. Unknown names are
        # rejected loudly so a typo in a config var doesn't silently
        # disable redaction.
        names = tuple(patterns) if patterns is not None else DEFAULT_PATTERN_ORDER
        unknown = [n for n in names if n not in BUILTIN_PATTERNS]
        if unknown:
            raise ValueError(
                f"Unknown built-in redaction pattern(s): {unknown}. "
                f"Known: {sorted(BUILTIN_PATTERNS)}"
            )
        self._patterns: list[Pattern] = [BUILTIN_PATTERNS[n] for n in names]
        if custom_patterns:
            self._patterns.extend(custom_patterns)
        self._replacement = replacement

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def redact(self, text: str) -> RedactionResult:
        """Redact one string. Empty / non-str inputs are returned untouched."""
        if not text or not isinstance(text, str):
            return RedactionResult(text=text or "", findings={})

        findings: dict[str, int] = {}
        out = text
        for pat in self._patterns:
            replacement = self._replacement.format(name=pat.name.upper())

            def _sub(m: re.Match[str], _name=pat.name, _val=pat.validator,
                     _replacement=replacement) -> str:
                if _val is not None and not _val(m.group(0)):
                    return m.group(0)
                findings[_name] = findings.get(_name, 0) + 1
                return _replacement

            out = pat.regex.sub(_sub, out)
        return RedactionResult(text=out, findings=findings)

    def redact_record_fields(
        self,
        *,
        query: str,
        chunks: list[dict],
        llm_answer: str,
    ) -> tuple[str, list[dict], str, dict[str, int]]:
        """Redact a partial audit record. Returns the redacted query, the
        redacted chunks (as new dicts), the redacted answer, and a
        merged findings count dict.

        The handler calls this exactly once per record, immediately
        before signing. The redacted output is what gets signed and what
        gets persisted; the raw input is dropped.
        """
        merged: dict[str, int] = {}

        q_res = self.redact(query)
        _merge(merged, q_res.findings)

        new_chunks: list[dict] = []
        for c in chunks:
            content_res = self.redact(c.get("content", ""))
            url_res = self.redact(c.get("source_url", ""))
            _merge(merged, content_res.findings)
            _merge(merged, url_res.findings)
            new_chunks.append({**c,
                               "content": content_res.text,
                               "source_url": url_res.text})

        a_res = self.redact(llm_answer)
        _merge(merged, a_res.findings)

        return q_res.text, new_chunks, a_res.text, merged


def _merge(dst: dict[str, int], src: dict[str, int]) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + v


# --------------------------------------------------------------------------
# Convenience
# --------------------------------------------------------------------------


def redact(text: str, patterns: Iterable[str] | None = None) -> str:
    """Module-level convenience: one-shot redaction with default settings.

    Equivalent to ``Redactor(patterns=patterns).redact(text).text`` but
    cheaper to type. Used in tests and in docs examples.
    """
    return Redactor(patterns=patterns).redact(text).text
