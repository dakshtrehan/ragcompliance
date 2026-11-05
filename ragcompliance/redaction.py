"""PII / PHI redaction for audit records.

Why this exists
---------------
A RAG audit log is, by construction, a copy of every query a user ever
asked plus every chunk the retriever surfaced plus every word the LLM
generated. In a regulated industry that is a large attack surface.

This module redacts known sensitive patterns BEFORE the record is signed
and persisted, so:

* the content signature is computed over the redacted payload and an
  auditor reproducing it never needs the raw value;
* storage never sees the raw value, so there is no "redact on read"
  that a direct database query bypasses;
* findings (counts by pattern name) land in ``extra["redaction_findings"]``
  so a dashboard or an evidence pack can show what was caught.

Three extension points
----------------------
``custom_patterns``
    Extra ``Pattern`` objects (compiled regex plus optional validator).
``replacement``
    Either a template string (``"[REDACTED:{name}]"``) or a callable
    ``(pattern_name, match) -> str``. ``pseudonymize(salt)`` returns a
    callable that replaces each value with a salted hash so analytics can
    still count distinct values without seeing them.
``hooks``
    Callables ``(text) -> str | RedactionResult`` run after the regex
    pass. Use them for an NER model, a dictionary of customer names, or
    anything a regex cannot express. A hook that raises is logged and
    skipped so a broken hook cannot stop the audit record from being
    written; the failure is counted under ``redaction_findings["hook_errors"]``.

Determinism
-----------
Default replacement is a fixed token, so same input gives same output.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from re import Pattern as _RePattern
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .models import AuditRecord

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Built-in patterns
# --------------------------------------------------------------------------

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}\b")

# US SSN: 9 digits with optional dashes/spaces. Excludes 000, 666, 9xx
# area numbers, 00 group and 0000 serial, which the SSA never issues.
_SSN = re.compile(r"\b(?!000|666|9\d\d)\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}\b")

# Credit card: 13-19 digits, optionally dash/space separated, Luhn checked.
_CREDIT_CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")

# US phone: optional +1, optional parens, at least one separator.
_PHONE_US = re.compile(r"(?:\+?1[ .\-])?\(?\b[2-9]\d{2}\)?[ .\-]\d{3}[ .\-]\d{4}\b")

# IPv4 with octet bounds.
_IPV4 = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)

_AWS_ACCESS_KEY = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")
_OPENAI_KEY = re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b")
_ANTHROPIC_KEY = re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-.=]{20,}\b")
# IBAN: country code, two check digits, 11 to 30 alphanumerics.
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?\b")


def _luhn_ok(digits_only: str) -> bool:
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


def _iban_ok(candidate: str) -> bool:
    """Mod-97 check as defined by ISO 13616."""
    s = re.sub(r"\s", "", candidate).upper()
    if not (15 <= len(s) <= 34):
        return False
    rearranged = s[4:] + s[:4]
    digits = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(digits) % 97 == 1


# --------------------------------------------------------------------------
# Pattern registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Pattern:
    """A named redaction pattern with an optional post-match validator."""

    name: str
    regex: _RePattern[str]
    validator: Callable[[str], bool] | None = None


BUILTIN_PATTERNS: dict[str, Pattern] = {
    "email": Pattern("email", _EMAIL),
    "ssn": Pattern("ssn", _SSN),
    "credit_card": Pattern(
        "credit_card", _CREDIT_CARD, validator=lambda s: _luhn_ok(re.sub(r"[ \-]", "", s))
    ),
    "phone_us": Pattern("phone_us", _PHONE_US),
    "ipv4": Pattern("ipv4", _IPV4),
    "aws_access_key": Pattern("aws_access_key", _AWS_ACCESS_KEY),
    "anthropic_key": Pattern("anthropic_key", _ANTHROPIC_KEY),
    "openai_key": Pattern("openai_key", _OPENAI_KEY),
    "bearer_token": Pattern("bearer_token", _BEARER),
    "iban": Pattern("iban", _IBAN, validator=_iban_ok),
}

# Anthropic must run before openai (longer prefix wins). ``iban`` is not in
# the default order because its regex is broad; opt in explicitly.
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
# Results and helpers
# --------------------------------------------------------------------------


@dataclass
class RedactionResult:
    """One redaction pass: the new text plus a per-pattern hit count."""

    text: str
    findings: dict[str, int] = field(default_factory=dict)


def _merge(dst: dict[str, int], src: dict[str, int]) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + v


ReplacementFn = Callable[[str, "re.Match[str]"], str]
Hook = Callable[[str], "str | RedactionResult"]


def pseudonymize(salt: str, *, length: int = 12, template: str = "[PSEUDO:{name}:{digest}]") -> ReplacementFn:
    """Replacement callable that maps each matched value to a salted
    SHA-256 digest, so the same value always maps to the same token
    without the token revealing the value. Keep the salt secret and per
    workspace."""
    if not salt:
        raise ValueError("pseudonymize needs a non-empty salt")

    def _replace(name: str, match: re.Match[str]) -> str:
        digest = hashlib.sha256((salt + match.group(0)).encode("utf-8")).hexdigest()[:length]
        return template.format(name=name.upper(), digest=digest)

    return _replace


# --------------------------------------------------------------------------
# Redactor
# --------------------------------------------------------------------------


class Redactor:
    """Apply a fixed set of patterns and hooks to text or a whole record."""

    def __init__(
        self,
        patterns: Iterable[str] | None = None,
        replacement: str | ReplacementFn = "[REDACTED:{name}]",
        custom_patterns: Iterable[Pattern] | None = None,
        hooks: Iterable[Hook] | None = None,
    ):
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
        self._hooks: list[Hook] = list(hooks or [])

    # ----------------------------------------------------------------------
    # Introspection
    # ----------------------------------------------------------------------

    @property
    def pattern_names(self) -> list[str]:
        return [p.name for p in self._patterns]

    def describe(self) -> dict[str, Any]:
        """Non-secret description for evidence packs."""
        return {
            "patterns": self.pattern_names,
            "hooks": [getattr(h, "__name__", h.__class__.__name__) for h in self._hooks],
            "replacement": (
                self._replacement if isinstance(self._replacement, str) else "callable"
            ),
        }

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
            out = pat.regex.sub(self._make_sub(pat, findings), out)

        for hook in self._hooks:
            try:
                res = hook(out)
            except Exception as exc:
                logger.warning("RAGCompliance: redaction hook %r raised: %s", hook, exc)
                findings["hook_errors"] = findings.get("hook_errors", 0) + 1
                continue
            if isinstance(res, RedactionResult):
                out = res.text
                _merge(findings, res.findings)
            elif isinstance(res, str):
                if res != out:
                    name = getattr(hook, "__name__", "hook")
                    findings[name] = findings.get(name, 0) + 1
                out = res
        return RedactionResult(text=out, findings=findings)

    def redact_record(self, record: AuditRecord) -> dict[str, int]:
        """Redact every free-text field of a record in place and return the
        merged findings. Covers query, chunk content, chunk source_url,
        string values in chunk metadata, the answer, and
        ``lineage["prompt_text"]`` when present."""
        merged: dict[str, int] = {}

        q = self.redact(record.query)
        record.query = q.text
        _merge(merged, q.findings)

        for chunk in record.retrieved_chunks:
            c = self.redact(chunk.content)
            chunk.content = c.text
            _merge(merged, c.findings)
            u = self.redact(chunk.source_url)
            chunk.source_url = u.text
            _merge(merged, u.findings)
            for key, value in list(chunk.metadata.items()):
                if isinstance(value, str):
                    m = self.redact(value)
                    chunk.metadata[key] = m.text
                    _merge(merged, m.findings)

        a = self.redact(record.llm_answer)
        record.llm_answer = a.text
        _merge(merged, a.findings)

        prompt = record.lineage.get("prompt_text") if isinstance(record.lineage, dict) else None
        if isinstance(prompt, str) and prompt:
            p = self.redact(prompt)
            record.lineage["prompt_text"] = p.text
            _merge(merged, p.findings)

        return merged

    def redact_record_fields(
        self,
        *,
        query: str,
        chunks: list[dict],
        llm_answer: str,
    ) -> tuple[str, list[dict], str, dict[str, int]]:
        """0.1.8 compatible helper over plain dicts."""
        merged: dict[str, int] = {}
        q_res = self.redact(query)
        _merge(merged, q_res.findings)
        new_chunks: list[dict] = []
        for c in chunks:
            content_res = self.redact(c.get("content", ""))
            url_res = self.redact(c.get("source_url", ""))
            _merge(merged, content_res.findings)
            _merge(merged, url_res.findings)
            new_chunks.append({**c, "content": content_res.text, "source_url": url_res.text})
        a_res = self.redact(llm_answer)
        _merge(merged, a_res.findings)
        return q_res.text, new_chunks, a_res.text, merged

    # ----------------------------------------------------------------------
    # Internal
    # ----------------------------------------------------------------------

    def _make_sub(self, pat: Pattern, findings: dict[str, int]) -> Callable[[re.Match[str]], str]:
        replacement = self._replacement
        static = replacement.format(name=pat.name.upper()) if isinstance(replacement, str) else None

        def _sub(m: re.Match[str]) -> str:
            if pat.validator is not None and not pat.validator(m.group(0)):
                return m.group(0)
            findings[pat.name] = findings.get(pat.name, 0) + 1
            if static is not None:
                return static
            assert callable(replacement)
            return replacement(pat.name, m)

        return _sub


# --------------------------------------------------------------------------
# Convenience
# --------------------------------------------------------------------------


def redact(text: str, patterns: Iterable[str] | None = None) -> str:
    """One-shot redaction with default settings."""
    return Redactor(patterns=patterns).redact(text).text
