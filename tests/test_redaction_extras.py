"""Redaction additions in 1.0: hooks, callable replacement, pseudonymisation,
IBAN, record level redaction incl. metadata and prompt text."""

from __future__ import annotations

import logging
import re

import pytest

from ragcompliance import AuditRecord, Pattern, Redactor, RetrievedChunk, pseudonymize, redact
from ragcompliance.redaction import RedactionResult, _iban_ok


class TestHooks:
    def test_string_hook_counts_when_it_changes_text(self):
        def strip_names(text: str) -> str:
            return text.replace("Alice", "[NAME]")

        r = Redactor(patterns=["email"], hooks=[strip_names])
        res = r.redact("Alice wrote to a@b.co")
        assert res.text == "[NAME] wrote to [REDACTED:EMAIL]"
        assert res.findings == {"email": 1, "strip_names": 1}
        assert r.redact("nothing here").findings == {}

    def test_result_hook_merges_findings(self):
        def ner(text: str) -> RedactionResult:
            return RedactionResult(text.replace("Paris", "[LOC]"), {"location": 1})

        r = Redactor(patterns=[], hooks=[ner])
        res = r.redact("in Paris")
        assert res.text == "in [LOC]" and res.findings == {"location": 1}

    def test_hook_error_is_logged_and_counted(self, caplog):
        def broken(text: str) -> str:
            raise RuntimeError("model offline")

        r = Redactor(patterns=["email"], hooks=[broken])
        with caplog.at_level(logging.WARNING, logger="ragcompliance.redaction"):
            res = r.redact("a@b.co")
        assert res.text == "[REDACTED:EMAIL]" and res.findings == {"email": 1, "hook_errors": 1}
        assert "model offline" in caplog.text
        assert r.describe()["hooks"] == ["broken"]


class TestReplacement:
    def test_callable_replacement_and_pseudonymize(self):
        fn = pseudonymize("salt")
        r = Redactor(patterns=["email"], replacement=fn)
        a = r.redact("a@b.co and a@b.co and c@d.co").text
        tokens = re.findall(r"\[PSEUDO:EMAIL:[0-9a-f]{12}\]", a)
        assert len(tokens) == 3 and tokens[0] == tokens[1] != tokens[2]
        assert r.describe()["replacement"] == "callable"
        other = Redactor(patterns=["email"], replacement=pseudonymize("other-salt")).redact("a@b.co").text
        assert other != tokens[0]
        with pytest.raises(ValueError):
            pseudonymize("")

    def test_validator_rejects_keep_text(self):
        r = Redactor(patterns=["credit_card"], replacement=lambda name, m: "X")
        assert r.redact("1234567890123456").text == "1234567890123456"
        assert r.redact("4111 1111 1111 1111").text == "X"


class TestIBAN:
    def test_iban_valid_and_invalid(self):
        assert _iban_ok("GB82 WEST 1234 5698 7654 32")
        assert not _iban_ok("GB82 WEST 1234 5698 7654 33")
        assert not _iban_ok("GB82")
        out = redact("pay to GB82 WEST 1234 5698 7654 32 now", patterns=["iban"])
        assert "[REDACTED:IBAN]" in out
        assert "REDACTED" not in redact("code GB82 WEST 1234 5698 7654 33", patterns=["iban"])


class TestRecordRedaction:
    def test_redact_record_covers_all_fields(self):
        r = Redactor()
        rec = AuditRecord(
            query="ssn 123-45-6789",
            retrieved_chunks=[RetrievedChunk("card 4111-1111-1111-1111", "http://10.0.0.1/x", "c",
                                             metadata={"author": "a@b.co", "n": 3, "key": "AKIAIOSFODNN7EXAMPLE"})],
            llm_answer="mail a@b.co",
            lineage={"prompt_text": "Bearer abcdefghijklmnopqrstuvwxyz0123456789"},
        )
        findings = r.redact_record(rec)
        assert rec.query == "ssn [REDACTED:SSN]"
        assert rec.retrieved_chunks[0].content == "card [REDACTED:CREDIT_CARD]"
        assert rec.retrieved_chunks[0].source_url == "http://[REDACTED:IPV4]/x"
        assert rec.retrieved_chunks[0].metadata == {"author": "[REDACTED:EMAIL]", "n": 3, "key": "[REDACTED:AWS_ACCESS_KEY]"}
        assert rec.llm_answer == "mail [REDACTED:EMAIL]"
        assert rec.lineage["prompt_text"] == "[REDACTED:BEARER_TOKEN]"
        assert findings == {"ssn": 1, "credit_card": 1, "ipv4": 1, "email": 2, "aws_access_key": 1, "bearer_token": 1}

    def test_legacy_record_fields_helper(self):
        r = Redactor(patterns=["email"])
        q, chunks, a, f = r.redact_record_fields(query="a@b.co", chunks=[{"content": "c@d.co", "source_url": "u", "chunk_id": "1"}], llm_answer="x")
        assert q == "[REDACTED:EMAIL]" and chunks[0]["content"] == "[REDACTED:EMAIL]" and chunks[0]["chunk_id"] == "1"
        assert a == "x" and f == {"email": 2}

    def test_pattern_names_and_describe(self):
        r = Redactor(patterns=["email"], custom_patterns=[Pattern("case", re.compile(r"CASE-\d+"))])
        assert r.pattern_names == ["email", "case"]
        d = r.describe()
        assert d["patterns"] == ["email", "case"] and d["replacement"] == "[REDACTED:{name}]" and d["hooks"] == []
