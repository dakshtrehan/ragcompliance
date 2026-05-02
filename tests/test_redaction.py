"""
Tests for ragcompliance.redaction.

These cover three layers:

1. Pattern correctness — each built-in pattern hits what it should and
   does not false-positive on common decoy strings.
2. Redactor behavior — order resolution, custom patterns, replacement
   templating, no-op on empty input.
3. End-to-end through the handler — when redact_pii is on, the
   AuditRecord written to dev-mode storage has redacted fields, the
   chain signature is computed over the redacted payload, and findings
   land in extra["redaction_findings"].

The handler-level tests use the dev-mode stdout storage path so they
do not depend on Supabase being installed.
"""

import json
from unittest.mock import patch
from uuid import uuid4

import pytest

from ragcompliance.config import RAGComplianceConfig
from ragcompliance.handler import RAGComplianceHandler
from ragcompliance.models import RetrievedChunk
from ragcompliance.redaction import (
    BUILTIN_PATTERNS,
    DEFAULT_PATTERN_ORDER,
    Pattern,
    Redactor,
    redact,
)


# ----------------------------------------------------------------------
# Pattern correctness
# ----------------------------------------------------------------------


class TestEmail:
    def test_basic(self):
        out = redact("contact alice@example.com please")
        assert "[REDACTED:EMAIL]" in out
        assert "alice@example.com" not in out

    def test_subdomain_and_plus_addressing(self):
        out = redact("ping bob+filter@mail.corp.example.co.uk")
        assert "[REDACTED:EMAIL]" in out

    def test_does_not_match_plain_at(self):
        out = redact("the @ symbol alone is fine")
        assert "REDACTED" not in out


class TestSSN:
    def test_dashed(self):
        out = redact("SSN is 123-45-6789 here")
        assert "[REDACTED:SSN]" in out
        assert "123-45-6789" not in out

    def test_undashed(self):
        out = redact("SSN: 123456789 in record")
        assert "[REDACTED:SSN]" in out

    def test_skips_invalid_prefix(self):
        # 000, 666, 9xx are never issued by SSA
        for s in ("000-12-3456", "666-12-3456", "900-12-3456"):
            assert "REDACTED" not in redact(f"id {s} end")


class TestCreditCard:
    def test_visa_with_luhn(self):
        # 4111 1111 1111 1111 is the canonical Visa test card
        out = redact("card 4111-1111-1111-1111 charged")
        assert "[REDACTED:CREDIT_CARD]" in out
        assert "4111" not in out

    def test_amex_with_luhn(self):
        out = redact("amex 378282246310005 ok")
        assert "[REDACTED:CREDIT_CARD]" in out

    def test_skips_non_luhn(self):
        # Same length as a CC but fails Luhn -> must NOT be redacted
        out = redact("not a card 1234567890123456")
        assert "REDACTED" not in out


class TestPhoneUS:
    def test_dashed(self):
        assert "[REDACTED:PHONE_US]" in redact("call 415-555-1234 now")

    def test_with_country_code_and_parens(self):
        assert "[REDACTED:PHONE_US]" in redact("dial +1 (415) 555-1234")

    def test_skips_long_id(self):
        # 10 digits with no separator should not match — phone regex
        # requires at least one separator
        out = redact("transaction 4155551234 logged")
        assert "REDACTED" not in out


class TestIPv4:
    def test_basic(self):
        out = redact("client 192.168.1.42 connected")
        assert "[REDACTED:IPV4]" in out
        assert "192.168.1.42" not in out

    def test_skips_invalid_octet(self):
        out = redact("not-an-ip 999.1.2.3 here")
        assert "REDACTED" not in out


class TestSecrets:
    def test_aws_access_key(self):
        out = redact("key=AKIAIOSFODNN7EXAMPLE in env")
        assert "[REDACTED:AWS_ACCESS_KEY]" in out

    def test_openai_key(self):
        out = redact("OPENAI_API_KEY=sk-abcdef0123456789ABCDEF set")
        assert "[REDACTED:OPENAI_KEY]" in out

    def test_anthropic_key_takes_priority(self):
        # ``sk-ant-...`` must redact as anthropic_key (longer prefix
        # wins), not double-redact or fall through to openai_key.
        out = redact("ANTHROPIC=sk-ant-abcdef0123456789ABCDEF here")
        assert "[REDACTED:ANTHROPIC_KEY]" in out
        assert "OPENAI" not in out

    def test_bearer_token(self):
        out = redact("Authorization: Bearer abc.def-123_456ABCDEF7890")
        assert "[REDACTED:BEARER_TOKEN]" in out


# ----------------------------------------------------------------------
# Redactor behavior
# ----------------------------------------------------------------------


class TestRedactorBehavior:
    def test_empty_and_none_passthrough(self):
        r = Redactor()
        assert r.redact("").text == ""
        # type-mismatched inputs should not crash
        assert r.redact(None).text == ""  # type: ignore[arg-type]

    def test_custom_replacement_template(self):
        r = Redactor(replacement="<<{name}>>")
        out = r.redact("a@b.com").text
        assert out == "<<EMAIL>>"

    def test_unknown_pattern_name_raises(self):
        with pytest.raises(ValueError, match="Unknown built-in"):
            Redactor(patterns=["email", "not_a_pattern"])

    def test_custom_pattern(self):
        import re
        case_id = Pattern("case_id", re.compile(r"\bCASE-\d{4}\b"))
        r = Redactor(patterns=["email"], custom_patterns=[case_id])
        out = r.redact("CASE-9999 was opened by a@b.com")
        assert "[REDACTED:CASE_ID]" in out.text
        assert "[REDACTED:EMAIL]" in out.text
        assert out.findings == {"case_id": 1, "email": 1}

    def test_findings_are_counted_per_pattern(self):
        r = Redactor()
        out = r.redact("a@b.com and c@d.com and SSN 123-45-6789")
        assert out.findings == {"email": 2, "ssn": 1}

    def test_default_pattern_set_completeness(self):
        # All names in DEFAULT_PATTERN_ORDER must resolve in
        # BUILTIN_PATTERNS — guards against typos in the default list.
        for name in DEFAULT_PATTERN_ORDER:
            assert name in BUILTIN_PATTERNS


# ----------------------------------------------------------------------
# End-to-end through the handler
# ----------------------------------------------------------------------


class TestHandlerIntegration:
    def _make_handler(self, *, redact_pii: bool):
        # dev_mode + no supabase creds -> storage logs to stdout, never
        # touches the network. async_writes off so the record lands
        # synchronously and we can read it back from the patched
        # storage.save call.
        cfg = RAGComplianceConfig(
            dev_mode=True,
            async_writes=False,
            redact_pii=redact_pii,
        )
        return RAGComplianceHandler(config=cfg, session_id="test")

    def _drive_chain(self, handler, *, query, chunks, answer):
        """Replay a minimal LCEL-shaped sequence on the handler."""
        run_id = uuid4()
        handler.on_chain_start({"name": "RunnableSequence"}, {"query": query},
                               run_id=run_id, parent_run_id=None)
        # Retriever fake
        ret_id = uuid4()
        handler.on_retriever_start({}, query,
                                   run_id=ret_id, parent_run_id=run_id)
        from langchain_core.documents import Document
        docs = [Document(page_content=c["content"],
                         metadata={"source": c["source_url"],
                                   "chunk_id": c["chunk_id"]}) for c in chunks]
        handler.on_retriever_end(docs, run_id=ret_id, parent_run_id=run_id)
        # LLM fake
        llm_id = uuid4()
        handler.on_llm_start({}, [query],
                             run_id=llm_id, parent_run_id=run_id)
        from langchain_core.outputs import LLMResult, Generation
        result = LLMResult(generations=[[Generation(text=answer)]],
                           llm_output={"model_name": "gpt-4o-mini"})
        handler.on_llm_end(result, run_id=llm_id, parent_run_id=run_id)
        # Chain end
        handler.on_chain_end({"output": answer}, run_id=run_id,
                             parent_run_id=None)

    def test_off_by_default_preserves_raw_payload(self):
        handler = self._make_handler(redact_pii=False)
        with patch.object(handler.storage, "save") as save:
            self._drive_chain(
                handler,
                query="patient alice@hospital.org has SSN 123-45-6789",
                chunks=[{"content": "card 4111-1111-1111-1111",
                         "source_url": "http://10.0.0.5/file",
                         "chunk_id": "c1"}],
                answer="ok contact alice@hospital.org",
            )
            assert save.called
            record = save.call_args[0][0]
            # Raw values still present — handler did not redact
            assert "alice@hospital.org" in record.query
            assert "REDACTED" not in record.query
            assert "redaction_findings" not in record.extra

    def test_on_redacts_query_chunks_answer(self):
        handler = self._make_handler(redact_pii=True)
        with patch.object(handler.storage, "save") as save:
            self._drive_chain(
                handler,
                query="patient alice@hospital.org has SSN 123-45-6789",
                chunks=[{"content": "card 4111-1111-1111-1111",
                         "source_url": "http://10.0.0.5/file",
                         "chunk_id": "c1"}],
                answer="ok contact alice@hospital.org",
            )
            record = save.call_args[0][0]
            # Query redacted
            assert "alice@hospital.org" not in record.query
            assert "[REDACTED:EMAIL]" in record.query
            assert "[REDACTED:SSN]" in record.query
            # Chunk content + URL redacted
            assert "4111" not in record.retrieved_chunks[0].content
            assert "[REDACTED:CREDIT_CARD]" in record.retrieved_chunks[0].content
            assert "[REDACTED:IPV4]" in record.retrieved_chunks[0].source_url
            # Answer redacted
            assert "alice@hospital.org" not in record.llm_answer
            # Findings recorded
            findings = record.extra["redaction_findings"]
            assert findings.get("email", 0) >= 2
            assert findings.get("ssn", 0) == 1
            assert findings.get("credit_card", 0) == 1
            assert findings.get("ipv4", 0) == 1

    def test_signature_is_over_redacted_payload(self):
        """An auditor reproducing the chain signature must be able to do
        so from the redacted record alone — no access to raw secrets."""
        import hashlib
        import json as _json

        handler = self._make_handler(redact_pii=True)
        with patch.object(handler.storage, "save") as save:
            self._drive_chain(
                handler,
                query="ssn 123-45-6789",
                chunks=[{"content": "doc", "source_url": "u",
                         "chunk_id": "c1"}],
                answer="answer",
            )
            record = save.call_args[0][0]

        # Recompute signature from the persisted (redacted) record.
        payload = {
            "query": record.query,
            "chunks": [
                {"content": c.content, "source_url": c.source_url,
                 "chunk_id": c.chunk_id}
                for c in record.retrieved_chunks
            ],
            "answer": record.llm_answer,
        }
        recomputed = hashlib.sha256(
            _json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        assert recomputed == record.chain_signature
