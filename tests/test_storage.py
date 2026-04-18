"""Tests for AuditStorage."""

from unittest.mock import MagicMock, patch

import pytest

from ragcompliance import AuditRecord, AuditStorage, RAGComplianceConfig


@pytest.fixture
def dev_storage():
    config = RAGComplianceConfig(dev_mode=True)
    return AuditStorage(config)


@pytest.fixture
def sample_record():
    return AuditRecord(
        session_id="sess-001",
        workspace_id="ws-test",
        query="What is explainability in AI?",
        llm_answer="Explainability refers to...",
        chain_signature="abc123",
    )


class TestDevMode:
    def test_save_returns_true_in_dev_mode(self, dev_storage, sample_record, capsys):
        result = dev_storage.save(sample_record)
        assert result is True

    def test_dev_mode_prints_record(self, dev_storage, sample_record, capsys):
        dev_storage.save(sample_record)
        captured = capsys.readouterr()
        assert "[RAGCompliance DEV]" in captured.out
        assert "What is explainability in AI?" in captured.out


class TestDisabledMode:
    def test_save_skips_when_disabled(self, sample_record):
        config = RAGComplianceConfig(enabled=False)
        storage = AuditStorage(config)
        result = storage.save(sample_record)
        assert result is True  # Skips silently


class TestSupabaseMode:
    def test_save_calls_supabase_insert(self, sample_record):
        config = RAGComplianceConfig(
            supabase_url="https://fake.supabase.co",
            supabase_key="fake-key",
            workspace_id="ws-prod",
        )
        mock_client = MagicMock()
        mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock()

        with patch("ragcompliance.storage.AuditStorage.__init__", lambda self, cfg: None):
            storage = AuditStorage.__new__(AuditStorage)
            storage.config = config
            storage._client = mock_client

        result = storage.save(sample_record)
        assert result is True
        mock_client.table.assert_called_once_with(config.table_name)

    def test_save_returns_false_on_supabase_error(self, sample_record):
        config = RAGComplianceConfig(supabase_url="https://fake.supabase.co", supabase_key="key")
        mock_client = MagicMock()
        mock_client.table.return_value.insert.return_value.execute.side_effect = Exception("DB error")

        storage = AuditStorage.__new__(AuditStorage)
        storage.config = config
        storage._client = mock_client

        result = storage.save(sample_record)
        assert result is False


class TestGetById:
    """Single-record lookup pushes the filter to Supabase via .eq('id', ...)
    rather than scanning an in-memory page. These tests verify the query
    shape and the not-found / workspace-scoped behavior."""

    def _storage_with_mock(self, mock_client, workspace_id="ws-prod"):
        config = RAGComplianceConfig(
            supabase_url="https://fake.supabase.co",
            supabase_key="fake-key",
            workspace_id=workspace_id,
        )
        storage = AuditStorage.__new__(AuditStorage)
        storage.config = config
        storage._client = mock_client
        storage._queue = None
        storage._stop_event = None
        storage._worker = None
        return storage

    def test_get_by_id_returns_record_when_found(self):
        mock_client = MagicMock()
        q = mock_client.table.return_value.select.return_value
        q.eq.return_value = q  # chainable
        q.limit.return_value.execute.return_value = MagicMock(
            data=[{"id": "rec-1", "query": "hello"}]
        )
        storage = self._storage_with_mock(mock_client)

        result = storage.get_by_id("rec-1", workspace_id="ws-prod")

        assert result == {"id": "rec-1", "query": "hello"}
        # The lookup must filter by id (and workspace when provided), not
        # scan a large window and pick one in memory.
        eq_calls = q.eq.call_args_list
        eq_keys = {call.args[0] for call in eq_calls}
        assert "id" in eq_keys
        assert "workspace_id" in eq_keys

    def test_get_by_id_returns_none_when_not_found(self):
        mock_client = MagicMock()
        q = mock_client.table.return_value.select.return_value
        q.eq.return_value = q
        q.limit.return_value.execute.return_value = MagicMock(data=[])
        storage = self._storage_with_mock(mock_client)

        result = storage.get_by_id("missing-id")
        assert result is None

    def test_get_by_id_returns_none_on_supabase_error(self):
        mock_client = MagicMock()
        q = mock_client.table.return_value.select.return_value
        q.eq.return_value = q
        q.limit.return_value.execute.side_effect = Exception("boom")
        storage = self._storage_with_mock(mock_client)

        assert storage.get_by_id("rec-1") is None

    def test_get_by_id_returns_none_without_supabase_client(self):
        config = RAGComplianceConfig(dev_mode=True)
        storage = AuditStorage(config)
        # No supabase client configured -> returns None rather than a fake hit.
        assert storage.get_by_id("rec-1") is None


class TestAuditRecordSerialization:
    def test_to_dict_contains_all_fields(self, sample_record):
        d = sample_record.to_dict()
        assert "id" in d
        assert "session_id" in d
        assert "query" in d
        assert "retrieved_chunks" in d
        assert "chain_signature" in d
        assert "timestamp" in d
        assert d["query"] == "What is explainability in AI?"

    def test_timestamp_is_iso_string(self, sample_record):
        d = sample_record.to_dict()
        # Should be parseable ISO format
        from datetime import datetime
        dt = datetime.fromisoformat(d["timestamp"])
        assert dt is not None
