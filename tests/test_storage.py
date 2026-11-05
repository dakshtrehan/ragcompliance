"""Tests for AuditStorage."""

from unittest.mock import MagicMock

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
        config = RAGComplianceConfig(enabled=False, dev_mode=True)
        storage = AuditStorage(config)
        result = storage.save(sample_record)
        assert result is True  # Skips silently
        assert storage.count() == 0


class TestSupabaseMode:
    def _storage(self, mock_client, **cfg):
        from ragcompliance.sinks.supabase import SupabaseSink

        config = RAGComplianceConfig(
            supabase_url="https://fake.supabase.co",
            supabase_key="fake-key",
            workspace_id="ws-prod",
            async_writes=False,
            **cfg,
        )
        sink = SupabaseSink(client=mock_client, table=config.table_name)
        return AuditStorage(config, sink=sink)

    def test_save_calls_supabase_insert(self, sample_record):
        mock_client = MagicMock()
        q = mock_client.table.return_value.select.return_value
        q.eq.return_value = q
        q.order.return_value = q
        q.limit.return_value.execute.return_value = MagicMock(data=[])
        mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock()

        storage = self._storage(mock_client)
        result = storage.save(sample_record)

        assert result is True
        mock_client.table.assert_called_with(storage.config.table_name)
        row = mock_client.table.return_value.insert.call_args[0][0]
        # Layout 1: legacy columns only, chain fields nested under extra.
        assert "record_hash" not in row
        assert row["extra"]["_ragcompliance"]["sequence"] == 1
        assert sample_record.sequence == 1
        assert len(sample_record.record_hash) == 64

    def test_save_returns_false_on_supabase_error(self, sample_record):
        mock_client = MagicMock()
        q = mock_client.table.return_value.select.return_value
        q.eq.return_value = q
        q.order.return_value = q
        q.limit.return_value.execute.return_value = MagicMock(data=[])
        mock_client.table.return_value.insert.return_value.execute.side_effect = Exception("DB error")

        storage = self._storage(mock_client)
        result = storage.save(sample_record)
        assert result is False
        assert storage.dropped_count == 1


class TestGetById:
    """Single-record lookup pushes the filter to the backend via .eq('id', ...)."""

    def _storage_with_mock(self, mock_client, workspace_id="ws-prod"):
        from ragcompliance.sinks.supabase import SupabaseSink

        config = RAGComplianceConfig(
            supabase_url="https://fake.supabase.co",
            supabase_key="fake-key",
            workspace_id=workspace_id,
            async_writes=False,
        )
        return AuditStorage(config, sink=SupabaseSink(client=mock_client))

    def test_get_by_id_returns_record_when_found(self):
        mock_client = MagicMock()
        q = mock_client.table.return_value.select.return_value
        q.eq.return_value = q  # chainable
        q.limit.return_value.execute.return_value = MagicMock(
            data=[{"id": "rec-1", "query": "hello"}]
        )
        storage = self._storage_with_mock(mock_client)

        result = storage.get_by_id("rec-1", workspace_id="ws-prod")

        assert result is not None
        assert result["id"] == "rec-1" and result["query"] == "hello"
        eq_keys = {call.args[0] for call in q.eq.call_args_list}
        assert "id" in eq_keys
        assert "workspace_id" in eq_keys

    def test_get_by_id_returns_none_when_not_found(self):
        mock_client = MagicMock()
        q = mock_client.table.return_value.select.return_value
        q.eq.return_value = q
        q.limit.return_value.execute.return_value = MagicMock(data=[])
        storage = self._storage_with_mock(mock_client)
        assert storage.get_by_id("missing-id") is None

    def test_get_by_id_returns_none_on_supabase_error(self):
        mock_client = MagicMock()
        q = mock_client.table.return_value.select.return_value
        q.eq.return_value = q
        q.limit.return_value.execute.side_effect = Exception("boom")
        storage = self._storage_with_mock(mock_client)
        assert storage.get_by_id("rec-1") is None

    def test_get_by_id_returns_none_without_sink(self):
        config = RAGComplianceConfig()  # no sink, not dev mode
        storage = AuditStorage(config)
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
