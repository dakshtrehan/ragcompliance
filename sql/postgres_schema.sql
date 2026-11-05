-- RAGCompliance 1.0: PostgreSQL sink schema (generated from PostgresSink.schema_sql()).
-- The sink creates this automatically unless ?create_schema=0 is in the URL.
-- Default table name: rag_audit_records (override with ?table=...).

CREATE TABLE IF NOT EXISTS rag_audit_records (
                record_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                chain_id TEXT NOT NULL,
                sequence BIGINT NOT NULL,
                ts_utc TIMESTAMPTZ NOT NULL,
                kind TEXT NOT NULL,
                session_id TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                record_hash TEXT NOT NULL,
                payload JSONB NOT NULL,
                UNIQUE (workspace_id, chain_id, sequence)
            );

CREATE INDEX IF NOT EXISTS rag_audit_records_ws_ts ON rag_audit_records (workspace_id, ts_utc);

CREATE INDEX IF NOT EXISTS rag_audit_records_session ON rag_audit_records (session_id);

CREATE OR REPLACE FUNCTION rag_audit_records_append_only() RETURNS trigger AS $$
            BEGIN
                IF current_setting('ragcompliance.allow_purge', true) = 'on' THEN
                    RETURN OLD;
                END IF;
                RAISE EXCEPTION 'ragcompliance audit log is append-only';
            END;
            $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS rag_audit_records_append_only ON rag_audit_records;

CREATE TRIGGER rag_audit_records_append_only BEFORE UPDATE OR DELETE ON rag_audit_records FOR EACH ROW EXECUTE FUNCTION rag_audit_records_append_only();
