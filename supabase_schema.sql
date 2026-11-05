-- RAGCompliance: Supabase schema (0.1.x column set).
-- Run once in the Supabase SQL editor before using the Supabase sink.
-- Then run supabase_migration_v1.sql for the 1.0 hash chain columns.
-- Safe to re-run.

create extension if not exists "pgcrypto";

create table if not exists rag_audit_logs (
    id                uuid primary key default gen_random_uuid(),
    session_id        text not null,
    workspace_id      text not null default 'default',
    query             text not null,
    retrieved_chunks  jsonb not null default '[]',
    llm_answer        text not null default '',
    model_name        text not null default '',
    chain_signature   text not null,
    timestamp         timestamptz not null default now(),
    latency_ms        integer not null default 0,
    extra             jsonb not null default '{}'
);

create index if not exists idx_audit_workspace  on rag_audit_logs (workspace_id);
create index if not exists idx_audit_session    on rag_audit_logs (session_id);
create index if not exists idx_audit_timestamp  on rag_audit_logs (timestamp desc);
create index if not exists idx_audit_signature  on rag_audit_logs (chain_signature);

alter table rag_audit_logs enable row level security;

-- NOTE: the library connects with the service role key, which bypasses row
-- level security. This policy protects reads made with user scoped JWTs
-- (for example from your own dashboard); it does not isolate tenants from
-- a holder of the service role key. See docs/threat-model.md.
drop policy if exists "workspace_isolation" on rag_audit_logs;
create policy "workspace_isolation" on rag_audit_logs
    for all
    using (workspace_id = current_setting('app.workspace_id', true));

create or replace view rag_audit_summary as
select
    workspace_id,
    count(*)                                         as total_queries,
    round(avg(latency_ms))                           as avg_latency_ms,
    round(avg(jsonb_array_length(retrieved_chunks))) as avg_chunks_retrieved,
    max(timestamp)                                   as last_query_at,
    min(timestamp)                                   as first_query_at
from rag_audit_logs
group by workspace_id;
