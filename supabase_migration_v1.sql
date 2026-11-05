-- RAGCompliance 1.0: hash chain and lineage columns for rag_audit_logs.
--
-- Optional. Without it the 1.0 sink keeps writing the 0.1.x column set and
-- nests the new fields under extra->'_ragcompliance' (layout 1). After
-- running it, set RAGCOMPLIANCE_SUPABASE_SCHEMA_VERSION=2 so the sink uses
-- these columns (layout 2) and the unique index serialises concurrent
-- writers on the same chain.
--
-- Safe to re-run.

alter table rag_audit_logs add column if not exists kind            text    not null default 'rag_invocation';
alter table rag_audit_logs add column if not exists actor           jsonb   not null default '{}';
alter table rag_audit_logs add column if not exists lineage         jsonb   not null default '{}';
alter table rag_audit_logs add column if not exists schema_version  integer not null default 1;
alter table rag_audit_logs add column if not exists tool_version    text    not null default '';
alter table rag_audit_logs add column if not exists chain_id        text    not null default 'default';
alter table rag_audit_logs add column if not exists sequence        bigint;
alter table rag_audit_logs add column if not exists prev_hash       text    not null default '';
alter table rag_audit_logs add column if not exists record_hash     text    not null default '';
alter table rag_audit_logs add column if not exists record_hmac     text    not null default '';

-- One sequence number per (workspace, chain). A duplicate key error tells a
-- second writer to re-read the tail and retry, which keeps the chain linear.
create unique index if not exists idx_audit_chain_sequence
    on rag_audit_logs (workspace_id, chain_id, sequence)
    where sequence is not null;

create index if not exists idx_audit_actor on rag_audit_logs ((actor->>'id'));

-- Back-fill layout 1 rows that already carry the nested fields.
update rag_audit_logs
set kind           = coalesce(extra->'_ragcompliance'->>'kind', kind),
    actor          = coalesce(extra->'_ragcompliance'->'actor', actor),
    lineage        = coalesce(extra->'_ragcompliance'->'lineage', lineage),
    schema_version = coalesce((extra->'_ragcompliance'->>'schema_version')::integer, schema_version),
    tool_version   = coalesce(extra->'_ragcompliance'->>'tool_version', tool_version),
    chain_id       = coalesce(extra->'_ragcompliance'->>'chain_id', chain_id),
    sequence       = coalesce((extra->'_ragcompliance'->>'sequence')::bigint, sequence),
    prev_hash      = coalesce(extra->'_ragcompliance'->>'prev_hash', prev_hash),
    record_hash    = coalesce(extra->'_ragcompliance'->>'record_hash', record_hash),
    record_hmac    = coalesce(extra->'_ragcompliance'->>'record_hmac', record_hmac),
    extra          = extra - '_ragcompliance'
where extra ? '_ragcompliance';

-- Append-only guard. Purges set the session flag inside their transaction.
create or replace function rag_audit_logs_append_only() returns trigger as $$
begin
    if current_setting('ragcompliance.allow_purge', true) = 'on' then
        return old;
    end if;
    raise exception 'ragcompliance audit log is append-only';
end;
$$ language plpgsql;

drop trigger if exists rag_audit_logs_append_only on rag_audit_logs;
create trigger rag_audit_logs_append_only
    before update or delete on rag_audit_logs
    for each row execute function rag_audit_logs_append_only();
