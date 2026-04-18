-- RAGCompliance billing migration
-- Adds workspace subscription tracking + atomic usage counter.
-- Run in Supabase SQL editor after supabase_schema.sql.

create table if not exists public.workspace_subscriptions (
    workspace_id text primary key,
    stripe_customer_id text,
    stripe_subscription_id text,
    tier text not null default 'free',
    status text not null default 'inactive',
    current_period_end timestamptz,
    query_count_current_period integer not null default 0,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists workspace_subscriptions_customer_idx
    on public.workspace_subscriptions (stripe_customer_id);
create index if not exists workspace_subscriptions_status_idx
    on public.workspace_subscriptions (status);

-- Row-level security: a workspace can only read/write its own row.
alter table public.workspace_subscriptions enable row level security;

drop policy if exists workspace_subscriptions_own_row on public.workspace_subscriptions;
create policy workspace_subscriptions_own_row
    on public.workspace_subscriptions
    for all
    using (workspace_id = coalesce(current_setting('request.jwt.claim.workspace_id', true), workspace_id))
    with check (workspace_id = coalesce(current_setting('request.jwt.claim.workspace_id', true), workspace_id));

-- Atomic usage counter: creates the row if missing, increments if present,
-- returns the new count. Called from Python via supabase-py .rpc().
create or replace function public.increment_workspace_usage(p_workspace_id text)
returns table(query_count_current_period integer)
language plpgsql
security definer
as $$
begin
    insert into public.workspace_subscriptions (workspace_id, query_count_current_period)
    values (p_workspace_id, 1)
    on conflict (workspace_id) do update
    set query_count_current_period = public.workspace_subscriptions.query_count_current_period + 1,
        updated_at = now()
    returning public.workspace_subscriptions.query_count_current_period into query_count_current_period;

    return next;
end;
$$;

grant execute on function public.increment_workspace_usage(text) to anon, authenticated, service_role;

-- Reset counter at the start of each billing period. Call from the
-- customer.subscription.updated webhook when current_period_end advances.
create or replace function public.reset_workspace_usage(p_workspace_id text)
returns void
language sql
security definer
as $$
    update public.workspace_subscriptions
    set query_count_current_period = 0, updated_at = now()
    where workspace_id = p_workspace_id;
$$;

grant execute on function public.reset_workspace_usage(text) to anon, authenticated, service_role;
