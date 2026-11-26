-- Apply to the dedicated LongTerm-Torn-Helper-ML Supabase project.
-- Additive only: does not modify existing tables, grants or RLS policies.
create table if not exists long_term.portfolio_allocations (
    allocation_id bigserial primary key,
    snapshot_week timestamptz not null,
    analysis_range_days integer not null check (analysis_range_days > 0),
    horizon_days integer not null check (horizon_days > 0),
    status text not null check (status in (
        'optimized', 'no_eligible_assets', 'insufficient_common_history'
    )),
    eligible_assets integer not null default 0 check (eligible_assets >= 0),
    allocations jsonb not null default '{}'::jsonb,
    metrics jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (snapshot_week, analysis_range_days, horizon_days)
);

create index if not exists portfolio_allocations_snapshot_idx
    on long_term.portfolio_allocations(snapshot_week desc, horizon_days);

alter table long_term.portfolio_allocations enable row level security;
revoke all on table long_term.portfolio_allocations from public, anon, authenticated;
revoke all on sequence long_term.portfolio_allocations_allocation_id_seq
    from public, anon, authenticated;
