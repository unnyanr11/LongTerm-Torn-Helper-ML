-- Additive private registry for per-stock/per-analysis-range/per-horizon winners.
-- Does not modify existing tables, grants, or RLS policies.
create table if not exists long_term.stock_algorithm_registry (
    stock_id integer not null,
    analysis_range_days integer not null check (analysis_range_days > 0),
    horizon_days integer not null check (horizon_days > 0),
    algorithm_id integer not null check (algorithm_id between 1 and 6),
    model_family text not null check (model_family in (
        'ridge','elastic_net','random_forest','extra_trees',
        'hist_gradient_boosting','ensemble'
    )),
    version integer not null default 1 check (version > 0),
    data_through date,
    source_model_run_id text,
    metrics jsonb not null default '{}'::jsonb,
    selected_at timestamptz not null default now(),
    primary key (stock_id, analysis_range_days, horizon_days)
);
alter table long_term.stock_algorithm_registry enable row level security;
revoke all on table long_term.stock_algorithm_registry from public, anon, authenticated;
