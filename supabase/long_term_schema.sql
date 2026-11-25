-- Long-term weekly history schema.
-- Run only in the dedicated long-term Supabase project.
-- Private schema: no browser/client access is granted here.
begin;

create schema if not exists long_term;

create table if not exists long_term.weekly_prices (
  stock_id integer not null check (stock_id between 1 and 35),
  week_start timestamptz not null,
  week_end timestamptz not null,
  open_price numeric(20,6),
  high_price numeric(20,6),
  low_price numeric(20,6),
  close_price numeric(20,6),
  average_price numeric(20,6),
  return_pct double precision,
  volatility_pct double precision,
  observation_count integer not null default 0 check (observation_count >= 0),
  first_observed_at timestamptz,
  last_observed_at timestamptz,
  source text not null,
  is_complete boolean not null default false,
  coverage_ratio double precision check (coverage_ratio is null or coverage_ratio between 0 and 1),
  ingested_at timestamptz not null default now(),
  primary key (stock_id, week_start),
  check (week_end > week_start),
  check (high_price is null or low_price is null or high_price >= low_price),
  check (open_price is null or open_price > 0),
  check (high_price is null or high_price > 0),
  check (low_price is null or low_price > 0),
  check (close_price is null or close_price > 0),
  check (average_price is null or average_price > 0)
);

create index if not exists weekly_prices_stock_week_desc
  on long_term.weekly_prices (stock_id, week_start desc);

create table if not exists long_term.backfill_state (
  stock_id integer primary key check (stock_id between 1 and 35),
  source text not null,
  requested_from timestamptz not null,
  earliest_available_at timestamptz,
  latest_available_at timestamptz,
  next_cursor text,
  pages_completed integer not null default 0,
  rows_written bigint not null default 0,
  status text not null default 'pending'
    check (status in ('pending','running','complete','partial','failed')),
  last_error_code text,
  updated_at timestamptz not null default now()
);

create table if not exists long_term.model_runs (
  run_id uuid primary key default gen_random_uuid(),
  stock_id integer not null check (stock_id between 1 and 35),
  model_name text not null,
  model_version text not null,
  data_through timestamptz not null,
  training_start timestamptz not null,
  validation_start timestamptz not null,
  test_start timestamptz,
  metrics jsonb not null default '{}'::jsonb,
  status text not null check (status in ('research','validated','rejected','failed')),
  created_at timestamptz not null default now(),
  check (validation_start > training_start)
);

create index if not exists model_runs_stock_created
  on long_term.model_runs (stock_id, created_at desc);

create table if not exists long_term.forecasts (
  forecast_id uuid primary key default gen_random_uuid(),
  stock_id integer not null check (stock_id between 1 and 35),
  model_run_id uuid not null references long_term.model_runs(run_id),
  issued_at timestamptz not null default now(),
  origin_week timestamptz not null,
  target_week timestamptz not null,
  origin_price numeric(20,6) not null check (origin_price > 0),
  predicted_price numeric(20,6) not null check (predicted_price > 0),
  predicted_return_pct double precision not null,
  signal text not null check (signal in ('buy','sell','watch','wait')),
  validation jsonb not null default '{}'::jsonb,
  analysis_range_days integer not null check (analysis_range_days > 0),
  horizon_days integer not null check (horizon_days > 0),
  unique (stock_id, model_run_id, analysis_range_days, horizon_days),
  check (target_week > origin_week)
);

create index if not exists forecasts_stock_target_idx
  on long_term.forecasts (stock_id, target_week desc);

alter table long_term.weekly_prices enable row level security;
alter table long_term.backfill_state enable row level security;
alter table long_term.model_runs enable row level security;
alter table long_term.forecasts enable row level security;

revoke all on schema long_term from public, anon, authenticated;
revoke all on all tables in schema long_term from public, anon, authenticated;

commit;
