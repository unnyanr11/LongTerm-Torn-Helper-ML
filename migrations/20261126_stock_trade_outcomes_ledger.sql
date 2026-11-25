-- Stock trade-outcome ledger belongs only in the dedicated stock-central database.
-- The main Torn-Helper Supabase project must not receive stock tables or cron jobs.
begin;
create schema if not exists stock_central;

create table if not exists stock_central.trade_outcome_forecasts (
  id bigint generated always as identity primary key,
  stock_id smallint not null check (stock_id between 1 and 35),
  strategy text not null check (strategy in ('momentum','dip')),
  horizon_hours smallint not null check (horizon_hours in (3,6,12)),
  plan text not null check (plan in ('quick','balanced','wide')),
  origin_at timestamptz not null,
  price numeric not null check (price > 0),
  target_net numeric not null check (target_net > 0 and target_net < 0.1),
  stop_net numeric not null check (stop_net > 0 and stop_net < 0.1),
  neighbors smallint not null default 0 check (neighbors >= 0),
  target_probability numeric check (target_probability between 0 and 1),
  stop_probability numeric check (stop_probability between 0 and 1),
  timeout_probability numeric check (timeout_probability between 0 and 1),
  expected_net numeric,
  validation jsonb not null default '{}'::jsonb,
  model_version text not null,
  created_at timestamptz not null default now(),
  unique(stock_id,strategy,horizon_hours,plan,origin_at)
);
create index if not exists trade_outcome_forecasts_latest_idx
  on stock_central.trade_outcome_forecasts(stock_id,origin_at desc);

create table if not exists stock_central.trade_outcome_results (
  forecast_id bigint primary key references stock_central.trade_outcome_forecasts(id),
  status text not null check (status in ('target','stop','timeout','unavailable')),
  net_return numeric,
  entry_at timestamptz,
  entry_price numeric,
  exit_at timestamptz,
  exit_price numeric,
  reconciled_at timestamptz not null default now()
);

create table if not exists stock_central.trade_outcome_settings (
  user_id uuid primary key,
  min_expected_net_profit_pct numeric not null default 0.2
    check (min_expected_net_profit_pct >= 0),
  updated_at timestamptz not null default now()
);

alter table stock_central.trade_outcome_forecasts enable row level security;
alter table stock_central.trade_outcome_results enable row level security;
alter table stock_central.trade_outcome_settings enable row level security;
revoke all on schema stock_central from public, anon, authenticated;
revoke all on stock_central.trade_outcome_forecasts, stock_central.trade_outcome_results,
  stock_central.trade_outcome_settings from public, anon, authenticated;
grant usage on schema stock_central to service_role;
grant select,insert,update on stock_central.trade_outcome_forecasts to service_role;
grant select,insert,update on stock_central.trade_outcome_results to service_role;
grant select,insert,update on stock_central.trade_outcome_settings to service_role;
grant usage,select on all sequences in schema stock_central to service_role;
commit;
