-- Apply once to the dedicated long-term Supabase database.
-- Does not alter grants, RLS settings or existing policies.
create table if not exists long_term.forecast_calibrations (
    calibration_id bigserial primary key,
    stock_id integer not null,
    analysis_range_days integer not null check (analysis_range_days > 0),
    horizon_days integer not null check (horizon_days > 0),
    version integer not null check (version > 0),
    scale double precision not null,
    bias double precision not null,
    sample_count integer not null,
    training_samples integer not null,
    holdout_samples integer not null,
    raw_holdout_mae double precision not null,
    calibrated_holdout_mae double precision not null,
    holdout_mae_improvement double precision not null,
    status text not null check (status in ('active', 'superseded')),
    metrics jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (stock_id, analysis_range_days, horizon_days, version)
);
create unique index if not exists forecast_calibrations_one_active
    on long_term.forecast_calibrations(stock_id, analysis_range_days, horizon_days)
    where status = 'active';
create index if not exists forecast_calibrations_lookup
    on long_term.forecast_calibrations(stock_id, analysis_range_days, horizon_days, status);

-- Match the existing private-schema posture: no browser-role access and no
-- policies are created. The server-side database owner remains able to write.
alter table long_term.forecast_calibrations enable row level security;
revoke all on table long_term.forecast_calibrations from public, anon, authenticated;
revoke all on sequence long_term.forecast_calibrations_calibration_id_seq
    from public, anon, authenticated;
