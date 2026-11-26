"""Stable numbered algorithm registry for long-term stock-specific forecasts."""
from __future__ import annotations

ALGORITHM_REGISTRY = {
    1: "ridge",
    2: "elastic_net",
    3: "random_forest",
    4: "extra_trees",
    5: "hist_gradient_boosting",
    6: "ensemble",
}
FAMILY_TO_ALGORITHM_ID = {family: algorithm_id
                          for algorithm_id, family in ALGORITHM_REGISTRY.items()}


def algorithm_id_for_family(family: str) -> int:
    try:
        return FAMILY_TO_ALGORITHM_ID[family]
    except KeyError as exc:
        raise ValueError(f"Unregistered long-term algorithm: {family}") from exc


def family_for_algorithm_id(algorithm_id: int) -> str:
    try:
        return ALGORITHM_REGISTRY[int(algorithm_id)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Unknown long-term algorithm ID: {algorithm_id}") from exc


def ensure_algorithm_registry(cur) -> None:
    """Create the additive private registry if this deployment has not migrated yet."""
    cur.execute("""
        create table if not exists long_term.stock_algorithm_registry (
            stock_id integer not null,
            analysis_range_days integer not null check (analysis_range_days > 0),
            horizon_days integer not null check (horizon_days > 0),
            algorithm_id integer not null check (algorithm_id between 1 and 6),
            model_family text not null,
            version integer not null default 1 check (version > 0),
            data_through date,
            source_model_run_id text,
            metrics jsonb not null default '{}'::jsonb,
            selected_at timestamptz not null default now(),
            primary key (stock_id, analysis_range_days, horizon_days),
            check (model_family in (
                'ridge','elastic_net','random_forest','extra_trees',
                'hist_gradient_boosting','ensemble'
            ))
        )
    """)
    cur.execute("alter table long_term.stock_algorithm_registry enable row level security")
    cur.execute("revoke all on table long_term.stock_algorithm_registry from public, anon, authenticated")


def load_active_algorithms(cur, stock_id: int) -> dict[tuple[int, int], dict]:
    cur.execute("""
        select analysis_range_days, horizon_days, algorithm_id, model_family, version
        from long_term.stock_algorithm_registry
        where stock_id = %s
    """, (stock_id,))
    return {
        (int(row[0]), int(row[1])): {
            "algorithm_id": int(row[2]),
            "model_family": row[3],
            "version": int(row[4]),
        }
        for row in cur.fetchall()
    }


def publish_algorithm(cur, *, stock_id: int, analysis_range_days: int,
                      horizon_days: int, model_family: str, data_through,
                      source_model_run_id, metrics: dict) -> dict:
    algorithm_id = algorithm_id_for_family(model_family)
    cur.execute("""
        insert into long_term.stock_algorithm_registry
            (stock_id, analysis_range_days, horizon_days, algorithm_id,
             model_family, version, data_through, source_model_run_id, metrics)
        values (%s,%s,%s,%s,%s,1,%s,%s,%s::jsonb)
        on conflict (stock_id, analysis_range_days, horizon_days)
        do update set
            version = case
                when long_term.stock_algorithm_registry.algorithm_id <> excluded.algorithm_id
                  or long_term.stock_algorithm_registry.model_family <> excluded.model_family
                then long_term.stock_algorithm_registry.version + 1
                else long_term.stock_algorithm_registry.version
            end,
            algorithm_id = excluded.algorithm_id,
            model_family = excluded.model_family,
            data_through = excluded.data_through,
            source_model_run_id = excluded.source_model_run_id,
            metrics = excluded.metrics,
            selected_at = now()
        returning algorithm_id, model_family, version
    """, (stock_id, analysis_range_days, horizon_days, algorithm_id,
          model_family, data_through, str(source_model_run_id),
          __import__("json").dumps(metrics, sort_keys=True)))
    row = cur.fetchone()
    return {"algorithm_id": int(row[0]), "model_family": row[1], "version": int(row[2])}
