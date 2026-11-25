"""Resumable five-year weekly OHLC backfill into long_term.weekly_prices."""
from datetime import datetime, timedelta, timezone
import argparse
import math
from statistics import pstdev
from .db import connect
from .stock_universe import STOCKS
from .tornsy import fetch_weekly

def utc_week(ts: int):
    dt = datetime.fromtimestamp(int(ts), timezone.utc)
    start = (dt - timedelta(days=dt.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=7)

def parse_candle(row):
    if not isinstance(row, list) or len(row) < 5:
        raise ValueError("Malformed weekly candle")
    ts = int(row[0])
    o, h, l, c = (float(row[i]) for i in range(1, 5))
    if min(o, h, l, c) <= 0 or h < max(o, c, l) or l > min(o, c, h):
        raise ValueError(f"Invalid OHLC candle at {ts}")
    return ts, o, h, l, c

def ingest_stock(stock_id: int, symbol: str, start: datetime, end: datetime):
    raw = fetch_weekly(symbol, start, end)
    candles = sorted((parse_candle(x) for x in raw), key=lambda x: x[0])
    if len(candles) > 2000:
        raise RuntimeError(f"{symbol}: API returned more than 2,000 rows; split date range")
    if len({x[0] for x in candles}) != len(candles):
        raise ValueError(f"{symbol}: duplicate candle timestamps")
    now = datetime.now(timezone.utc)
    by_week = {}
    for ts, o, h, l, c in candles:
        ws, we = utc_week(ts)
        item = by_week.setdefault(ws, {"open": o, "high": h, "low": l, "close": c,
                                      "count": 0, "first": ts, "last": ts, "closes": []})
        item["high"] = max(item["high"], h)
        item["low"] = min(item["low"], l)
        item["close"] = c
        item["count"] += 1
        item["last"] = ts
        item["closes"].append(c)

    records = []
    previous_close = None
    for ws, v in sorted(by_week.items()):
        returns = []
        if previous_close and previous_close > 0:
            returns.append(math.log(v["close"] / previous_close))
        # Weekly candles are already aggregated; do not fabricate minute-level
        # volatility. A one-candle weekly observation has unknown intraweek volatility.
        weekly_vol = (pstdev(returns) * 100.0) if len(returns) > 1 else None
        ret = ((v["close"] / previous_close) - 1.0) * 100.0 if previous_close else None
        is_complete = ws + timedelta(days=7) <= now
        records.append((stock_id, ws, ws + timedelta(days=7), v["open"], v["high"],
                        v["low"], v["close"], sum(v["closes"])/len(v["closes"]),
                        ret, weekly_vol, v["count"],
                        datetime.fromtimestamp(v["first"], timezone.utc),
                        datetime.fromtimestamp(v["last"], timezone.utc),
                        "tornsy_w1", is_complete, None))
        previous_close = v["close"]

    with connect() as conn:
        with conn.cursor() as cur:
            cur.executemany("""
                insert into long_term.weekly_prices
                (stock_id,week_start,week_end,open_price,high_price,low_price,close_price,
                 average_price,return_pct,volatility_pct,observation_count,
                 first_observed_at,last_observed_at,source,is_complete,coverage_ratio)
                values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                on conflict (stock_id,week_start) do update set
                 week_end=excluded.week_end, open_price=excluded.open_price,
                 high_price=excluded.high_price, low_price=excluded.low_price,
                 close_price=excluded.close_price, average_price=excluded.average_price,
                 return_pct=excluded.return_pct, volatility_pct=excluded.volatility_pct,
                 observation_count=excluded.observation_count,
                 first_observed_at=excluded.first_observed_at,
                 last_observed_at=excluded.last_observed_at, source=excluded.source,
                 is_complete=excluded.is_complete, coverage_ratio=excluded.coverage_ratio,
                 ingested_at=now()
            """, records)
            earliest = min((r[1] for r in records), default=None)
            latest = max((r[2] for r in records), default=None)
            status = "complete" if earliest and earliest <= start + timedelta(days=10) else "partial"
            cur.execute("""
                insert into long_term.backfill_state
                (stock_id,source,requested_from,earliest_available_at,latest_available_at,
                 pages_completed,rows_written,status,last_error_code,updated_at)
                values (%s,'tornsy_w1',%s,%s,%s,1,%s,%s,null,now())
                on conflict (stock_id) do update set
                 source=excluded.source, requested_from=excluded.requested_from,
                 earliest_available_at=excluded.earliest_available_at,
                 latest_available_at=excluded.latest_available_at,
                 pages_completed=long_term.backfill_state.pages_completed+1,
                 rows_written=excluded.rows_written,status=excluded.status,
                 last_error_code=null,updated_at=now()
            """, (stock_id, start, earliest, latest, len(records), status))
        conn.commit()
    return {"stock_id": stock_id, "symbol": symbol, "rows": len(records),
            "earliest": earliest.isoformat() if earliest else None,
            "latest": latest.isoformat() if latest else None, "status": status}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--stock-id", type=int)
    args = parser.parse_args()
    if not 1 <= args.years <= 10:
        raise SystemExit("--years must be between 1 and 10")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365.2425 * args.years)
    selected = {args.stock_id: STOCKS[args.stock_id]} if args.stock_id else STOCKS
    for stock_id, symbol in selected.items():
        try:
            print(ingest_stock(stock_id, symbol, start, end), flush=True)
        except Exception as exc:
            with connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                      insert into long_term.backfill_state
                      (stock_id,source,requested_from,status,last_error_code,updated_at)
                      values (%s,'tornsy_w1',%s,'failed',%s,now())
                      on conflict (stock_id) do update set status='failed',
                      last_error_code=excluded.last_error_code,updated_at=now()
                    """, (stock_id, start, type(exc).__name__[:120]))
                conn.commit()
            print({"stock_id": stock_id, "symbol": symbol, "error": str(exc)}, flush=True)
            raise

if __name__ == "__main__":
    main()
