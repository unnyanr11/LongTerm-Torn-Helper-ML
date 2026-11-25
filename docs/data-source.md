# Historical data source verification

## Tornsy API contract

The public documentation at https://tornsy.com/api documents:
- Historical endpoint: `GET https://tornsy.com/api/:stock`
- `interval=w1` for weekly OHLC candles
- `from` inclusive and `to` exclusive Unix timestamps
- `limit` from 1 to 2,000 rows (default 1,000)
- Candle arrays: timestamp, open, high, low, close, total shares
- No API key required

Five years of weekly data is approximately 261 rows per stock, within the documented per-request limit. This does not guarantee that five years exist for every symbol. The importer records the actual earliest/latest timestamps and marks coverage as complete only if it reaches the requested start boundary; otherwise it is partial.

## Important metric limitation

Tornsy weekly OHLC candles do not contain the minute observations used to calculate true intraweek realized volatility or an arithmetic average of every minute price. The current importer stores the candle close as weekly close and computes the average of available candle closes in that week (normally one). It deliberately leaves volatility unknown when there are insufficient weekly return observations. Do not interpret this as minute-level volatility.

## Run

Manual GitHub Actions workflow: **Long-term weekly stock history**. Default request is five years. The workflow uses the GitHub Actions secret `LONG_TERM_DATABASE_URL`.

The weekly job refreshes the requested five-year window each Monday, updating the current partial week and idempotently upserting prior weeks.
