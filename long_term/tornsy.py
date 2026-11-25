"""Tornsy weekly OHLC historical client."""
from datetime import datetime, timezone
import requests

BASE = "https://tornsy.com/api"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "LongTerm-Torn-Helper-ML/1.0 (historical research)"})

def fetch_weekly(symbol: str, start: datetime, end: datetime):
    params = {
        "interval": "w1",
        "from": int(start.timestamp()),
        "to": int(end.timestamp()),
        "limit": 2000,
    }
    response = SESSION.get(f"{BASE}/{symbol.lower()}", params=params, timeout=45)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected Tornsy response for {symbol}")
    return rows
