"""Database access. Uses only the dedicated long-term Supabase Postgres URL."""
import os
from contextlib import contextmanager
import psycopg

def database_url():
    value = os.environ.get("LONG_TERM_DATABASE_URL")
    if not value:
        raise RuntimeError("Missing LONG_TERM_DATABASE_URL secret")
    return value

@contextmanager
def connect():
    with psycopg.connect(database_url(), connect_timeout=15) as conn:
        yield conn
