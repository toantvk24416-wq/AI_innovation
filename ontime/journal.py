"""Message review log (Section 5.5): who reviewed, when, what was edited. Stored in SQLite next to the app (data/review_log.db)."""
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DB = Path(__file__).resolve().parents[1] / "data" / "review_log.db"


def _conn():
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS review_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, order_id TEXT, reviewer TEXT, decision TEXT,
        draft_source TEXT, draft TEXT, final_text TEXT, edited INTEGER)""")
    return c


def add(order_id, reviewer, decision, source, draft, final_text):
    with _conn() as c:
        c.execute("INSERT INTO review_log (ts, order_id, reviewer, decision, draft_source, draft, final_text, edited)"
                  " VALUES (?,?,?,?,?,?,?,?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(order_id), reviewer, decision, source,
                   draft, final_text, int(draft.strip() != final_text.strip())))


def read():
    with _conn() as c:
        return pd.read_sql("SELECT * FROM review_log ORDER BY id DESC", c)
