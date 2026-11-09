"""Nhật ký duyệt tin nhắn (Mục 5.5): ai duyệt, lúc nào, sửa gì. Lưu SQLite cạnh app (data/nhat_ky_duyet.db)."""
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DB = Path(__file__).resolve().parents[1] / "data" / "nhat_ky_duyet.db"


def _conn():
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS nhat_ky (
        id INTEGER PRIMARY KEY AUTOINCREMENT, thoi_gian TEXT, ma_don TEXT, nguoi_duyet TEXT, quyet_dinh TEXT,
        nguon_ban_nhap TEXT, ban_nhap TEXT, ban_cuoi TEXT, da_sua INTEGER)""")
    return c


def add(ma_don, nguoi_duyet, quyet_dinh, nguon, ban_nhap, ban_cuoi):
    with _conn() as c:
        c.execute("INSERT INTO nhat_ky (thoi_gian, ma_don, nguoi_duyet, quyet_dinh, nguon_ban_nhap, ban_nhap, ban_cuoi, da_sua)"
                  " VALUES (?,?,?,?,?,?,?,?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(ma_don), nguoi_duyet, quyet_dinh, nguon,
                   ban_nhap, ban_cuoi, int(ban_nhap.strip() != ban_cuoi.strip())))


def read():
    with _conn() as c:
        return pd.read_sql("SELECT * FROM nhat_ky ORDER BY id DESC", c)
