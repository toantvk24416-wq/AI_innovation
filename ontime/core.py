"""Các hàm theo tab "Hợp đồng" của Sheet:

  score_order(order)       -> {p_late, risk_level, rule_applied}
  recommend_promise(order) -> {p50, p80, p90, promise_days, delta_vs_cu, nguon}
  explain_order(order)     -> {p_late, reasons: [{rank, feature, label_vi, shap}]}
  assist_order(order)      -> {giai_thich, hanh_dong, tin_nhan_nhap, nguon, ...}
  ask(question)            -> chatbot.answer (Text-to-SQL trên tầng Gold)

order là dict 18 biến đầu vào (đơn mới) hoặc order_id (đơn có sẵn trong hàng đợi data/order_queue.csv).
"""
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from ontime import llm
from ontime.features import FEATURES, MODEL_VERSION, risk_tier, shap_by_feature, top_reasons

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"


# ------------------------------------------------------------------ nạp dữ liệu (đọc một lần, giữ trong bộ nhớ)
@st.cache_resource(show_spinner="Đang tải mô hình...")
def load_model():
    return joblib.load(BASE / "best_pipe.joblib")


@st.cache_resource(show_spinner=False)
def load_promise_models():
    return {k: joblib.load(BASE / "models" / f"promise_{k}.joblib") for k in ("p50", "p80", "p90")}


@st.cache_data(show_spinner="Đang tải hàng đợi đơn...")
def load_queue():
    q = pd.read_csv(DATA / "order_queue.csv")
    q["order_ts"] = pd.to_datetime(q["order_ts"])
    return q


@st.cache_data
def load_promise_table():
    return pd.read_csv(DATA / "promise_table.csv")


@st.cache_data
def load_evaluation():
    return json.loads((DATA / "evaluation.json").read_text(encoding="utf-8"))


@st.cache_data
def load_rules():
    return pd.read_csv(DATA / "luat_hanh_dong.csv", dtype=str).fillna("")


@st.cache_data
def top20_value():
    """Ngưỡng giá trị đơn thuộc top 20% (dùng trong luật R2)."""
    return float(load_queue()["order_value"].quantile(0.8))


def _row(order):
    """order_id -> dòng trong hàng đợi; dict -> giữ nguyên."""
    if isinstance(order, dict):
        return dict(order)
    q = load_queue()
    hit = q[q["order_id"] == int(order)]
    if hit.empty:
        raise KeyError(f"Không có đơn {order} trong hàng đợi")
    return hit.iloc[0].to_dict()


# ------------------------------------------------------------------ 5.2 chấm điểm đơn
def score_order(order):
    r = _row(order)
    if "p_late" in r and not isinstance(order, dict):
        p = float(r["p_late"])
    else:
        p = float(load_model().predict_proba(pd.DataFrame([r])[FEATURES])[0, 1])
    rule = "R1" if r["shipping_mode"] == "First Class" else None
    return {"p_late": p, "risk_level": risk_tier(p), "rule_applied": rule,
            "nguon": "Quy tắc First Class (luật R1)" if rule else "Mô hình XGBoost"}


# ------------------------------------------------------------------ 5.1 ngày hẹn giao
def recommend_promise(order):
    r = _row(order)
    chosen = load_evaluation()["promise"]["chosen"]
    if {"p50", "p80", "p90"} <= set(r) and not isinstance(order, dict):
        q = {k: float(r[k]) for k in ("p50", "p80", "p90")}
    elif chosen == "model":
        X = pd.DataFrame([r])[FEATURES]
        vals = sorted(float(m.predict(X)[0]) for m in load_promise_models().values())
        q = dict(zip(("p50", "p80", "p90"), vals))
    else:
        t = load_promise_table()
        hit = t[(t["order_region"] == r["order_region"]) & (t["shipping_mode"] == r["shipping_mode"])]
        if hit.empty:
            hit = t[t["shipping_mode"] == r["shipping_mode"]]
        q = {k: float(hit[k].mean()) for k in ("p50", "p80", "p90")}
    promise = math.ceil(q["p80"] - 1e-9)
    if r["shipping_mode"] == "First Class":
        promise = max(promise, 2)                                   # luật R1
    cur = int(r["scheduled_shipping_days"])
    return {**{k: round(v, 2) for k, v in q.items()}, "promise_days": promise, "delta_vs_cu": promise - cur,
            "nguon": "Bảng phân vị tuyến x phương thức" if chosen == "table" else "Mô hình hồi quy phân vị"}


# ------------------------------------------------------------------ 5.3 lý do từng đơn
def explain_order(order):
    r = _row(order)
    if "reasons_json" in r and not isinstance(order, dict):
        reasons = json.loads(r["reasons_json"])
    else:
        X = pd.DataFrame([r])
        reasons = top_reasons(r, shap_by_feature(load_model(), X).iloc[0])
    return {"p_late": score_order(order)["p_late"], "reasons": reasons}


# ------------------------------------------------------------------ 5.5 trợ lý đơn hàng
def actions_for(r):
    """Các luật trong data/luat_hanh_dong.csv khớp với đơn (sửa bảng là đổi chính sách, không cần sửa code)."""
    one = pd.DataFrame([{"shipping_mode": r["shipping_mode"], "scheduled_shipping_days": int(r["scheduled_shipping_days"]),
                         "risk_tier": r["risk_tier"], "order_value": float(r["order_value"])}])
    out = []
    for rule in load_rules().to_dict("records"):
        try:
            if len(one.query(rule["dieu_kien"], local_dict={"top20_value": top20_value()})):
                out.append(rule)
        except Exception:
            continue                                  # luật viết sai cú pháp thì bỏ qua, không làm hỏng màn hình
    return out


def order_date(r):
    ts = r.get("order_ts")
    return pd.to_datetime(ts).to_pydatetime() if ts is not None else datetime.now()


def build_payload(order):
    """Bước 1 của khung LLM: bản ghi JSON duy nhất mà LLM được đọc."""
    r = _row(order)
    if "risk_tier" not in r:
        r["risk_tier"] = score_order(r)["risk_level"]
    if "order_value" not in r:
        r["order_value"] = float(r.get("order_item_total", 0))
    s, pr, ex = score_order(order), recommend_promise(order), explain_order(order)
    due = order_date(r) + timedelta(days=pr["promise_days"])
    acts = actions_for(r)
    return {
        "ma_don": str(r.get("order_id", "MỚI")), "model_version": MODEL_VERSION,
        "phuong_thuc": r["shipping_mode"], "tuyen": f"{r['order_region']} - {r['shipping_mode']}",
        "xac_suat_tre": f"{s['p_late'] * 100:.1f}".replace(".", ",") + "%", "muc_rui_ro": s["risk_level"],
        "so_ngay_hen_cu": int(r["scheduled_shipping_days"]), "so_ngay_hen_moi": pr["promise_days"],
        "ngay_hen_giao": due.strftime("%d/%m/%Y"),
        "ly_do": [x["label_vi"] for x in ex["reasons"]],
        # không gửi mã luật (R1, R2...) cho LLM để bộ kiểm tra số liệu không bị nới lỏng bởi các chữ số trong mã
        "hanh_dong": [f"{a['hanh_dong']} ({a['nguoi_thuc_hien']})" for a in acts],
        "ma_luat": [a["ma"] for a in acts],
        "can_bao_khach": any(a["bao_khach"] == "1" for a in acts),
        "customer_name": r.get("customer_name", ""), "customer_address": r.get("customer_address", ""),
    }


def template(p):
    """Bước 6: câu mẫu ghép từ JSON khi không gọi được LLM hoặc đầu ra không qua kiểm tra."""
    def lower_first(x):            # "Đặt hàng lúc 14h" -> "đặt hàng...", giữ nguyên tên riêng như "Second Class"
        return x if x.split()[0] in ("First", "Second", "Same", "Standard") else x[0].lower() + x[1:]
    reasons = " và ".join(lower_first(x) for x in p["ly_do"][:2]) or "đặc điểm của tuyến giao"
    action = p["hanh_dong"][0] if p["hanh_dong"] else "theo dõi đơn"
    expl = f"Đơn {p['ma_don']} có nguy cơ trễ do {reasons}. Đề xuất: {action}."
    msg = (f"Chào {p.get('customer_name') or '[KHACH_HANG]'}, cảm ơn anh/chị đã đặt hàng. Đơn {p['ma_don']} "
           f"có thể đến muộn hơn thời gian hẹn ban đầu, dự kiến giao chậm nhất ngày {p['ngay_hen_giao']}. "
           "Shop sẽ cập nhật ngay khi đơn được bàn giao cho đơn vị vận chuyển.")
    return {"giai_thich": expl, "tin_nhan_nhap": msg}


def assist_order(order, use_llm=True):
    p = build_payload(order)
    for_llm = {k: v for k, v in p.items() if k != "ma_luat"}
    out = llm.compose(for_llm, template) if use_llm else {**template(for_llm), "nguon": "Câu mẫu", "ghi_chu": "Không dùng LLM"}
    return {**out, "hanh_dong": list(zip(p["ma_luat"], p["hanh_dong"])), "can_bao_khach": p["can_bao_khach"], "payload": p}


def ask(question, history=None):
    from chatbot import answer
    return answer(question, history or [])
