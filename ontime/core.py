"""Functions behind the "Contract" tab of the Sheet:

  score_order(order)       -> {p_late, risk_level, rule_applied, source}
  recommend_promise(order) -> {p50, p80, p90, promise_days, delta_vs_old, source}
  explain_order(order)     -> {p_late, reasons: [{rank, feature, label, shap}]}
  assist_order(order)      -> {explanation, action, draft_message, source, ...}
  ask(question)            -> chatbot.answer (Text-to-SQL on the Gold layer)

order is a dict of the 18 input features (new order) or an order_id (an order already in data/order_queue.csv).
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


# ------------------------------------------------------------------ load data (read once, kept in memory)
@st.cache_resource(show_spinner="Loading the model...")
def load_model():
    return joblib.load(BASE / "best_pipe.joblib")


@st.cache_resource(show_spinner=False)
def load_promise_models():
    return {k: joblib.load(BASE / "models" / f"promise_{k}.joblib") for k in ("p50", "p80", "p90")}


@st.cache_data(show_spinner="Loading the order queue...")
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
    """Order value threshold for the top 20% (used by rule R2)."""
    return float(load_queue()["order_value"].quantile(0.8))


def _row(order):
    """order_id -> row of the queue; dict -> kept as is."""
    if isinstance(order, dict):
        return dict(order)
    q = load_queue()
    hit = q[q["order_id"] == int(order)]
    if hit.empty:
        raise KeyError(f"Order {order} is not in the queue")
    return hit.iloc[0].to_dict()


# ------------------------------------------------------------------ 5.2 score an order
def score_order(order):
    r = _row(order)
    if "p_late" in r and not isinstance(order, dict):
        p = float(r["p_late"])
    else:
        p = float(load_model().predict_proba(pd.DataFrame([r])[FEATURES])[0, 1])
    rule = "R1" if r["shipping_mode"] == "First Class" else None
    return {"p_late": p, "risk_level": risk_tier(p), "rule_applied": rule,
            "source": "First Class rule (R1)" if rule else "XGBoost model"}


# ------------------------------------------------------------------ 5.1 promised delivery date
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
        promise = max(promise, 2)                                   # rule R1
    cur = int(r["scheduled_shipping_days"])
    return {**{k: round(v, 2) for k, v in q.items()}, "promise_days": promise, "delta_vs_old": promise - cur,
            "source": "Route x mode quantile table" if chosen == "table" else "Quantile regression model"}


# ------------------------------------------------------------------ 5.3 reasons per order
def explain_order(order):
    r = _row(order)
    if "reasons_json" in r and not isinstance(order, dict):
        reasons = json.loads(r["reasons_json"])
    else:
        X = pd.DataFrame([r])
        reasons = top_reasons(r, shap_by_feature(load_model(), X).iloc[0])
    return {"p_late": score_order(order)["p_late"], "reasons": reasons}


# ------------------------------------------------------------------ 5.5 order assistant
def actions_for(r):
    """Rules in data/luat_hanh_dong.csv that match the order (edit the table to change the policy, no code change)."""
    one = pd.DataFrame([{"shipping_mode": r["shipping_mode"], "scheduled_shipping_days": int(r["scheduled_shipping_days"]),
                         "risk_tier": r["risk_tier"], "order_value": float(r["order_value"])}])
    out = []
    for rule in load_rules().to_dict("records"):
        try:
            if len(one.query(rule["condition"], local_dict={"top20_value": top20_value()})):
                out.append(rule)
        except Exception:
            continue                                  # a rule with a syntax error is skipped, it must not break the screen
    return out


def order_date(r):
    ts = r.get("order_ts")
    return pd.to_datetime(ts).to_pydatetime() if ts is not None else datetime.now()


def build_payload(order):
    """Step 1 of the LLM framework: the only JSON record the LLM is allowed to read."""
    r = _row(order)
    if "risk_tier" not in r:
        r["risk_tier"] = score_order(r)["risk_level"]
    if "order_value" not in r:
        r["order_value"] = float(r.get("order_item_total", 0))
    s, pr, ex = score_order(order), recommend_promise(order), explain_order(order)
    due = order_date(r) + timedelta(days=pr["promise_days"])
    acts = actions_for(r)
    return {
        "order_id": str(r.get("order_id", "NEW")), "model_version": MODEL_VERSION,
        "shipping_mode": r["shipping_mode"], "route": f"{r['order_region']} - {r['shipping_mode']}",
        "late_probability": f"{s['p_late'] * 100:.1f}%", "risk_level": s["risk_level"],
        "old_promise_days": int(r["scheduled_shipping_days"]), "new_promise_days": pr["promise_days"],
        "promised_delivery_date": due.strftime("%d %b %Y"),
        "reasons": [x["label"] for x in ex["reasons"]],
        # rule codes (R1, R2...) are not sent to the LLM, so digits in the codes cannot loosen the number check
        "actions": [f"{a['action']} ({a['owner']})" for a in acts],
        "rule_codes": [a["code"] for a in acts],
        "needs_customer_notice": any(a["notify_customer"] == "1" for a in acts),
        "customer_name": r.get("customer_name", ""), "customer_address": r.get("customer_address", ""),
    }


def template(p):
    """Step 6: fallback text built from the JSON when the LLM is unavailable or its output fails the checks."""
    reasons = "; ".join(p["reasons"][:2]) or "the characteristics of this route"
    action = p["actions"][0] if p["actions"] else "keep monitoring the order"
    expl = (f"Order {p['order_id']} is at risk of arriving late. Main factors: {reasons}. "
            f"Suggested action: {action}.")
    msg = (f"Hello {p.get('customer_name') or '[CUSTOMER_NAME]'}, thank you for your order. Order {p['order_id']} "
           f"may arrive later than the original promised date; we now expect to deliver it by "
           f"{p['promised_delivery_date']} at the latest. We will update you as soon as it is handed over to the carrier.")
    return {"explanation": expl, "draft_message": msg}


def assist_order(order, use_llm=True):
    p = build_payload(order)
    for_llm = {k: v for k, v in p.items() if k != "rule_codes"}
    out = llm.compose(for_llm, template) if use_llm else {**template(for_llm), "source": "Template", "note": "LLM not used"}
    return {**out, "actions": list(zip(p["rule_codes"], p["actions"])), "needs_customer_notice": p["needs_customer_notice"],
            "payload": p}


def ask(question, history=None):
    from chatbot import answer
    return answer(question, history or [])
