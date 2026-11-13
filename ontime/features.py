"""Shared by the build script and the app: the 18 order-time features, risk tiers, SHAP and English reason labels."""
import calendar

import numpy as np
import pandas as pd
import xgboost as xgb

CAT_COLS = ["shipping_mode", "market", "order_region", "order_country",
            "category_name", "department_name", "customer_segment", "payment_type"]
NUM_COLS = ["scheduled_shipping_days", "quantity", "unit_price", "sales", "discount_rate",
            "order_item_total", "order_month", "order_day_of_week", "order_is_weekend", "order_hour"]
FEATURES = CAT_COLS + NUM_COLS
QUANTILES = {"p50": 0.5, "p80": 0.8, "p90": 0.9}
MODEL_VERSION = "xgb_2026-11"
DOW_EN = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday", 7: "Saturday"}


def risk_tier(p):
    """Order risk tier: High from 60%, Medium from 30%, Low below 30%."""
    p = np.asarray(p, dtype=float)
    out = np.where(p >= 0.6, "High", np.where(p >= 0.3, "Medium", "Low"))
    return out if out.ndim else str(out)


def reason_label(feature, r):
    """Maps a raw feature to a plain English sentence (rule-based, no LLM, so a factor always has the same wording)."""
    m, d = r["shipping_mode"], int(r["scheduled_shipping_days"])
    days = "same-day" if d == 0 else f"{d}-day"
    return {
        "shipping_mode": f"{m} promises {days} delivery, often exceeded",
        "scheduled_shipping_days": f"{days.capitalize()} deadline is short compared with the route's real delivery time",
        "order_hour": f"Ordered at {int(r['order_hour'])}:00",
        "order_country": f"Ships to {r['order_country']}",
        "order_region": f"Destination region: {r['order_region']}",
        "market": f"Market: {r['market']}",
        "category_name": f"Category: {r['category_name']}",
        "department_name": f"Department: {r['department_name']}",
        "customer_segment": f"Customer segment: {r['customer_segment']}",
        "payment_type": f"Payment: {r['payment_type']}",
        "order_month": f"Ordered in {calendar.month_name[int(r['order_month'])]}",
        "order_day_of_week": f"Ordered on {DOW_EN[int(r['order_day_of_week'])]}",
        "order_is_weekend": "Ordered on a weekend" if r["order_is_weekend"] else "Ordered on a weekday",
        "quantity": f"Quantity: {int(r['quantity'])}",
        "unit_price": f"Unit price: {r['unit_price']:.2f} USD",
        "sales": f"Line sales: {r['sales']:.2f} USD",
        "order_item_total": f"Line total: {r['order_item_total']:.2f} USD",
        "discount_rate": f"Discount: {r['discount_rate'] * 100:.0f}%",
    }[feature]


def shap_by_feature(pipe, X):
    """SHAP chính xác từ XGBoost (pred_contribs, tương đương TreeExplainer), gộp các cột one-hot về 18 biến gốc."""
    pre, clf = pipe[:-1], pipe.steps[-1][1]
    names = list(pre.get_feature_names_out())
    contrib = clf.get_booster().predict(xgb.DMatrix(pre.transform(X[FEATURES]), feature_names=names),
                                        pred_contribs=True)[:, :-1]
    owner = []
    for n in names:
        kind, rest = n.split("__", 1)
        owner.append(rest if kind == "num" else next(c for c in CAT_COLS if rest.startswith(c + "_")))
    out = pd.DataFrame(contrib, columns=names).T.groupby(owner).sum().T.reindex(columns=FEATURES, fill_value=0)
    out.index = X.index
    return out


def top_reasons(row, shap_row, k=3):
    """Up to k factors that INCREASE the risk (largest positive SHAP), with English labels."""
    pos = shap_row[shap_row > 0].sort_values(ascending=False).head(k)
    return [{"rank": i + 1, "feature": f, "label": reason_label(f, row), "shap": round(float(v), 3)}
            for i, (f, v) in enumerate(pos.items())]
