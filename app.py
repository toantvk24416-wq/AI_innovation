import json
from datetime import datetime, timedelta
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from overview import ACCENT, F, INK, apply_style, card_header, clean, page_overview, sidebar_nav, fmt_pct
from ranking import page_ranking
from chatbot import page_chatbot
from assistant import page_assistant
from promise import page_promise
from savings import page_savings
from ontime import core

st.set_page_config(page_title="DataCo Delivery Risk", page_icon="🚚", layout="wide")

BASE = Path(__file__).parent
CAT_COLS = ["shipping_mode", "market", "order_region", "order_country",
            "category_name", "department_name", "customer_segment", "payment_type"]
NUM_COLS = ["scheduled_shipping_days", "quantity", "unit_price", "sales", "discount_rate",
            "order_item_total", "order_month", "order_day_of_week", "order_is_weekend", "order_hour"]
FEATURES = CAT_COLS + NUM_COLS
TIER_ORDER = ["High", "Medium", "Low", "Insufficient data"]
TIER_COLOR = ["#d62728", "#ff9f1c", "#2ca02c", "#9e9e9e"]
TIER_VI = {t: t for t in TIER_ORDER}   # tier names are already English


@st.cache_resource
def load_model():
    return joblib.load(BASE / "best_pipe.joblib")


@st.cache_data
def load_options():
    return json.loads((BASE / "options.json").read_text(encoding="utf-8"))


@st.cache_data
def load_routes():
    df = pd.read_csv(BASE / "route_risk.csv")
    df["risk_tier_vi"] = df["risk_tier"].map(TIER_VI).fillna(df["risk_tier"])
    return df


def check_files():
    missing = [f for f in ["best_pipe.joblib", "options.json", "route_risk.csv"] if not (BASE / f).exists()]
    if missing:
        st.error("Missing files: " + ", ".join(missing) + ". Put them in the same folder as app.py.")
        st.stop()


def level(p):
    if p >= 0.6:
        return "HIGH", "🔴"
    if p >= 0.3:
        return "MEDIUM", "🟠"
    return "LOW", "🟢"


# ---------------------------------------------------------------- Trang 1 (bản cũ, đã thay bằng ranking.py)
def page_routes():
    st.title("Delivery risk ranking by route")
    st.caption("Route = order region × shipping mode. Data comes from the gold.route_risk_expected table.")
    df = load_routes()

    c1, c2, c3 = st.columns(3)
    tiers = c1.multiselect("Risk level", TIER_ORDER, default=TIER_ORDER, format_func=lambda x: TIER_VI.get(x, x))
    modes = c2.multiselect("Shipping mode", sorted(df["shipping_mode"].unique()),
                           default=sorted(df["shipping_mode"].unique()))
    regions = c3.multiselect("Region", sorted(df["order_region"].unique()), default=sorted(df["order_region"].unique()))
    f = df[df["risk_tier"].isin(tiers) & df["shipping_mode"].isin(modes) & df["order_region"].isin(regions)]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Routes", len(f))
    k2.metric("High-risk routes", int((f["risk_tier"] == "High").sum()))
    k3.metric("Actual late rate (avg)", f"{f['actual_late_rate'].mean():.1%}" if len(f) else "-")
    k4.metric("Expected revenue at risk", f"{f['expected_revenue_at_risk'].sum():,.0f}")

    if f.empty:
        st.info("No routes match the filters.")
        return

    ranked = f.dropna(subset=["risk_score"]).sort_values("risk_score", ascending=False)
    if ranked.empty:
        st.info("The selected routes do not have enough data to score.")
        top_n, top = 0, ranked
    elif len(ranked) <= 5:
        top_n, top = len(ranked), ranked
    else:
        top_n = st.slider("Routes shown in the chart", 5, min(30, len(ranked)), min(15, len(ranked)))
        top = ranked.head(top_n)
    bar = None if top.empty else (alt.Chart(top).mark_bar().encode(
        x=alt.X("risk_score:Q", title="Risk score"),
        y=alt.Y("route_name:N", sort="-x", title=None),
        color=alt.Color("risk_tier:N", scale=alt.Scale(domain=TIER_ORDER, range=TIER_COLOR), title="Tier"),
        tooltip=["route_name", "risk_rank", "risk_score", "actual_late_rate", "predicted_late_rate",
                 "expected_revenue_at_risk"]).properties(height=28 * top_n + 30))
    if top_n:
        st.subheader(f"Top {top_n} riskiest routes")
        st.altair_chart(bar, use_container_width=True)

    st.subheader("Actual vs predicted late rate")
    sc = (alt.Chart(f).mark_circle(size=90).encode(
        x=alt.X("actual_late_rate:Q", title="Actual late", axis=alt.Axis(format="%")),
        y=alt.Y("predicted_late_rate:Q", title="Predicted late", axis=alt.Axis(format="%")),
        color=alt.Color("risk_tier:N", scale=alt.Scale(domain=TIER_ORDER, range=TIER_COLOR), title="Tier"),
        tooltip=["route_name", "actual_late_rate", "predicted_late_rate", "n_orders"]))
    st.altair_chart(sc, use_container_width=True)

    st.subheader("Detail table")
    f = f.assign(actual_late_rate=f["actual_late_rate"] * 100, predicted_late_rate=f["predicted_late_rate"] * 100)
    show = f.sort_values("risk_rank")[["risk_rank", "route_name", "risk_tier_vi", "n_orders", "actual_late_rate",
                                       "predicted_late_rate", "expected_revenue_at_risk", "risk_score"]]
    st.dataframe(show, use_container_width=True, hide_index=True, column_config={
        "risk_rank": "Rank", "route_name": "Route", "risk_tier_vi": "Risk tier", "n_orders": "Orders",
        "actual_late_rate": st.column_config.NumberColumn("Actual late (%)", format="%.1f"),
        "predicted_late_rate": st.column_config.NumberColumn("Predicted late (%)", format="%.1f"),
        "expected_revenue_at_risk": st.column_config.NumberColumn("Expected revenue at risk", format="%.0f"),
        "risk_score": st.column_config.NumberColumn("Risk score", format="%.3f")})
    st.download_button("Download CSV", f.to_csv(index=False).encode("utf-8-sig"), "route_risk.csv", "text/csv")


# ---------------------------------------------------------------- Trang 2
LEVEL_COLOR = {"HIGH": ACCENT, "MEDIUM": "#F5A3A1", "LOW": "#CFCFCF"}

PREDICT_CSS = f"""
<style>
.pd-group {{ font-size:15px; font-weight:600; color:{INK}; margin-bottom:6px; }}
.pd-calc  {{ font-size:14px; color:{INK}; margin-top:6px; }}
.pd-dot   {{ display:inline-block; width:14px; height:14px; border-radius:50%; margin-right:10px; vertical-align:middle; }}
.pd-gauge {{ position:relative; height:12px; border-radius:999px; background:#F3F3F3; margin:18px 0 6px 0; }}
.pd-gauge .fill {{ position:absolute; left:0; top:0; bottom:0; border-radius:999px; background:{ACCENT}; }}
.pd-gauge .mark {{ position:absolute; top:-4px; bottom:-4px; width:2px; background:#FFFFFF; }}
.pd-scale {{ position:relative; height:16px; font-size:12px; color:{INK}; }}
.pd-scale span {{ position:absolute; transform:translateX(-50%); }}
.pd-note  {{ font-size:13.5px; color:{INK}; line-height:1.55; margin-top:6px; }}
[class*="st-key-card_in"] {{ min-height: 400px; }}
.st-key-card_kpi_p, .st-key-card_kpi_lv, .st-key-card_kpi_money {{ min-height: 130px !important; }}
.st-key-card_gauge, .st-key-card_whatif {{ min-height: 250px; }}
.st-key-card_reasons, .st-key-card_promise {{ padding-bottom: 26px; }}
.st-key-predict_btn button {{ background:{ACCENT}; border-color:{ACCENT}; color:#FFFFFF; }}
.st-key-predict_btn button:hover {{ background:#D93C3C; border-color:#D93C3C; color:#FFFFFF; }}
</style>
"""


def fmt_usd(x):
    return f"{x:,.2f} USD"


def result_card(key, label, value_html, pill=None):
    with st.container(key=key):
        st.markdown(f'<div class="kpi-label">{label}</div><div class="kpi-value">{value_html}</div>'
                    + (f'<span class="kpi-pill">{pill}</span>' if pill else ""), unsafe_allow_html=True)


def gauge_html(p):
    marks = "".join(f'<div class="mark" style="left:{x}%"></div>' for x in (30, 60))
    scale = "".join(f'<span style="left:{x}%">{t}</span>' for x, t in
                    ((15, "Low"), (45, "Medium"), (80, "High")))
    return (f'<div class="pd-gauge"><div class="fill" style="width:{p * 100:.1f}%"></div>{marks}</div>'
            f'<div class="pd-scale">{scale}</div>')


def whatif_chart(wi, mode):
    d = pd.DataFrame({
        "Mode": wi["shipping_mode"],
        "Expected days": wi["scheduled_shipping_days"],
        "Late probability": wi["p_late"],   # không làm tròn trước, để nhãn khớp với câu gợi ý bên cạnh
        "Expected revenue at risk (USD)": wi["Expected revenue at risk"].round(2),
    })
    d["Mode and promise"] = d["Mode"] + " - " + d["Expected days"].map(lambda n: f"{n} day" + ("" if n == 1 else "s"))
    d["Selected"] = d["Mode"] == mode
    d["Label"] = d["Late probability"].map(fmt_pct)
    order = d.sort_values("Late probability")["Mode and promise"].tolist()
    base = alt.Chart(d).encode(y=alt.Y(**F("Mode and promise", "N"), sort=order, title=None,
                                       axis=alt.Axis(labelLimit=260, labelFontSize=12.5, labelColor=INK, labelPadding=8)))
    track = base.mark_bar(cornerRadius=9, size=20, color="#F3F3F3").encode(
        x=alt.X(field="nen", type="quantitative", axis=None, scale=alt.Scale(domain=[0, 1]))).transform_calculate(nen="1")
    bars = base.mark_bar(cornerRadius=9, size=20).encode(
        x=alt.X(**F("Late probability", "Q"), axis=None, scale=alt.Scale(domain=[0, 1])),
        color=alt.condition("datum['Selected']", alt.value(ACCENT), alt.value("#F2A6A4")),
        tooltip=[alt.Tooltip(**F("Mode", "N")),
                 alt.Tooltip(**F("Expected days", "Q")),
                 alt.Tooltip(**F("Late probability", "Q"), format=".1%"),
                 alt.Tooltip(**F("Expected revenue at risk (USD)", "Q"), format=",.2f")])
    # nhãn nằm trong cột nếu cột đủ dài, ngược lại nằm ngay sau cột
    inside = base.transform_filter("datum['Late probability'] >= 0.12").mark_text(
        align="right", dx=-8, color="#FFFFFF", fontWeight=600, font="Inter", fontSize=12).encode(
        x=alt.X(**F("Late probability", "Q")), text=alt.Text(**F("Label", "N")))
    outside = base.transform_filter("datum['Late probability'] < 0.12").mark_text(
        align="left", dx=6, color=INK, fontWeight=600, font="Inter", fontSize=12).encode(
        x=alt.X(**F("Late probability", "Q")), text=alt.Text(**F("Label", "N")))
    return clean(alt.layer(track, bars, inside, outside), 230)


def page_predict():
    st.markdown(PREDICT_CSS, unsafe_allow_html=True)
    st.markdown('<div class="ov-title">New order prediction</div>', unsafe_allow_html=True)
    st.write("")
    model = load_model()

    col1, col2, col3 = st.columns(3)
    with col1, st.container(key="card_in1"):
        st.markdown('<div class="pd-group">Delivery</div>', unsafe_allow_html=True)
        mode = st.selectbox("Shipping mode", opt["shipping_mode"],
                            index=opt["shipping_mode"].index("Standard Class") if "Standard Class" in opt["shipping_mode"] else 0)
        days = opt["mode_days"][mode]
        st.text_input("Expected delivery days", f"{days} days", disabled=True)
        region = st.selectbox("Order region", opt["order_region"])
        market = opt["region_market"].get(region, opt["market"][0])
        st.text_input("Market", market, disabled=True)
        country = st.selectbox("Country", opt["region_country"].get(region, opt["order_country"]))
    with col2, st.container(key="card_in2"):
        st.markdown('<div class="pd-group">Product</div>', unsafe_allow_html=True)
        category = st.selectbox("Category", opt["category_name"])
        dept = opt["category_department"].get(category, opt["department_name"][0])
        st.text_input("Department", dept, disabled=True)
        segment = st.selectbox("Customer segment", opt["customer_segment"])
        payment = st.selectbox("Payment type", opt["payment_type"])
    with col3, st.container(key="card_in3"):
        st.markdown('<div class="pd-group">Value and timing</div>', unsafe_allow_html=True)
        qty = st.number_input("Quantity", 1, 5, 1)
        price = st.number_input("Unit price (USD)", 9.99, 1999.99, 59.99, step=1.0)
        disc = st.slider("Discount rate", 0.0, 0.25, 0.0, 0.01, format="%.2f")
        d = st.date_input("Order date", datetime.now().date(), format="YYYY/MM/DD")
        t = st.time_input("Order time", datetime.now().time().replace(second=0, microsecond=0))

    sales = qty * price
    total = sales * (1 - disc)
    dow = (d.weekday() + 1) % 7 + 1  # 1 = Chủ nhật ... 7 = Thứ bảy (giống dữ liệu huấn luyện)
    row = {"shipping_mode": mode, "market": market, "order_region": region, "order_country": country,
           "category_name": category, "department_name": dept, "customer_segment": segment, "payment_type": payment,
           "scheduled_shipping_days": days, "quantity": qty, "unit_price": price, "sales": sales,
           "discount_rate": disc, "order_item_total": total, "order_month": d.month,
           "order_day_of_week": dow, "order_is_weekend": int(dow in (1, 7)), "order_hour": t.hour}

    left, right = st.columns([3, 1], vertical_alignment="center")
    left.markdown(f'<div class="pd-calc">Sales <b>{fmt_usd(sales)}</b> · after discount <b>{fmt_usd(total)}</b></div>',
                  unsafe_allow_html=True)
    if right.button("Predict", type="primary", width="stretch", key="predict_btn"):
        st.session_state["pred_row"] = row
    if st.session_state.get("pred_row") != row:
        return

    p = float(model.predict_proba(pd.DataFrame([row])[FEATURES])[0, 1])
    name, _ = level(p)
    st.write("")
    a, b, c = st.columns(3)
    with a:
        result_card("card_kpi_p", "Late delivery probability", fmt_pct(p))
    with b:
        result_card("card_kpi_lv", "Risk level",
                    f'<span class="pd-dot" style="background:{LEVEL_COLOR[name]}"></span>{name.capitalize()}')
    with c:
        result_card("card_kpi_money", "Expected revenue at risk", fmt_usd(p * total).replace(" USD", "<small>USD</small>"))

    rows = [{**row, "shipping_mode": m, "scheduled_shipping_days": opt["mode_days"][m]} for m in opt["shipping_mode"]]
    wi = pd.DataFrame(rows)
    wi["p_late"] = model.predict_proba(wi[FEATURES])[:, 1]
    wi["Expected revenue at risk"] = wi["p_late"] * wi["order_item_total"]

    st.write("")
    g, w = st.columns([2, 3])
    with g, st.container(key="card_gauge"):
        card_header("Position on the risk scale")
        st.markdown(gauge_html(p), unsafe_allow_html=True)
        best = wi.loc[wi["p_late"].idxmin()]
        tip = ("Already the lowest-risk shipping mode." if best["shipping_mode"] == mode else
               f'Switch to <b>{best["shipping_mode"]}</b>: risk drops to <b>{fmt_pct(best["p_late"])}</b>.')
        st.markdown(f'<div class="pd-note" style="margin-top:22px">{tip}</div>'
                    '<div class="pd-note">For reference only, not a delivery guarantee.</div>', unsafe_allow_html=True)
    with w, st.container(key="card_whatif"):
        card_header("If the shipping mode changes")
        st.altair_chart(whatif_chart(wi, mode), width="stretch")

    # Ngày hẹn đề xuất (5.1), lý do SHAP (5.3) và việc nên làm theo bảng luật (5.5)
    pr, ex = core.recommend_promise(row), core.explain_order(row)
    ordered = datetime.combine(d, t)
    due_old, due_new = ordered + timedelta(days=days), ordered + timedelta(days=pr["promise_days"])
    acts = core.actions_for({**row, "risk_tier": core.risk_tier(p), "order_value": total})
    st.write("")
    x, y = st.columns([2, 3])
    with x, st.container(key="card_promise"):
        card_header("Suggested promise date", pr["source"])
        st.markdown(
            f'<div class="kpi-value">{due_new:%d %b %Y}</div>'
            f'<span class="kpi-pill">promise {pr["promise_days"]} days instead of {days} ({due_old:%d %b})</span>'
            f'<div class="pd-note" style="margin-top:14px">P50 {pr["p50"]:g} · P80 {pr["p80"]:g} · P90 {pr["p90"]:g} days. '
            f'About 8 in 10 similar orders arrive by the P80 date.</div>',
            unsafe_allow_html=True)
    with y, st.container(key="card_reasons"):
        card_header("Risk factors and what to do")
        html = "".join(f'<div class="pd-note">• {r["label"]} <span style="color:#8C8C8C">(SHAP +{r["shap"]:.2f})</span></div>'
                       for r in ex["reasons"]) or '<div class="pd-note">No factor increases the risk.</div>'
        html += "".join(f'<div class="pd-note"><b>{a["code"]}</b>: {a["action"]}' + (f' · <i>{a["owner"]}</i>' if a["owner"] not in ("", "-") else "") + '</div>'
                        for a in acts)
        st.markdown(html, unsafe_allow_html=True)


def main():
    global opt
    check_files()
    opt = load_options()
    apply_style()
    page = sidebar_nav({
        "Overview": (":material/dashboard:", page_overview),
        "Route ranking": (":material/route:", page_ranking),
        "New order prediction": (":material/online_prediction:", page_predict),
        "Order assistant": (":material/support_agent:", page_assistant),
        "Promise and evaluation": (":material/event_available:", page_promise),
        "Savings estimate": (":material/savings:", page_savings),
        "Data assistant": (":material/smart_toy:", page_chatbot),
    })
    page()


try:
    from streamlit.runtime import exists as _in_streamlit
except Exception:  # phiên bản streamlit cũ
    _in_streamlit = lambda: True

if _in_streamlit():
    main()
elif __name__ == "__main__":
    # Chạy bằng nút Run của VS Code (python app.py) thì tự gọi streamlit run
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())])
            
