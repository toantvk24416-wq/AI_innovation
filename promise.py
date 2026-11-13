"""Trang Ngày hẹn và đánh giá: kết quả Mục 5.1 (ngày hẹn P50/P80/P90), backtest 3 cách hẹn (PCCV 6.1),
Precision@K của hàng đợi (PCCV 6.2) và mức ảnh hưởng SHAP của các biến (Mục 5.3).

Mọi số trên trang đọc từ data/evaluation.json, do scripts/build_artifacts.py tính trên tập kiểm tra
(đơn đặt từ 10/07/2017), không tính lại khi mở app.
"""
import altair as alt
import pandas as pd
import streamlit as st

from ontime import core
from overview import ACCENT, INK, MODE_ORDER, MUTED, F, card_header, clean, kpi_card, fmt_int, fmt_pct

PAGE_CSS = f"""
<style>
.st-key-card_bt, .st-key-card_pk {{ min-height: 340px; }}
.st-key-card_q, .st-key-card_shap {{ min-height: 330px; }}
.pr-note {{ font-size:13px; color:{MUTED}; line-height:1.55; margin-top:6px; }}
.pr-note b {{ color:{INK}; }}
</style>
"""
WAYS = ["Current", "First Class 2 days", "Promise P50", "Promise P80", "Promise P90"]


def backtest_chart(bt):
    d = bt[bt["mode"] == "All"][["way", "late_vs_promise", "avg_extra_days"]].rename(columns={
        "way": "Promise method", "late_vs_promise": "Late vs promise", "avg_extra_days": "Extra days promised"})
    d["Label"] = d["Late vs promise"].map(fmt_pct) + " · +" + d["Extra days promised"].map(lambda x: f"{x:.2f}") + " days"
    d["In use"] = d["Promise method"] == "Promise P80"
    base = alt.Chart(d).encode(y=alt.Y(**F("Promise method", "N"), sort=WAYS, title=None,
                                       axis=alt.Axis(labelColor=INK, labelFontSize=12.5, labelLimit=220)))
    bars = base.mark_bar(cornerRadius=8, size=20).encode(
        x=alt.X(**F("Late vs promise", "Q"), axis=None, scale=alt.Scale(domain=[0, 1.7])),   # chừa chỗ cho nhãn
        color=alt.condition("datum['In use']", alt.value(ACCENT), alt.value("#F2A6A4")),
        tooltip=[alt.Tooltip(**F("Promise method", "N")), alt.Tooltip(**F("Late vs promise", "Q"), format=".1%"),
                 alt.Tooltip(**F("Extra days promised", "Q"), format=".2f")])
    text = base.mark_text(align="left", dx=6, color=INK, fontSize=12, font="Inter", fontWeight=600).encode(
        x=alt.X(**F("Late vs promise", "Q")), text=alt.Text(**F("Label", "N")))
    return clean(alt.layer(bars, text), 260)


def shap_chart(sg):
    d = pd.DataFrame({"Feature": list(sg), "Impact": list(sg.values())}).head(8)
    d["Highest"] = d["Impact"] == d["Impact"].max()
    base = alt.Chart(d).encode(y=alt.Y(**F("Feature", "N"), sort=None, title=None,
                                       axis=alt.Axis(labelColor=INK, labelFontSize=12)))
    bars = base.mark_bar(cornerRadius=8, size=16).encode(
        x=alt.X(**F("Impact", "Q"), axis=None),
        color=alt.condition("datum['Highest']", alt.value(ACCENT), alt.value("#F2A6A4")),
        tooltip=[alt.Tooltip(**F("Feature", "N")), alt.Tooltip(**F("Impact", "Q"), format=".3f")])
    text = base.mark_text(align="left", dx=6, color=INK, fontSize=12, font="Inter").encode(
        x=alt.X(**F("Impact", "Q")), text=alt.Text(**F("Impact", "Q"), format=".2f"))
    return clean(alt.layer(bars, text), 280)


def page_promise():
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    ev = core.load_evaluation()
    bt = pd.DataFrame(ev["backtest"])
    pe = ev["promise"]
    allr = bt[bt["mode"] == "All"].set_index("way")
    pk = ev["precision_at_k"]

    st.markdown('<div class="ov-title">Promised date and evaluation</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ov-sub">Test set: {fmt_int(ev["n_test_orders"])} orders placed from 10 Jul 2017 · '
                f'ROC-AUC of the alert model {ev["auc_test"]:.4f}' + '</div>', unsafe_allow_html=True)
    st.write("")

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        kpi_card("card_pr1", "late", "Late vs current promise", fmt_pct(allr.loc["Current", "late_vs_promise"]),
                 "current promise", True, None, None)
    with c2:
        kpi_card("card_pr2", "box", "If promised at P80", fmt_pct(allr.loc["Promise P80", "late_vs_promise"]),
                 "new promised date", False, None, None)
    with c3:
        extra = allr.loc["Promise P80", "avg_extra_days"]
        kpi_card("card_pr3", "delay", "Cost: extra promised days", f"+{extra:.2f}" + "<small>days</small>",
                 "average per order", True, None, None)
    with c4:
        p10 = pk["Excluding First Class (handled by rule)"]["Top 10 per day, by late probability"]
        kpi_card("card_pr4", "money", "Precision@10 per day", fmt_pct(p10, 0),
                 "excluding First Class", False, None, None)

    # ---------- Backtest 3 cách hẹn (6.1)
    a, b = st.columns([1.25, 1], gap="medium")
    with a, st.container(key="card_bt"):
        card_header("Promise methods compared", "Late orders · extra promised days")
        st.altair_chart(backtest_chart(bt), width="stretch")
        st.markdown('<div class="pr-note">P80: almost no late orders but longer promises · P50: shorter promises but about 1 in 3 orders is still late.</div>',
                    unsafe_allow_html=True)
    with b, st.container(key="card_pk"):
        card_header("By shipping mode")
        mode = st.segmented_control("Mode", MODE_ORDER, default="Second Class", key="pr_mode",
                                    label_visibility="collapsed") or "Second Class"
        t = bt[bt["mode"] == mode].set_index("way").reindex(WAYS)
        st.dataframe(t[["late_vs_promise", "avg_promise_days", "avg_extra_days"]].reset_index(), hide_index=True,
                     width="stretch", column_config={
                         "way": "Method",
                         "late_vs_promise": st.column_config.ProgressColumn("Late vs promise", format="percent",
                                                                                 min_value=0, max_value=1),
                         "avg_promise_days": st.column_config.NumberColumn("Avg days", format="%.2f"),
                         "avg_extra_days": st.column_config.NumberColumn("Extra days", format="%+.2f")})
        st.caption(f"{int(t['n_orders'].iloc[0]):,} orders")

    # ---------- Mô hình phân vị so với bảng phân vị (5.1)
    x, y = st.columns([1, 1], gap="medium")
    with x, st.container(key="card_q"):
        card_header("Model vs quantile table", "Lower pinball loss is better")
        rows = []
        for name, lab in (("model", "Quantile regression model"), ("table", "Route x mode quantile table")):
            rows.append({"Method": lab, **{f"Pinball {k.upper()}": v for k, v in pe["pinball"][name].items()},
                         "P80 coverage": pe["coverage_p80"][name]["All"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"P80 coverage": st.column_config.NumberColumn(format="percent")})
        chosen = "the quantile table" if pe["chosen"] == "table" else "the quantile model"
        st.markdown(f'<div class="pr-note">The app uses <b>{chosen}</b> to suggest promised dates (simpler, lower error).</div>',
                    unsafe_allow_html=True)
    with y, st.container(key="card_shap"):
        card_header("What the alert model relies on", "Mean |SHAP|")
        st.altair_chart(shap_chart(ev["shap_global"]), width="stretch")

    # ---------- Precision@K (6.2) và bảng P50/P80/P90 gộp trong một thẻ có tab
    with st.container(key="card_tbl"):
        t1, t2 = st.tabs(["Queue Precision@K", "P50 · P80 · P90 by route"])
        with t1:
            st.caption("If staff handle the first K orders of the queue each day, what share is really late. money = ranked by expected revenue at risk, prob. = ranked by late probability")
            short = {" per day, by expected revenue at risk": " · money",
                     " per day, by late probability": " · prob.",
                     "Baseline late rate (random pick)": "Random pick"}

            def short_name(c):
                for a, b in short.items():
                    c = c.replace(a, b)
                return c
            pkd = pd.DataFrame(pk).T.rename(columns=short_name)
            st.dataframe(pkd, width="stretch", column_config={
                c: st.column_config.NumberColumn(c, format="percent") for c in pkd.columns if c != "Days"})
        with t2:
            t = core.load_promise_table()
            modes = st.multiselect("Mode", MODE_ORDER, placeholder="All modes", key="pr_tbl_modes",
                                   label_visibility="collapsed")
            if modes:
                t = t[t["shipping_mode"].isin(modes)]
            t = t.assign(hen_p80=t["p80"].map(lambda v: int(-(-v // 1))))
            st.dataframe(t[["route_name", "n_lines", "p50", "p80", "p90", "hen_p80"]], hide_index=True, width="stretch",
                         height=340, column_config={
                             "route_name": "Route", "n_lines": st.column_config.NumberColumn("Order lines", format="localized"),
                             "p50": "P50", "p80": "P80", "p90": "P90", "hen_p80": "Suggested promise (days)"})
