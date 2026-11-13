"""Trang Ước tính tiết kiệm (PCCV 9.4, cùng cách tính với mục 7.1 và 7.4 của báo cáo).

Chi phí giao trễ mỗi tháng:  C = N x r x (c_CSKH + h x c_hoàn)
  N: số đơn mỗi tháng · r: tỷ lệ đơn trễ hiện tại
  c_CSKH = số phút xử lý x 801,28đ/phút (lương 10 triệu / 26 ngày / 8 giờ / 60 phút)
  c_hoàn = cước đi x 1,07 (phụ phí xăng dầu) x 1,5 (cước đi + phí hoàn 50%)
Khi áp dụng cách hẹn mới, số đơn trễ giảm đúng bằng tỷ lệ giảm đo trong backtest (data/evaluation.json), nên
% tiết kiệm chi phí giao trễ = % giảm đơn trễ. Doanh thu giữ lại = số đơn trễ tránh được x 23% khách bỏ đi x giá trị TB.
"""
import pandas as pd
import streamlit as st

from ontime import core
from views.overview import INK, MUTED, card_header, kpi_card, fmt_int, fmt_pct

PER_MIN = 10_000_000 / 26 / 8 / 60      # đồng mỗi phút của nhân viên chăm sóc khách hàng
RETURN_FACTOR = 1.07 * 1.5               # phụ phí xăng dầu x (cước đi + phí hoàn 50%)
CHURN = 0.23                             # khách có thể bỏ đi sau giao hàng không ổn định (Milieu 2025)
SCEN = ["Low", "Base", "High"]
ASSUME = pd.DataFrame({
    "Scenario": SCEN,
    "Current late rate": [0.03, 0.08, 0.10],
    "New promise method": ["First Class 2 days", "Promise P50", "Promise P80"],
    "Minutes to handle a late order": [5, 10, 15],
    "Share of late orders returned": [0.15, 0.175, 0.20],
    "Outbound shipping fee (VND)": [22_000, 30_000, 40_000],
})


def calc(row, n, drop, aov):
    cost = row["Minutes to handle a late order"] * PER_MIN + row["Share of late orders returned"] * row["Outbound shipping fee (VND)"] * RETURN_FACTOR
    before = n * row["Current late rate"]
    after = before * (1 - drop)
    saved = (before - after) * cost
    kept = (before - after) * CHURN * aov
    return {"cost": cost, "before": before, "after": after, "drop": drop, "saved": saved, "kept": kept, "total": saved + kept}


def page_savings():
    st.markdown(f"<style>.sv-note {{ font-size:13px; color:{MUTED}; }} .sv-note b {{ color:{INK}; }}</style>",
                unsafe_allow_html=True)
    bt = pd.DataFrame(core.load_evaluation()["backtest"])
    allr = bt[bt["mode"] == "All"].set_index("way")
    old = allr.loc["Current", "late_vs_promise"]
    ways = [w for w in allr.index if w != "Current"]

    st.markdown('<div class="ov-title">Savings estimate</div>', unsafe_allow_html=True)
    st.markdown('<div class="ov-sub">One shop, three scenarios · before implementation cost</div>', unsafe_allow_html=True)
    st.write("")

    with st.container(key="card_sv_in"):
        c1, c2 = st.columns([1, 3], vertical_alignment="bottom")
        n = c1.number_input("Orders per month", 100, 1_000_000, 1000, step=100)
        with c2:
            with st.expander("Adjust assumptions"):
                a = st.data_editor(ASSUME, hide_index=True, width="stretch", disabled=["Scenario"], column_config={
                    "Current late rate": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01, format="percent"),
                    "New promise method": st.column_config.SelectboxColumn(options=ways),
                    "Minutes to handle a late order": st.column_config.NumberColumn(min_value=0, step=1),
                    "Share of late orders returned": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.025, format="percent"),
                    "Outbound shipping fee (VND)": st.column_config.NumberColumn(min_value=0, step=1000, format="localized")})
                aov = st.number_input("Average value per item (VND)", 0, 100_000_000, 105_000, step=5_000)

    res = []
    for _, r in a.fillna(0).iterrows():
        drop = 1 - allr.loc[r["New promise method"], "late_vs_promise"] / old if old else 0
        res.append(calc(r, n, drop, aov))

    cols = st.columns(3, gap="medium")
    for i, (col, r) in enumerate(zip(cols, res)):
        with col:
            kpi_card(f"card_sv{i}", "money", f"{SCEN[i]} scenario",
                     fmt_int(r["total"]) + "<small>VND/month</small>",
                     f"late orders {fmt_int(r['before'])} → {fmt_int(r['after'])} (−{fmt_pct(r['drop'], 0)})",
                     i != 1, None, None)

    with st.container(key="card_sv_tbl"):
        card_header("Details")
        rows = {
            "New promise method": [x for x in a["New promise method"]],
            "Cost per late order": [fmt_int(r["cost"]) + " VND" for r in res],
            "Savings on late-delivery cost": [fmt_int(r["saved"]) + " VND" for r in res],
            "Revenue retained": [fmt_int(r["kept"]) + " VND" for r in res],
            "Benefit per month": [fmt_int(r["total"]) + " VND" for r in res],
        }
        st.dataframe(pd.DataFrame(rows, index=SCEN).T, width="stretch")
    st.markdown('<div class="sv-note">DataCo data, not Vietnamese data. Handling minutes, return rate and shipping fees are team estimates; '
                'retained revenue counts only one lost repurchase (23% of customers), so it is a minimum.</div>', unsafe_allow_html=True)
