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
from overview import INK, MUTED, card_header, kpi_card, vn_int, vn_pct

PER_MIN = 10_000_000 / 26 / 8 / 60      # đồng mỗi phút của nhân viên chăm sóc khách hàng
RETURN_FACTOR = 1.07 * 1.5               # phụ phí xăng dầu x (cước đi + phí hoàn 50%)
CHURN = 0.23                             # khách có thể bỏ đi sau giao hàng không ổn định (Milieu 2025)
SCEN = ["Thấp", "Cơ sở", "Cao"]
ASSUME = pd.DataFrame({
    "Kịch bản": SCEN,
    "Tỷ lệ trễ hiện tại": [0.03, 0.08, 0.10],
    "Cách hẹn mới": ["First Class hẹn 2 ngày", "Hẹn theo P50", "Hẹn theo P80"],
    "Phút xử lý 1 đơn trễ": [5, 10, 15],
    "Tỷ lệ đơn trễ bị hoàn": [0.15, 0.175, 0.20],
    "Cước giao đi (đ)": [22_000, 30_000, 40_000],
})


def calc(row, n, drop, aov):
    cost = row["Phút xử lý 1 đơn trễ"] * PER_MIN + row["Tỷ lệ đơn trễ bị hoàn"] * row["Cước giao đi (đ)"] * RETURN_FACTOR
    before = n * row["Tỷ lệ trễ hiện tại"]
    after = before * (1 - drop)
    saved = (before - after) * cost
    kept = (before - after) * CHURN * aov
    return {"cost": cost, "before": before, "after": after, "drop": drop, "saved": saved, "kept": kept, "total": saved + kept}


def page_savings():
    st.markdown(f"<style>.sv-note {{ font-size:13px; color:{MUTED}; }} .sv-note b {{ color:{INK}; }}</style>",
                unsafe_allow_html=True)
    bt = pd.DataFrame(core.load_evaluation()["backtest"])
    allr = bt[bt["phuong_thuc"] == "Tất cả"].set_index("cach_hen")
    old = allr.loc["Như cũ", "ty_le_tre_so_voi_hen"]
    ways = [w for w in allr.index if w != "Như cũ"]

    st.markdown('<div class="ov-title">Ước tính tiết kiệm</div>', unsafe_allow_html=True)
    st.markdown('<div class="ov-sub">Một shop, ba kịch bản · chưa trừ chi phí triển khai</div>', unsafe_allow_html=True)
    st.write("")

    with st.container(key="card_sv_in"):
        c1, c2 = st.columns([1, 3], vertical_alignment="bottom")
        n = c1.number_input("Số đơn mỗi tháng", 100, 1_000_000, 1000, step=100)
        with c2:
            with st.expander("Tinh chỉnh giả định"):
                a = st.data_editor(ASSUME, hide_index=True, width="stretch", disabled=["Kịch bản"], column_config={
                    "Tỷ lệ trễ hiện tại": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01, format="percent"),
                    "Cách hẹn mới": st.column_config.SelectboxColumn(options=ways),
                    "Phút xử lý 1 đơn trễ": st.column_config.NumberColumn(min_value=0, step=1),
                    "Tỷ lệ đơn trễ bị hoàn": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.025, format="percent"),
                    "Cước giao đi (đ)": st.column_config.NumberColumn(min_value=0, step=1000, format="localized")})
                aov = st.number_input("Giá trị trung bình mỗi sản phẩm (đ)", 0, 100_000_000, 105_000, step=5_000)

    res = []
    for _, r in a.fillna(0).iterrows():
        drop = 1 - allr.loc[r["Cách hẹn mới"], "ty_le_tre_so_voi_hen"] / old if old else 0
        res.append(calc(r, n, drop, aov))

    cols = st.columns(3, gap="medium")
    for i, (col, r) in enumerate(zip(cols, res)):
        with col:
            kpi_card(f"card_sv{i}", "money", f"Kịch bản {SCEN[i].lower()}",
                     vn_int(r["total"]) + "<small>₫/tháng</small>",
                     f"đơn trễ {vn_int(r['before'])} → {vn_int(r['after'])} (−{vn_pct(r['drop'], 0)})",
                     i != 1, None, None)

    with st.container(key="card_sv_tbl"):
        card_header("Chi tiết")
        rows = {
            "Cách hẹn mới": [x for x in a["Cách hẹn mới"]],
            "Chi phí mỗi đơn trễ": [vn_int(r["cost"]) + " ₫" for r in res],
            "Tiết kiệm chi phí giao trễ": [vn_int(r["saved"]) + " ₫" for r in res],
            "Doanh thu giữ lại": [vn_int(r["kept"]) + " ₫" for r in res],
            "Lợi ích mỗi tháng": [vn_int(r["total"]) + " ₫" for r in res],
        }
        st.dataframe(pd.DataFrame(rows, index=SCEN).T, width="stretch")
    st.markdown('<div class="sv-note">Dữ liệu DataCo, không phải dữ liệu Việt Nam. Phút xử lý, tỷ lệ hoàn và cước là ước tính của nhóm; '
                'doanh thu giữ lại chỉ tính một lần mua lại bị mất (23% khách) nên là mức tối thiểu.</div>', unsafe_allow_html=True)
