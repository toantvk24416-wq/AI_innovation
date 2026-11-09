"""Trang Ước tính tiết kiệm (PCCV 9.4, dùng cho 7.1 và 7.4).

Tiền tiết kiệm = số đơn x (tỷ lệ trễ cũ - tỷ lệ trễ mới) x chi phí xử lý một đơn trễ.
Tỷ lệ trễ cũ và mới lấy từ backtest (data/evaluation.json). Chi phí một đơn trễ lấy từ bảng
"Chi phí xử lý đơn" của nhóm (việc 4.3) và có thể sửa trực tiếp.
"""
import pandas as pd
import streamlit as st

from ontime import core
from overview import INK, MUTED, card_header, kpi_card, vn_int, vn_pct

COSTS = pd.DataFrame({
    "Kịch bản": ["Thấp", "Cơ sở", "Cao"],
    "Chi phí 1 đơn trễ (VNĐ)": [10_000, 29_000, 62_000],
    "Cách tính (bảng chi phí việc 4.3)": [
        "Giao lại 1 lần (GHTK Express 10.000)",
        "Giao lại GHN 11.000 + lưu kho 2 ngày 3.000 + voucher giao trễ Shopee 15.000",
        "Phí trả hàng Shopee 40.000 + xử lý hàng hoàn 8.000 + đóng gói lại 8.000 + lưu kho 3 ngày 6.000",
    ],
})


def page_savings():
    st.markdown(f"<style>.sv-note {{ font-size:13px; color:{MUTED}; line-height:1.55; }} .sv-note b {{ color:{INK}; }}</style>",
                unsafe_allow_html=True)
    bt = pd.DataFrame(core.load_evaluation()["backtest"])
    allr = bt[bt["phuong_thuc"] == "Tất cả"].set_index("cach_hen")
    st.markdown('<div class="ov-title">Ước tính tiết kiệm</div>', unsafe_allow_html=True)
    st.markdown('<div class="ov-sub">Tiền tiết kiệm = số đơn x (tỷ lệ trễ cũ - tỷ lệ trễ mới) x chi phí xử lý một đơn trễ</div>',
                unsafe_allow_html=True)
    st.write("")

    with st.container(key="card_sv_in"):
        c1, c2, c3 = st.columns(3)
        orders = c1.number_input("Số đơn mỗi tháng", 100, 1_000_000, 1000, step=100)
        way = c2.selectbox("Cách hẹn ngày giao mới", [w for w in allr.index if w != "Như cũ"],
                           index=list(allr.index).index("Hẹn theo P80") - 1)
        aov = c3.number_input("Giá trị đơn trung bình (VNĐ)", 0, 100_000_000, 300_000, step=50_000,
                              help="Dùng để ước tính doanh thu giữ lại: 23% khách có thể ngừng mua sau giao hàng thất thường (Milieu).")
        old, new = allr.loc["Như cũ", "ty_le_tre_so_voi_hen"], allr.loc[way, "ty_le_tre_so_voi_hen"]
        st.markdown(f'<div class="sv-note">Theo backtest: tỷ lệ trễ so với ngày hẹn <b>{vn_pct(old)}</b> (như cũ) '
                    f'→ <b>{vn_pct(new)}</b> ({way}), đổi lại hẹn thêm trung bình '
                    f'<b>{allr.loc[way, "ngay_hen_them_tb"]:.2f}'.replace(".", ",") + ' ngày</b>.</div>', unsafe_allow_html=True)
        st.write("")
        costs = st.data_editor(COSTS, hide_index=True, width="stretch", disabled=["Kịch bản", "Cách tính (bảng chi phí việc 4.3)"],
                               column_config={"Chi phí 1 đơn trễ (VNĐ)": st.column_config.NumberColumn(
                                   min_value=0, step=1000, format="localized")})

    fewer = orders * (old - new)
    cols = st.columns(3, gap="medium")
    for i, (col, r) in enumerate(zip(cols, costs.fillna(0).itertuples())):
        cost = r[2]
        with col:
            kpi_card(f"card_sv{i}", "money", f"Kịch bản {r[1].lower()}: tiết kiệm / tháng",
                     vn_int(fewer * cost) + "<small>₫</small>",
                     f"bớt {vn_int(fewer)} đơn trễ · giữ lại ~{vn_int(fewer * 0.23 * aov)} ₫ doanh thu", i != 1, None, None)
    base = costs.fillna(0).iloc[1, 1]
    st.write("")
    st.success(f"Kịch bản cơ sở: cửa hàng {vn_int(orders)} đơn/tháng, cách \"{way}\", giảm khoảng "
               f"{vn_pct(1 - new / old, 0) if old else '-'} chi phí do giao trễ, tương đương {vn_int(fewer * base)} ₫ mỗi tháng.")
    st.markdown('<div class="sv-note">Đây là phần trăm chi phí do giao trễ, không phải phần trăm tổng chi phí. '
                'Tỷ lệ trễ lấy từ dữ liệu DataCo (không phải dữ liệu Việt Nam); chi phí một đơn trễ ghép từ bảng giá '
                'GHN, GHTK, Shopee trong bảng chi phí của nhóm. Chưa tính chi phí của việc hẹn ngày giao dài hơn '
                '(một số khách có thể bỏ giỏ hàng).</div>', unsafe_allow_html=True)
