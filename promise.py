"""Trang Ngày hẹn và đánh giá: kết quả Mục 5.1 (ngày hẹn P50/P80/P90), backtest 3 cách hẹn (PCCV 6.1),
Precision@K của hàng đợi (PCCV 6.2) và mức ảnh hưởng SHAP của các biến (Mục 5.3).

Mọi số trên trang đọc từ data/evaluation.json, do scripts/build_artifacts.py tính trên tập kiểm tra
(đơn đặt từ 10/07/2017), không tính lại khi mở app.
"""
import altair as alt
import pandas as pd
import streamlit as st

from ontime import core
from overview import ACCENT, INK, MODE_ORDER, MUTED, F, card_header, clean, kpi_card, vn_int, vn_pct

PAGE_CSS = f"""
<style>
.st-key-card_bt, .st-key-card_pk {{ min-height: 430px; }}
.pr-note {{ font-size:13px; color:{MUTED}; line-height:1.55; margin-top:6px; }}
.pr-note b {{ color:{INK}; }}
</style>
"""
WAYS = ["Như cũ", "First Class hẹn 2 ngày", "Hẹn theo P50", "Hẹn theo P80", "Hẹn theo P90"]


def backtest_chart(bt):
    d = bt[bt["phuong_thuc"] == "Tất cả"][["cach_hen", "ty_le_tre_so_voi_hen", "ngay_hen_them_tb"]].rename(columns={
        "cach_hen": "Cách hẹn", "ty_le_tre_so_voi_hen": "Trễ so với ngày hẹn", "ngay_hen_them_tb": "Số ngày hẹn thêm"})
    d["Nhãn"] = d["Trễ so với ngày hẹn"].map(vn_pct) + " · +" + d["Số ngày hẹn thêm"].map(lambda x: f"{x:.2f}".replace(".", ",")) + " ngày"
    d["Đang dùng"] = d["Cách hẹn"] == "Hẹn theo P80"
    base = alt.Chart(d).encode(y=alt.Y(**F("Cách hẹn", "N"), sort=WAYS, title=None,
                                       axis=alt.Axis(labelColor=INK, labelFontSize=12.5, labelLimit=220)))
    bars = base.mark_bar(cornerRadius=8, size=20).encode(
        x=alt.X(**F("Trễ so với ngày hẹn", "Q"), axis=None, scale=alt.Scale(domain=[0, 1.7])),   # chừa chỗ cho nhãn
        color=alt.condition("datum['Đang dùng']", alt.value(ACCENT), alt.value("#F2A6A4")),
        tooltip=[alt.Tooltip(**F("Cách hẹn", "N")), alt.Tooltip(**F("Trễ so với ngày hẹn", "Q"), format=".1%"),
                 alt.Tooltip(**F("Số ngày hẹn thêm", "Q"), format=".2f")])
    text = base.mark_text(align="left", dx=6, color=INK, fontSize=12, font="Inter", fontWeight=600).encode(
        x=alt.X(**F("Trễ so với ngày hẹn", "Q")), text=alt.Text(**F("Nhãn", "N")))
    return clean(alt.layer(bars, text), 260)


def shap_chart(sg):
    d = pd.DataFrame({"Biến": list(sg), "Mức ảnh hưởng": list(sg.values())}).head(8)
    d["Cao nhất"] = d["Mức ảnh hưởng"] == d["Mức ảnh hưởng"].max()
    base = alt.Chart(d).encode(y=alt.Y(**F("Biến", "N"), sort=None, title=None,
                                       axis=alt.Axis(labelColor=INK, labelFontSize=12)))
    bars = base.mark_bar(cornerRadius=8, size=16).encode(
        x=alt.X(**F("Mức ảnh hưởng", "Q"), axis=None),
        color=alt.condition("datum['Cao nhất']", alt.value(ACCENT), alt.value("#F2A6A4")),
        tooltip=[alt.Tooltip(**F("Biến", "N")), alt.Tooltip(**F("Mức ảnh hưởng", "Q"), format=".3f")])
    text = base.mark_text(align="left", dx=6, color=INK, fontSize=12, font="Inter").encode(
        x=alt.X(**F("Mức ảnh hưởng", "Q")), text=alt.Text(**F("Mức ảnh hưởng", "Q"), format=".2f"))
    return clean(alt.layer(bars, text), 280)


def page_promise():
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    ev = core.load_evaluation()
    bt = pd.DataFrame(ev["backtest"])
    pe = ev["promise"]
    allr = bt[bt["phuong_thuc"] == "Tất cả"].set_index("cach_hen")
    pk = ev["precision_at_k"]

    st.markdown('<div class="ov-title">Ngày hẹn giao và đánh giá</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ov-sub">Tập kiểm tra: {vn_int(ev["n_test_orders"])} đơn đặt từ 10/07/2017 · '
                f'ROC-AUC mô hình cảnh báo {ev["auc_test"]:.4f}'.replace(".", ",") + '</div>', unsafe_allow_html=True)
    st.write("")

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        kpi_card("card_pr1", "late", "Trễ so với ngày hẹn hiện tại", vn_pct(allr.loc["Như cũ", "ty_le_tre_so_voi_hen"]),
                 "giữ cam kết như cũ", True, None, None)
    with c2:
        kpi_card("card_pr2", "box", "Nếu hẹn theo P80", vn_pct(allr.loc["Hẹn theo P80", "ty_le_tre_so_voi_hen"]),
                 "đơn trễ so với ngày hẹn mới", False, None, None)
    with c3:
        extra = allr.loc["Hẹn theo P80", "ngay_hen_them_tb"]
        kpi_card("card_pr3", "delay", "Cái giá: hẹn thêm", f"+{extra:.2f}".replace(".", ",") + "<small>ngày</small>",
                 "trung bình mỗi đơn", True, None, None)
    with c4:
        p10 = pk["Bỏ First Class (đã xử lý bằng luật)"]["Top 10/ngày, theo xác suất trễ"]
        kpi_card("card_pr4", "money", "Precision@10 mỗi ngày", vn_pct(p10, 0),
                 "bỏ First Class, xếp theo xác suất", False, None, None)

    # ---------- Backtest 3 cách hẹn (6.1)
    a, b = st.columns([1.25, 1], gap="medium")
    with a, st.container(key="card_bt"):
        card_header("So sánh các cách hẹn ngày giao", "Tỷ lệ đơn trễ so với ngày hẹn · số ngày hẹn thêm trung bình")
        st.altair_chart(backtest_chart(bt), width="stretch")
        st.markdown('<div class="pr-note">Số ngày giao thực tế trong DataCo là số nguyên (0 đến 6 ngày), nên ngày hẹn P80 '
                    'làm tròn lên thường phủ nhiều hơn 80% đơn. Hẹn theo P80 gần như hết trễ hẹn nhưng ngày hẹn dài hơn; '
                    'hẹn theo P50 rút ngắn ngày hẹn nhưng còn khoảng 1/3 đơn trễ. Doanh nghiệp chọn mức theo khả năng '
                    'chấp nhận của khách.</div>', unsafe_allow_html=True)
    with b, st.container(key="card_pk"):
        card_header("Chi tiết theo phương thức")
        mode = st.segmented_control("Phương thức", MODE_ORDER, default="Second Class", key="pr_mode",
                                    label_visibility="collapsed") or "Second Class"
        t = bt[bt["phuong_thuc"] == mode].set_index("cach_hen").reindex(WAYS)
        st.dataframe(t[["ty_le_tre_so_voi_hen", "ngay_hen_tb", "ngay_hen_them_tb"]].reset_index(), hide_index=True,
                     width="stretch", column_config={
                         "cach_hen": "Cách hẹn",
                         "ty_le_tre_so_voi_hen": st.column_config.ProgressColumn("Trễ so với hẹn", format="percent",
                                                                                 min_value=0, max_value=1),
                         "ngay_hen_tb": st.column_config.NumberColumn("Hẹn TB (ngày)", format="%.2f"),
                         "ngay_hen_them_tb": st.column_config.NumberColumn("Hẹn thêm (ngày)", format="%+.2f")})
        st.caption(f"{int(t['so_don'].iloc[0]):,} đơn {mode} trong tập kiểm tra".replace(",", "."))

    # ---------- Mô hình phân vị so với bảng phân vị (5.1)
    x, y = st.columns([1, 1], gap="medium")
    with x, st.container(key="card_q"):
        card_header("Mô hình phân vị so với bảng phân vị lịch sử",
                    "Pinball loss càng thấp càng tốt · độ phủ P80 = tỷ lệ đơn giao kịp ngày hẹn P80")
        rows = []
        for name, lab in (("model", "Mô hình hồi quy phân vị"), ("table", "Bảng phân vị tuyến x phương thức")):
            rows.append({"Cách dự báo": lab, **{f"Pinball {k.upper()}": v for k, v in pe["pinball"][name].items()},
                         "Độ phủ P80": pe["coverage_p80"][name]["Tất cả"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"Độ phủ P80": st.column_config.NumberColumn(format="percent")})
        chosen = "bảng phân vị lịch sử" if pe["chosen"] == "table" else "mô hình hồi quy phân vị"
        st.markdown(f'<div class="pr-note">Theo nguyên tắc ở Mục 5.1, nếu mô hình không vượt mốc đối chiếu thì dùng bảng '
                    f'vì đơn giản hơn. Kết quả: app đang dùng <b>{chosen}</b> để đề xuất ngày hẹn. Lý do: thời gian giao '
                    f'thực tế phụ thuộc gần như hoàn toàn vào phương thức và tuyến, các biến khác không thêm thông tin.</div>',
                    unsafe_allow_html=True)
    with y, st.container(key="card_shap"):
        card_header("Mô hình cảnh báo dựa vào đâu", "Trung bình |SHAP| trên 3.000 dòng hàng của tập kiểm tra")
        st.altair_chart(shap_chart(ev["shap_global"]), width="stretch")

    # ---------- Precision@K (6.2) và bảng P50/P80/P90
    with st.container(key="card_pk2"):
        card_header("Hàng đợi cảnh báo đúng bao nhiêu (Precision@K)",
                    "Mỗi ngày nhân viên xử lý K đơn đầu hàng đợi thì bao nhiêu % trong đó thật sự trễ")
        pkd = pd.DataFrame(pk).T
        st.dataframe(pkd, width="stretch", column_config={
            c: st.column_config.NumberColumn(c, format="percent") for c in pkd.columns if c != "Số ngày"})
    with st.container(key="card_tbl"):
        card_header("Bảng P50 · P80 · P90 theo tuyến", "Số ngày giao thực tế, tính trên tập huấn luyện")
        t = core.load_promise_table()
        modes = st.multiselect("Phương thức", MODE_ORDER, placeholder="Tất cả phương thức", key="pr_tbl_modes")
        if modes:
            t = t[t["shipping_mode"].isin(modes)]
        t = t.assign(hen_p80=t["p80"].map(lambda v: int(-(-v // 1))))
        st.dataframe(t[["route_name", "n_lines", "p50", "p80", "p90", "hen_p80"]], hide_index=True, width="stretch",
                     height=380, column_config={
                         "route_name": "Tuyến", "n_lines": st.column_config.NumberColumn("Số dòng hàng", format="localized"),
                         "p50": "P50", "p80": "P80", "p90": "P90", "hen_p80": "Ngày hẹn đề xuất"})
