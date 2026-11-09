"""Trang Trợ lý đơn hàng (Mục 5.5): hàng đợi đơn rủi ro + chi tiết một đơn + tin nhắn nháp + người duyệt.

Đầu ca, nhân viên mở trang này, thấy đơn mức Cao và Trung bình sắp theo tiền rủi ro kỳ vọng.
Chọn một đơn: xác suất trễ, ngày hẹn P80, tối đa 3 lý do (SHAP), hành động từ bảng luật và tin nhắn nháp.
Tin nhắn chỉ là bản nháp: nhân viên sửa rồi bấm Phê duyệt, mỗi lần duyệt được ghi nhật ký.
"""
from datetime import timedelta

import pandas as pd
import streamlit as st

from ontime import core, journal
from overview import ACCENT, INK, MUTED, card_header, kpi_card, vn_int, vn_pct

PAGE_CSS = f"""
<style>
.st-key-card_queue, .st-key-card_detail {{ min-height: 640px; }}
.od-title {{ font-size:19px; font-weight:600; color:{INK}; margin:0; }}
.od-sub   {{ font-size:13px; color:{MUTED}; margin:2px 0 12px 0; line-height:1.5; }}
.od-row   {{ display:flex; gap:10px; margin:6px 0 14px 0; flex-wrap:wrap; }}
.od-box   {{ flex:1 1 120px; background:#F7F7F5; border-radius:14px; padding:10px 12px; }}
.od-box.red {{ background:#FDECEC; }}
.od-box .l {{ font-size:12px; color:{MUTED}; }}
.od-box .v {{ font-size:20px; font-weight:600; color:{INK}; margin-top:2px; }}
.od-box.red .v {{ color:{ACCENT}; }}
.od-box .s {{ font-size:11.5px; color:{MUTED}; margin-top:2px; }}
.od-h     {{ font-size:14px; font-weight:600; color:{INK}; margin:12px 0 6px 0; }}
.od-reason {{ border-left:3px solid {ACCENT}; background:#FFFFFF; padding:6px 10px; margin-bottom:6px;
              font-size:13.5px; color:{INK}; border-radius:4px; }}
.od-reason span {{ color:{MUTED}; font-size:12px; }}
.od-act   {{ font-size:13.5px; color:{INK}; padding:6px 0; border-bottom:1px dashed #EDEDED; }}
.od-act code {{ background:#FDECEC; color:{ACCENT}; border-radius:6px; padding:1px 6px; }}
.od-note  {{ font-size:12px; color:{MUTED}; line-height:1.5; }}
.st-key-approve_btn button {{ background:{ACCENT}; border-color:{ACCENT}; color:#FFFFFF; }}
</style>
"""

COLS = {"order_id": "Mã đơn", "route_name": "Tuyến", "risk_tier": "Mức", "p_late": "Xác suất trễ",
        "exp_rev_at_risk": "Tiền rủi ro kỳ vọng (USD)", "scheduled_shipping_days": "Hẹn cũ (ngày)",
        "promise_days": "Hẹn P80 (ngày)", "duyet": "Đã duyệt"}


def demo_orders(q):
    """3 đơn chọn sẵn để demo (PCCV 9.2): 1 First Class, 1 Second Class Tây Âu, 1 đơn an toàn."""
    pick = lambda d: int(d.sort_values("exp_rev_at_risk", ascending=False).iloc[0]["order_id"]) if len(d) else None
    return {
        "First Class": pick(q[q["shipping_mode"] == "First Class"]),
        "Second Class Tây Âu": pick(q[(q["shipping_mode"] == "Second Class") & (q["order_region"] == "Western Europe")]),
        "Đơn an toàn": pick(q[(q["risk_tier"] == "Thấp") & (q["shipping_mode"] == "Standard Class")]),
    }


def busy_day(q):
    """Ngày mặc định: ngày gần nhất có từ 40 đơn trở lên (cuối bộ DataCo có vài ngày rất ít đơn)."""
    n = q.groupby(q["order_ts"].dt.date).size()
    full = n[n >= 40]
    return (full.index.max() if len(full) else n.index.max())


def detail(r):
    """Chi tiết một đơn: số liệu, lý do, hành động, tin nhắn nháp và nút duyệt."""
    oid = int(r["order_id"])
    s, pr, ex = core.score_order(oid), core.recommend_promise(oid), core.explain_order(oid)
    old_due = r["order_ts"] + timedelta(days=int(r["scheduled_shipping_days"]))
    new_due = r["order_ts"] + timedelta(days=pr["promise_days"])
    st.markdown(
        f'<div class="od-title">Đơn {oid} · {r["route_name"]}</div>'
        f'<div class="od-sub">Đặt lúc {r["order_ts"]:%H:%M %d/%m/%Y} · {r["n_items"]} sản phẩm · '
        f'{r["order_value"]:,.2f} USD<br>{r["products"][:120]}</div>'
        '<div class="od-row">'
        f'<div class="od-box red"><div class="l">Xác suất trễ</div><div class="v">{vn_pct(s["p_late"])}</div>'
        f'<div class="s">Mức {s["risk_level"]} · {s["nguon"]}</div></div>'
        f'<div class="od-box"><div class="l">Ngày hẹn đề xuất (P80)</div><div class="v">{new_due:%d/%m/%Y}</div>'
        f'<div class="s">hẹn {pr["promise_days"]} ngày, cũ {int(r["scheduled_shipping_days"])} ngày ({old_due:%d/%m})</div></div>'
        f'<div class="od-box"><div class="l">P50 · P80 · P90</div><div class="v">{pr["p50"]:g} · {pr["p80"]:g} · {pr["p90"]:g}</div>'
        f'<div class="s">số ngày giao thực tế dự báo</div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="od-h">Yếu tố làm tăng rủi ro</div>', unsafe_allow_html=True)
    for x in ex["reasons"]:
        st.markdown(f'<div class="od-reason">{x["label_vi"]} <span>· SHAP +{x["shap"]:.2f}</span></div>',
                    unsafe_allow_html=True)
    st.markdown('<div class="od-note">SHAP cho biết mô hình dựa vào đâu để dự báo, không chứng minh quan hệ nhân quả.</div>',
                unsafe_allow_html=True)

    acts = core.actions_for(r)
    st.markdown('<div class="od-h">Việc nên làm (bảng luật hành động)</div>', unsafe_allow_html=True)
    for a in acts:
        st.markdown(f'<div class="od-act"><code>{a["ma"]}</code> {a["hanh_dong"]} · <i>{a["nguoi_thuc_hien"]}</i></div>',
                    unsafe_allow_html=True)
    if not any(a["bao_khach"] == "1" for a in acts):
        st.success("Không cần nhắn trước cho khách (luật R4).")
        return

    st.markdown('<div class="od-h">Lời giải thích và tin nhắn nháp</div>', unsafe_allow_html=True)
    key = f"draft_{oid}"
    if key not in st.session_state:
        if st.button("Soạn bằng AI", icon=":material/auto_awesome:", key=f"gen_{oid}"):
            with st.spinner("Đang soạn (che thông tin khách, kiểm tra số liệu)..."):
                st.session_state[key] = core.assist_order(oid)
            st.rerun()
        if st.button("Dùng câu mẫu (không gọi AI)", key=f"tpl_{oid}"):
            st.session_state[key] = core.assist_order(oid, use_llm=False)
            st.rerun()
        return
    d = st.session_state[key]
    st.caption(f"Nguồn: **{d['nguon']}** · {d['ghi_chu']}")
    st.info(d["giai_thich"])
    msg = st.text_area("Tin nhắn gửi khách (sửa trước khi duyệt)", d["tin_nhan_nhap"], height=130, key=f"msg_{oid}")
    who = st.text_input("Người duyệt", st.session_state.get("reviewer", ""), placeholder="Tên nhân viên CSKH",
                        key=f"who_{oid}")
    a, b, c = st.columns([1, 1, 1.4])
    if a.button("Phê duyệt", key="approve_btn", width="stretch", disabled=not who.strip()):
        journal.add(oid, who.strip(), "Phê duyệt", d["nguon"], d["tin_nhan_nhap"], msg)
        st.session_state["reviewer"] = who.strip()
        st.toast(f"Đã duyệt tin cho đơn {oid}. Hệ thống không tự gửi: nhân viên gửi qua kênh của shop.")
        st.rerun()
    if b.button("Từ chối", key=f"reject_{oid}", width="stretch", disabled=not who.strip()):
        journal.add(oid, who.strip(), "Từ chối", d["nguon"], d["tin_nhan_nhap"], msg)
        st.session_state["reviewer"] = who.strip()
        st.rerun()
    if c.button("Soạn lại", key=f"redo_{oid}", width="stretch"):
        st.session_state.pop(key, None)
        st.rerun()
    st.markdown('<div class="od-note">Tên và địa chỉ khách được che bằng [KHACH_HANG], [DIA_CHI] trước khi gửi cho AI '
                'và điền lại trên máy chủ. Mọi con số trong câu trả lời của AI phải có trong dữ liệu của đơn, '
                'lệch một số là dùng câu mẫu.</div>', unsafe_allow_html=True)


def page_assistant():
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    q = core.load_queue()
    log = journal.read()
    approved = set(log.loc[log["quyet_dinh"] == "Phê duyệt", "ma_don"].astype(str))

    # ---------- Bộ lọc
    days = sorted(q["order_ts"].dt.date.unique())
    st.session_state.setdefault("as_day", busy_day(q))
    st.session_state.setdefault("as_tiers", ["Cao", "Trung bình"])
    with st.sidebar:
        st.markdown('<div class="card-title" style="margin-top:18px">Bộ lọc</div>', unsafe_allow_html=True)
        day = st.date_input("Ngày đặt hàng", key="as_day", min_value=days[0], max_value=days[-1], format="DD/MM/YYYY")
        tiers = st.pills("Mức rủi ro", ["Cao", "Trung bình", "Thấp"], selection_mode="multi", key="as_tiers")
        modes = st.multiselect("Phương thức", sorted(q["shipping_mode"].unique()), placeholder="Tất cả", key="as_modes")
        oid_txt = st.text_input("Tìm mã đơn", key="as_find", placeholder="vd 68821")

    left, right = st.columns([1.5, 2], vertical_alignment="bottom")
    left.markdown('<div class="ov-title">Trợ lý đơn hàng</div>', unsafe_allow_html=True)
    with right:
        st.caption("Đơn mẫu để demo:")
        demos = demo_orders(q)
        cols = st.columns(len(demos))
        def open_demo(oid):                     # chạy trước khi vẽ lại trang nên được phép đổi giá trị bộ lọc
            st.session_state.update(sel_order=oid, as_find="", as_tiers=["Cao", "Trung bình", "Thấp"],
                                    as_day=q.loc[q["order_id"] == oid, "order_ts"].dt.date.iloc[0])
        for col, (name, oid) in zip(cols, demos.items()):
            if oid:
                col.button(name, key=f"demo_{name}", width="stretch", on_click=open_demo, args=(oid,))
    st.write("")

    v = q[q["order_ts"].dt.date == day]
    if oid_txt.strip().isdigit():
        v = q[q["order_id"] == int(oid_txt.strip())]
    if tiers:
        v = v[v["risk_tier"].isin(tiers)]
    if modes:
        v = v[v["shipping_mode"].isin(modes)]
    v = v.sort_values("exp_rev_at_risk", ascending=False).reset_index(drop=True)
    v["duyet"] = v["order_id"].astype(str).isin(approved)

    # ---------- Thẻ chỉ số của ngày đang xem
    allday = q[q["order_ts"].dt.date == day]
    need = allday[allday["risk_tier"].isin(["Cao", "Trung bình"])]
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        kpi_card("card_as1", "box", "Đơn trong ngày", vn_int(len(allday)), f"{day:%d/%m/%Y}", True, None, None)
    with c2:
        kpi_card("card_as2", "late", "Đơn mức Cao", vn_int((allday["risk_tier"] == "Cao").sum()),
                 f"{vn_int((allday['shipping_mode'] == 'First Class').sum())} đơn First Class theo luật", False, None, None)
    with c3:
        kpi_card("card_as3", "money", "Tiền rủi ro kỳ vọng", vn_int(allday["exp_rev_at_risk"].sum()) + "<small>USD</small>",
                 "xác suất trễ x giá trị đơn", False, None, None)
    with c4:
        left_n = (~need["order_id"].astype(str).isin(approved)).sum()
        kpi_card("card_as4", "delay", "Cần báo khách, chưa duyệt", vn_int(left_n), "mức Cao và Trung bình", True, None, None)

    # ---------- Hàng đợi + chi tiết
    a, b = st.columns([1.15, 1], gap="medium")
    with a, st.container(key="card_queue"):
        card_header("Hàng đợi cảnh báo", f"{len(v)} đơn · xếp theo tiền rủi ro kỳ vọng · bấm ô đầu dòng để mở đơn")
        # khóa bảng gắn với bộ lọc: đổi bộ lọc thì bỏ dòng đang chọn, tránh trỏ nhầm sang đơn khác
        tbl_key = f"as_tbl_{day}_{'-'.join(tiers or [])}_{'-'.join(modes)}_{oid_txt.strip()}"
        ev = st.dataframe(
            v[list(COLS)].rename(columns=COLS), hide_index=True, width="stretch", height=560,
            on_select="rerun", selection_mode="single-row", key=tbl_key,
            column_config={
                "Xác suất trễ": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                "Tiền rủi ro kỳ vọng (USD)": st.column_config.NumberColumn(format="localized"),
                "Mã đơn": st.column_config.NumberColumn(format="%d"),
                "Đã duyệt": st.column_config.CheckboxColumn()})
        rows = ev.selection.rows if ev else []
        if rows:
            st.session_state["sel_order"] = int(v.loc[rows[0], "order_id"])
    with b, st.container(key="card_detail"):
        oid = st.session_state.get("sel_order") or (int(v["order_id"].iloc[0]) if len(v) else None)
        hit = q[q["order_id"] == oid]
        if hit.empty:
            st.caption("Chọn một đơn ở hàng đợi.")
        else:
            detail(hit.iloc[0])

    # ---------- Nhật ký duyệt
    with st.container(key="card_log"):
        card_header("Nhật ký duyệt tin nhắn", "Ai duyệt, lúc nào, có sửa bản nháp không")
        if log.empty:
            st.caption("Chưa có lần duyệt nào.")
        else:
            edited = log["da_sua"].mean()
            st.markdown(f'<div class="od-note">{len(log)} lần duyệt · {vn_pct(edited, 0)} bản nháp bị sửa · '
                        f'{vn_int((log["quyet_dinh"] == "Từ chối").sum())} lần từ chối</div>', unsafe_allow_html=True)
            st.dataframe(log.drop(columns="id").rename(columns={
                "thoi_gian": "Thời gian", "ma_don": "Mã đơn", "nguoi_duyet": "Người duyệt", "quyet_dinh": "Quyết định",
                "nguon_ban_nhap": "Nguồn bản nháp", "ban_nhap": "Bản nháp", "ban_cuoi": "Bản đã duyệt", "da_sua": "Có sửa"}),
                hide_index=True, width="stretch", column_config={"Có sửa": st.column_config.CheckboxColumn()})
