"""Trang Trợ lý đơn hàng (Mục 5.5): hàng đợi đơn rủi ro + chi tiết một đơn + tin nhắn nháp + người duyệt.

Đầu ca, nhân viên mở trang này, thấy đơn mức Cao và Trung bình sắp theo tiền rủi ro kỳ vọng.
Chọn một đơn: xác suất trễ, ngày hẹn P80, tối đa 3 lý do (SHAP), hành động từ bảng luật và tin nhắn nháp.
Tin nhắn chỉ là bản nháp: nhân viên sửa rồi bấm Phê duyệt, mỗi lần duyệt được ghi nhật ký.
"""
from datetime import timedelta

import pandas as pd
import streamlit as st

from ontime import core, journal
from overview import ACCENT, INK, MUTED, card_header, kpi_card, fmt_int, fmt_pct

PAGE_CSS = f"""
<style>
[class*="st-key-card_as"] {{ min-height: 150px; }}
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

COLS = {"order_id": "Order ID", "route_name": "Route", "risk_tier": "Tier", "p_late": "Late prob.",
        "exp_rev_at_risk": "At risk (USD)", "scheduled_shipping_days": "Old (days)",
        "promise_days": "P80 (days)", "approved": "Approved"}


def demo_orders(q):
    """3 đơn chọn sẵn để demo (PCCV 9.2): 1 First Class, 1 Second Class Tây Âu, 1 đơn an toàn."""
    pick = lambda d: int(d.sort_values("exp_rev_at_risk", ascending=False).iloc[0]["order_id"]) if len(d) else None
    return {
        "First Class": pick(q[q["shipping_mode"] == "First Class"]),
        "Second Class (Europe)": pick(q[(q["shipping_mode"] == "Second Class") & (q["order_region"] == "Western Europe")]),
        "Safe order": pick(q[(q["risk_tier"] == "Low") & (q["shipping_mode"] == "Standard Class")]),
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
        f'<div class="od-title">Order {oid} · {r["route_name"]}</div>'
        f'<div class="od-sub">Placed at {r["order_ts"]:%H:%M, %d %b %Y} · {r["n_items"]} item{"s" if r["n_items"] != 1 else ""} · '
        f'{r["order_value"]:,.2f} USD<br>{r["products"][:120]}</div>'
        '<div class="od-row">'
        f'<div class="od-box red"><div class="l">Late probability</div><div class="v">{fmt_pct(s["p_late"])}</div>'
        f'<div class="s">{s["risk_level"]} risk · {s["source"]}</div></div>'
        f'<div class="od-box"><div class="l">Suggested promise date (P80)</div><div class="v">{new_due:%d %b %Y}</div>'
        f'<div class="s">promise {pr["promise_days"]} days instead of {int(r["scheduled_shipping_days"])} ({old_due:%d %b})</div></div>'
        f'<div class="od-box"><div class="l">P50 · P80 · P90</div><div class="v">{pr["p50"]:g} · {pr["p80"]:g} · {pr["p90"]:g}</div>'
        f'<div class="s">predicted actual delivery days</div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="od-h">Factors that increase risk</div>', unsafe_allow_html=True)
    for x in ex["reasons"]:
        st.markdown(f'<div class="od-reason">{x["label"]} <span>· SHAP +{x["shap"]:.2f}</span></div>',
                    unsafe_allow_html=True)
    st.markdown('<div class="od-note">SHAP shows what the model relies on to predict; it does not prove causation.</div>',
                unsafe_allow_html=True)

    acts = core.actions_for(r)
    st.markdown('<div class="od-h">What to do (action rule table)</div>', unsafe_allow_html=True)
    for a in acts:
        st.markdown(f'<div class="od-act"><code>{a["code"]}</code> {a["action"]} · <i>{a["owner"]}</i></div>',
                    unsafe_allow_html=True)
    if not any(a["notify_customer"] == "1" for a in acts):
        st.success("No need to message the customer in advance (rule R4).")
        return

    st.markdown('<div class="od-h">Explanation and draft message</div>', unsafe_allow_html=True)
    key = f"draft_{oid}"
    if key not in st.session_state:
        if st.button("Draft with AI", icon=":material/auto_awesome:", key=f"gen_{oid}"):
            with st.spinner("Drafting (customer data masked, numbers checked)..."):
                st.session_state[key] = core.assist_order(oid)
            st.rerun()
        if st.button("Use the template (no AI call)", key=f"tpl_{oid}"):
            st.session_state[key] = core.assist_order(oid, use_llm=False)
            st.rerun()
        return
    d = st.session_state[key]
    st.caption(f"Source: **{d['source']}** · {d['note']}")
    st.info(d["explanation"])
    msg = st.text_area("Message to the customer (edit before approving)", d["draft_message"], height=130, key=f"msg_{oid}")
    who = st.text_input("Reviewer", st.session_state.get("reviewer", ""), placeholder="Customer service staff name",
                        key=f"who_{oid}")
    a, b, c = st.columns([1, 1, 1.4])
    if a.button("Approve", key="approve_btn", width="stretch", disabled=not who.strip()):
        journal.add(oid, who.strip(), "Approved", d["source"], d["draft_message"], msg)
        st.session_state["reviewer"] = who.strip()
        st.toast(f"Message for order {oid} approved. The system does not send it: staff send it through the shop's channel.")
        st.rerun()
    if b.button("Reject", key=f"reject_{oid}", width="stretch", disabled=not who.strip()):
        journal.add(oid, who.strip(), "Rejected", d["source"], d["draft_message"], msg)
        st.session_state["reviewer"] = who.strip()
        st.rerun()
    if c.button("Redraft", key=f"redo_{oid}", width="stretch"):
        st.session_state.pop(key, None)
        st.rerun()
    st.markdown('<div class="od-note">Customer name and address are masked as [CUSTOMER_NAME] and [ADDRESS] before going to the AI '
                'and restored on the server. Every number in the AI answer must appear in the order data, '
                'one wrong number and the template is used.</div>', unsafe_allow_html=True)


def page_assistant():
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    q = core.load_queue()
    log = journal.read()
    approved = set(log.loc[log["decision"] == "Approved", "order_id"].astype(str))

    # ---------- Filters
    days = sorted(q["order_ts"].dt.date.unique())
    st.session_state.setdefault("as_day", busy_day(q))
    st.session_state.setdefault("as_tiers", ["High", "Medium"])
    with st.sidebar:
        st.markdown('<div class="card-title" style="margin-top:18px">Filters</div>', unsafe_allow_html=True)
        day = st.date_input("Order date", key="as_day", min_value=days[0], max_value=days[-1], format="YYYY/MM/DD")
        tiers = st.pills("Risk tier", ["High", "Medium", "Low"], selection_mode="multi", key="as_tiers")
        modes = st.multiselect("Mode", sorted(q["shipping_mode"].unique()), placeholder="All", key="as_modes")
        oid_txt = st.text_input("Find order ID", key="as_find", placeholder="e.g. 68821")

    left, right = st.columns([1.5, 2], vertical_alignment="bottom")
    left.markdown('<div class="ov-title">Order assistant</div>', unsafe_allow_html=True)
    with right:
        st.caption("Demo orders:")
        demos = demo_orders(q)
        cols = st.columns(len(demos))
        def open_demo(oid):                     # chạy trước khi vẽ lại trang nên được phép đổi giá trị bộ lọc
            st.session_state.update(sel_order=oid, as_find="", as_tiers=["High", "Medium", "Low"],
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
    v["approved"] = v["order_id"].astype(str).isin(approved)

    # ---------- Thẻ chỉ số của ngày đang xem
    allday = q[q["order_ts"].dt.date == day]
    need = allday[allday["risk_tier"].isin(["High", "Medium"])]
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        kpi_card("card_as1", "box", "Orders that day", fmt_int(len(allday)), f"{day:%d %b %Y}", True, None, None)
    with c2:
        kpi_card("card_as2", "late", "High-risk orders", fmt_int((allday["risk_tier"] == "High").sum()),
                 f"{fmt_int((allday['shipping_mode'] == 'First Class').sum())} First Class by rule", False, None, None)
    with c3:
        kpi_card("card_as3", "money", "Expected revenue at risk", fmt_int(allday["exp_rev_at_risk"].sum()) + "<small>USD</small>",
                 "prob. x order value", False, None, None)
    with c4:
        left_n = (~need["order_id"].astype(str).isin(approved)).sum()
        kpi_card("card_as4", "delay", "Awaiting approval", fmt_int(left_n), "High + Medium", True, None, None)

    # ---------- Hàng đợi + chi tiết
    a, b = st.columns([1.3, 1], gap="medium")
    with a, st.container(key="card_queue"):
        card_header("Alert queue", f"{len(v)} orders · sorted by expected revenue at risk · click the first cell of a row to open it")
        # khóa bảng gắn với bộ lọc: đổi bộ lọc thì bỏ dòng đang chọn, tránh trỏ nhầm sang đơn khác
        tbl_key = f"as_tbl_{day}_{'-'.join(tiers or [])}_{'-'.join(modes)}_{oid_txt.strip()}"
        ev = st.dataframe(
            v[list(COLS)].rename(columns=COLS), hide_index=True, width="stretch", height=560,
            on_select="rerun", selection_mode="single-row", key=tbl_key,
            column_config={
                "Late prob.": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1, width="small"),
                "At risk (USD)": st.column_config.NumberColumn(format="localized", width="small", help="Expected revenue at risk = late probability x order value"),
                "Order ID": st.column_config.NumberColumn(format="%d", width="small"),
                "Approved": st.column_config.CheckboxColumn(width="small"), "Tier": st.column_config.TextColumn(width="small"), "Old (days)": st.column_config.NumberColumn(width="small"), "P80 (days)": st.column_config.NumberColumn(width="small")})
        rows = ev.selection.rows if ev else []
        if rows:
            st.session_state["sel_order"] = int(v.loc[rows[0], "order_id"])
    with b, st.container(key="card_detail"):
        oid = st.session_state.get("sel_order") or (int(v["order_id"].iloc[0]) if len(v) else None)
        hit = q[q["order_id"] == oid]
        if hit.empty:
            st.caption("Select an order in the queue.")
        else:
            detail(hit.iloc[0])

    # ---------- Nhật ký duyệt
    with st.container(key="card_log"):
        card_header("Message review log", "Who approved, when, and whether the draft was edited")
        if log.empty:
            st.caption("No reviews yet.")
        else:
            edited = log["edited"].mean()
            st.markdown(f'<div class="od-note">{len(log)} reviews · {fmt_pct(edited, 0)} of drafts edited · '
                        f'{fmt_int((log["decision"] == "Rejected").sum())} rejections</div>', unsafe_allow_html=True)
            st.dataframe(log.drop(columns="id").rename(columns={
                "ts": "Time", "order_id": "Order ID", "reviewer": "Reviewer", "decision": "Decision",
                "draft_source": "Draft source", "draft": "Draft", "final_text": "Approved text", "edited": "Edited"}),
                hide_index=True, width="stretch", column_config={"Edited": st.column_config.CheckboxColumn()})
