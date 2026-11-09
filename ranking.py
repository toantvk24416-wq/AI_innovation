"""Trang Xếp hạng rủi ro tuyến: "Tuyến nào cần xử lý trước?"

Dữ liệu: route_risk.csv = bảng gold.route_risk_expected (mỗi dòng là một tuyến
= khu vực nhận hàng x phương thức vận chuyển, 92 tuyến).

Cấu trúc file:
  1. Hằng số: tọa độ minh họa của kho và 23 khu vực
  2. Chuẩn bị dữ liệu (lọc, gộp theo khu vực, xếp hạng)
  3. Các biểu đồ: bản đồ tuyến, ma trận ưu tiên, bản đồ nhiệt
  4. page_ranking(): ghép thành trang
"""
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from overview import (ACCENT, BG, INK, MODE_ORDER, MUTED, TIER_COLOR, TIER_ORDER, TIER_VI,
                      F, card_header, clean, kpi_card, load_mode_days, load_route_risk, vn_int, vn_pct)

# ============================================================ 1. HẰNG SỐ
# Kho xuất hàng: dữ liệu không có tọa độ kho, nên dùng vị trí cửa hàng phổ biến nhất
# trong dữ liệu (Caguas, Puerto Rico) làm điểm xuất phát MINH HỌA. Sửa ở đây nếu cần.
HUB = {"name": "Kho xuất hàng", "lat": 18.23, "lon": -66.04}

# Tọa độ tâm (gần đúng) của 23 khu vực nhận hàng, chỉ dùng để vẽ minh họa.
REGION_XY = {
    "Canada": (53, -105), "Caribbean": (18.8, -74), "Central Africa": (0, 18),
    "Central America": (14, -88), "Central Asia": (44, 66), "East Africa": (-4, 36),
    "East of USA": (40, -78), "Eastern Asia": (35, 115), "Eastern Europe": (52, 30),
    "North Africa": (28, 8), "Northern Europe": (61, 14), "Oceania": (-26, 138),
    "South America": (-15, -58), "South Asia": (23, 78), "South of  USA": (31, -90),
    "Southeast Asia": (6, 108), "Southern Africa": (-26, 25), "Southern Europe": (41, 14),
    "US Center": (40, -98), "West Africa": (12, -5), "West Asia": (30, 45),
    "West of USA": (38, -119), "Western Europe": (48, 4),
}
WORLD_FILE = "world-110m.json"   # bản đồ thế giới (nguồn: gói vega-datasets), để cùng thư mục với app.py

PAGE_CSS = f"""
<style>
.st-key-card_map, .st-key-card_alert {{ min-height: 470px; }}
.st-key-card_matrix, .st-key-card_heat {{ min-height: 640px; }}
[class*="st-key-card_rk"] {{ min-height: 150px; }}
.alert-row  {{ display:flex; align-items:center; gap:12px; padding:13px 0; border-bottom:1px solid #F1F1F1; }}
.alert-num  {{ width:30px; height:30px; border-radius:50%; background:{ACCENT}; color:#fff; font-weight:600;
               display:flex; align-items:center; justify-content:center; flex-shrink:0; font-size:14px; }}
.alert-name {{ font-weight:600; color:{INK}; font-size:14.5px; line-height:1.3; }}
.alert-meta {{ color:{MUTED}; font-size:12.5px; margin-top:2px; }}
.alert-body {{ flex:1; }}
.tier-pill  {{ font-size:12px; padding:3px 10px; border-radius:999px; font-weight:500; white-space:nowrap; }}
.tier-High  {{ background:#FDECEC; color:{ACCENT}; }}
.tier-Medium {{ background:#FEF3E8; color:#D9822B; }}
.tier-Low   {{ background:{BG}; color:{MUTED}; }}
.alert-foot {{ font-size:12.5px; color:{MUTED}; margin-top:12px; line-height:1.5; }}
.alert-big  {{ font-size:13px; color:{MUTED}; margin:4px 0 6px 0; }}
.alert-big b {{ font-size:26px; color:{INK}; font-weight:600; letter-spacing:-.5px; margin-right:4px; }}
.tb-wrap {{ max-height:470px; overflow:auto; margin-top:10px; border-radius:12px; border:1px solid #EFEFEF; }}
.tb {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
.tb th {{ position:sticky; top:0; background:#FFFFFF; color:{INK}; font-weight:600; text-align:left;
          padding:12px 12px; border:none; border-bottom:2px solid {INK}; white-space:nowrap; z-index:1; }}
.tb td {{ padding:10px 12px; border:none; border-bottom:1px solid #F2F2F2; color:{INK}; }}
.tb tr:hover td {{ background:#FCF6F6; }}
.tb .r {{ text-align:right; }}
.tb-rank {{ font-weight:600; width:56px; }}
.tb-name {{ font-weight:500; white-space:nowrap; }}
.tb tr {{ border:none; }}
.tb-muted {{ color:{MUTED}; }}
.tb-bar {{ display:flex; align-items:center; gap:8px; min-width:130px; }}
.tb-bar span {{ font-size:12.5px; min-width:40px; text-align:right; padding-right:4px; }}
.tb-track {{ flex:1; max-width:80px; height:6px; background:#F1F1F1; border-radius:6px; overflow:hidden; }}
.tb-fill {{ height:100%; background:{ACCENT}; border-radius:6px; }}
.tier-Insufficient {{ background:#F6F6F6; color:{MUTED}; }}
</style>
"""


# ============================================================ 2. CHUẨN BỊ DỮ LIỆU
def prepare(rr, region_market):
    d = rr.copy()
    d["market"] = d["order_region"].map(region_market)
    # Mẫu số của tỷ lệ trễ là số đơn KHÔNG bị hủy (= số đơn được mô hình chấm điểm).
    d["n_base"] = d["n_scored_orders"] if "n_scored_orders" in d else d["n_orders"]
    d["money"] = d["actual_revenue_at_risk"] if "actual_revenue_at_risk" in d else d["expected_revenue_at_risk"]
    d["n_late"] = d["actual_late_rate"] * d["n_base"]
    d["n_pred"] = d["predicted_late_rate"] * d["n_base"]
    d["tier_vi"] = d["risk_tier"].map(TIER_VI).fillna(d["risk_tier"])
    d["rank_in_mode"] = d.groupby("shipping_mode")["risk_score"].rank(ascending=False, method="min")
    return d


def weighted_rate(d, col="n_late"):
    """Tỷ lệ trễ chung = tổng đơn trễ / tổng đơn không hủy (không lấy trung bình các tuyến)."""
    base = d["n_base"].sum()
    return d[col].sum() / base if base else np.nan


def by_region(d):
    g = d.groupby("order_region", as_index=False).agg(
        n_orders=("n_orders", "sum"), n_base=("n_base", "sum"), n_late=("n_late", "sum"),
        money=("expected_revenue_at_risk", "sum"))
    g["late_rate"] = g["n_late"] / g["n_base"].where(g["n_base"] > 0)
    return g


def money_txt(x):
    return f"{x / 1e6:.2f}".replace(".", ",") + " triệu USD" if x >= 1e6 else f"{vn_int(x)} USD"


@st.cache_data
def load_world():
    from overview import BASE
    return json.loads((BASE / WORLD_FILE).read_text(encoding="utf-8"))


# ============================================================ 3. BIỂU ĐỒ
def arc_points(lat0, lon0, lat1, lon1, bend=0.22, n=40):
    """Điểm trên một đường cong (Bezier bậc 2) từ kho tới khu vực, để vẽ "tuyến" cho đẹp."""
    mx, my = (lon0 + lon1) / 2, (lat0 + lat1) / 2
    dx, dy = lon1 - lon0, lat1 - lat0
    cx, cy = mx - dy * bend, my + dx * bend
    t = np.linspace(0, 1, n)
    lon = (1 - t) ** 2 * lon0 + 2 * (1 - t) * t * cx + t ** 2 * lon1
    lat = (1 - t) ** 2 * lat0 + 2 * (1 - t) * t * cy + t ** 2 * lat1
    return lon, np.clip(lat, -58, 80)


MONEY = "Tiền rủi ro kỳ vọng (USD)"
TIER_VI_COLOR = {TIER_VI[k]: v for k, v in TIER_COLOR.items()}     # màu theo tên mức tiếng Việt


def route_map(reg, top):
    reg = reg[reg["order_region"].isin(REGION_XY)]
    reg = reg[["order_region", "n_orders", "late_rate", "money"]].rename(columns={
        "order_region": "Khu vực", "n_orders": "Số đơn", "late_rate": "Tỷ lệ trễ", "money": MONEY})
    reg = reg.round({"Tỷ lệ trễ": 4, MONEY: 0})
    reg["Vĩ độ"] = reg["Khu vực"].map(lambda r: REGION_XY[r][0])
    reg["Kinh độ"] = reg["Khu vực"].map(lambda r: REGION_XY[r][1])
    rows = []
    for r in reg.itertuples(index=False):
        lon, lat = arc_points(HUB["lat"], HUB["lon"], r[reg.columns.get_loc("Vĩ độ")], r[reg.columns.get_loc("Kinh độ")])
        rows += [{"Khu vực": r[0], "Thứ tự điểm": i, "Kinh độ": a, "Vĩ độ": b,
                  "Tỷ lệ trễ": r[2], MONEY: r[3]} for i, (a, b) in enumerate(zip(lon, lat))]
    lines = pd.DataFrame(rows)
    geo = dict(longitude=alt.Longitude(**F("Kinh độ", "Q")), latitude=alt.Latitude(**F("Vĩ độ", "Q")))

    color = alt.Color(**F("Tỷ lệ trễ", "Q"), scale=alt.Scale(range=["#F8C9C8", ACCENT, "#B3261E"]), legend=None)
    tip = [alt.Tooltip(**F("Khu vực", "N")), alt.Tooltip(**F("Tỷ lệ trễ", "Q"), format=".1%"),
           alt.Tooltip(**F(MONEY, "Q"), format=",.0f"), alt.Tooltip(**F("Số đơn", "Q"), format=",")]

    world = alt.Data(values=load_world(), format=alt.DataFormat(type="topojson", feature="countries"))
    land = alt.Chart(world).mark_geoshape(
        fill="#EDEDEB", stroke="#FFFFFF", strokeWidth=0.6).transform_filter("datum.id != 10")
    arcs = alt.Chart(lines).mark_line(strokeCap="round", opacity=.9).encode(
        **geo, detail=alt.Detail(**F("Khu vực", "N")), order=alt.Order(**F("Thứ tự điểm", "Q")), color=color,
        size=alt.Size(**F(MONEY, "Q"), scale=alt.Scale(range=[1.2, 6]), legend=None))
    dest = alt.Chart(reg).mark_circle(opacity=1, stroke="#FFFFFF", strokeWidth=1.5).encode(
        **geo, color=color, size=alt.Size(**F("Số đơn", "Q"), scale=alt.Scale(range=[30, 160]), legend=None),
        tooltip=tip)
    hub = pd.DataFrame([{"Điểm xuất phát": HUB["name"], "Vĩ độ": HUB["lat"], "Kinh độ": HUB["lon"]}])
    hub_dot = alt.Chart(hub).mark_circle(size=170, color=INK, opacity=1, stroke="#FFFFFF", strokeWidth=2).encode(
        **geo, tooltip=[alt.Tooltip(**F("Điểm xuất phát", "N"))])
    hub_txt = alt.Chart(hub).mark_text(align="left", dx=10, dy=12, fontSize=11, font="Inter", fontWeight=600,
                                       color=INK).encode(**geo, text=alt.Text(**F("Điểm xuất phát", "N")))
    layers = [land, arcs, dest, hub_dot, hub_txt]
    if len(top):
        t = top[["route_name", "order_region"]].copy()
        t["Vĩ độ"] = t["order_region"].map(lambda r: REGION_XY.get(r, (np.nan, np.nan))[0])
        t["Kinh độ"] = t["order_region"].map(lambda r: REGION_XY.get(r, (np.nan, np.nan))[1])
        k = t.groupby("order_region").cumcount()               # tách các số trùng một khu vực
        t["Vĩ độ"] = t["Vĩ độ"] + k * 9
        t["Kinh độ"] = t["Kinh độ"] - k * 9
        t["Thứ tự ưu tiên"] = [str(n) for n in range(1, len(t) + 1)]
        t = t.rename(columns={"route_name": "Tuyến"}).drop(columns="order_region")
        layers += [
            alt.Chart(t).mark_circle(size=420, color=ACCENT, opacity=1, stroke="#FFFFFF", strokeWidth=2.5).encode(
                **geo, tooltip=[alt.Tooltip(**F("Tuyến", "N")), alt.Tooltip(**F("Thứ tự ưu tiên", "N"))]),
            alt.Chart(t).mark_text(color="#FFFFFF", fontWeight=700, fontSize=12, font="Inter").encode(
                **geo, text=alt.Text(**F("Thứ tự ưu tiên", "N"))),
        ]
    chart = alt.layer(*layers).project(type="equalEarth").properties(height=400)
    return chart.configure_view(stroke=None)


def matrix_chart(d):
    """Ma trận ưu tiên: ngang = tỷ lệ trễ, dọc = tiền rủi ro kỳ vọng, chấm to = nhiều đơn."""
    d = d.dropna(subset=["actual_late_rate"])
    if d.empty:
        return None
    x_mid = weighted_rate(d)
    y_mid = float(d["expected_revenue_at_risk"].median())
    x_max = 1.04
    y_max = float(d["expected_revenue_at_risk"].max()) * 1.12
    d = d[["route_name", "tier_vi", "actual_late_rate", "predicted_late_rate", "expected_revenue_at_risk",
           "n_orders"]].rename(columns={
        "route_name": "Tuyến", "tier_vi": "Mức rủi ro", "actual_late_rate": "Tỷ lệ trễ thực tế",
        "predicted_late_rate": "Tỷ lệ trễ dự báo", "expected_revenue_at_risk": MONEY, "n_orders": "Số đơn"})
    d = d.round({"Tỷ lệ trễ thực tế": 4, "Tỷ lệ trễ dự báo": 4, MONEY: 0})
    top_lab = d.nlargest(3, MONEY)
    tip = [alt.Tooltip(**F(c, "N")) for c in ("Tuyến", "Mức rủi ro")] + [
        alt.Tooltip(**F("Tỷ lệ trễ thực tế", "Q"), format=".1%"),
        alt.Tooltip(**F("Tỷ lệ trễ dự báo", "Q"), format=".1%"),
        alt.Tooltip(**F(MONEY, "Q"), format=",.0f"), alt.Tooltip(**F("Số đơn", "Q"), format=",")]
    x = alt.X(**F("Tỷ lệ trễ thực tế", "Q"), title="Tỷ lệ giao trễ thực tế →", scale=alt.Scale(domain=[0, x_max]),
              axis=alt.Axis(format="%", tickCount=5))
    y = alt.Y(**F(MONEY, "Q"), title="Tiền rủi ro kỳ vọng (USD) →",
              scale=alt.Scale(domain=[0, y_max]), axis=alt.Axis(format="~s", tickCount=5))
    xq, yq = alt.X(**F("Tỷ lệ trễ thực tế", "Q")), alt.Y(**F(MONEY, "Q"))
    zone = alt.Chart(pd.DataFrame({"Từ tỷ lệ": [x_mid], "Đến tỷ lệ": [x_max], "Từ tiền": [y_mid], "Đến tiền": [y_max]})
                     ).mark_rect(color="#FDECEC", opacity=.7).encode(
        x=alt.X(**F("Từ tỷ lệ", "Q")), x2=alt.X2(field="Đến tỷ lệ"),
        y=alt.Y(**F("Từ tiền", "Q")), y2=alt.Y2(field="Đến tiền"))
    vline = alt.Chart(pd.DataFrame({"Tỷ lệ trễ chung": [x_mid]})).mark_rule(
        strokeDash=[4, 4], color="#BDBDBD").encode(x=alt.X(**F("Tỷ lệ trễ chung", "Q")))
    hline = alt.Chart(pd.DataFrame({"Trung vị tiền rủi ro": [y_mid]})).mark_rule(
        strokeDash=[4, 4], color="#BDBDBD").encode(y=alt.Y(**F("Trung vị tiền rủi ro", "Q")))
    corners = pd.DataFrame({"Vị trí x": [x_max - 0.01, 0.01], "Vị trí y": [y_max * 0.97, y_max * 0.97],
                            "Nhãn vùng": ["ƯU TIÊN XỬ LÝ", "ÍT TRỄ NHƯNG MẤT NHIỀU TIỀN"]})
    cpos = dict(x=alt.X(**F("Vị trí x", "Q")), y=alt.Y(**F("Vị trí y", "Q")), text=alt.Text(**F("Nhãn vùng", "N")))
    corner_r = alt.Chart(corners.iloc[[0]]).mark_text(align="right", baseline="middle", fontSize=11, font="Inter",
                                                      fontWeight=700, color=ACCENT).encode(**cpos)
    corner_l = alt.Chart(corners.iloc[[1]]).mark_text(align="left", baseline="middle", fontSize=11, font="Inter",
                                                      fontWeight=700, color=MUTED).encode(**cpos)
    dots = alt.Chart(d).mark_circle(opacity=.9, stroke="#FFFFFF", strokeWidth=1).encode(
        x=x, y=y, size=alt.Size(**F("Số đơn", "Q"), scale=alt.Scale(range=[30, 700]), legend=None),
        color=alt.Color(**F("Mức rủi ro", "N"), scale=alt.Scale(domain=list(TIER_VI_COLOR),
                        range=list(TIER_VI_COLOR.values())), legend=None), tooltip=tip)
    lab = dict(x=xq, y=yq, text=alt.Text(**F("Tuyến", "N")))
    lab_l = alt.Chart(top_lab[top_lab["Tỷ lệ trễ thực tế"] <= 0.7]).mark_text(
        align="left", dx=12, dy=-4, fontSize=11, font="Inter", color=INK).encode(**lab)
    lab_r = alt.Chart(top_lab[top_lab["Tỷ lệ trễ thực tế"] > 0.7]).mark_text(
        align="right", dx=-12, dy=-4, fontSize=11, font="Inter", color=INK).encode(**lab)
    return clean(alt.layer(zone, vline, hline, corner_r, corner_l, dots, lab_l, lab_r), 540)


def heatmap(d):
    """23 khu vực x 4 phương thức. Ô xám nhạt = tuyến thiếu dữ liệu."""
    reg_order = (d.groupby("order_region")
                  .apply(lambda g: g["n_late"].sum() / max(g["n_base"].sum(), 1), include_groups=False)
                  .sort_values(ascending=False).index.tolist())
    modes = [m for m in MODE_ORDER if m in set(d["shipping_mode"])]
    d = d[["route_name", "order_region", "shipping_mode", "tier_vi", "risk_tier", "actual_late_rate",
           "n_orders"]].rename(columns={
        "route_name": "Tuyến", "order_region": "Khu vực", "shipping_mode": "Phương thức", "tier_vi": "Mức rủi ro",
        "actual_late_rate": "Tỷ lệ trễ thực tế", "n_orders": "Số đơn"})
    d["Tỷ lệ trễ thực tế"] = d["Tỷ lệ trễ thực tế"].round(4)
    d["Nhãn"] = d["Tỷ lệ trễ thực tế"].map(lambda v: "" if pd.isna(v) else f"{v * 100:.0f}")
    d["Chữ trắng"] = d["Tỷ lệ trễ thực tế"] > 0.7
    d["Đủ dữ liệu"] = d["risk_tier"] != "Insufficient data"
    d = d.drop(columns="risk_tier")
    base = alt.Chart(d).encode(
        x=alt.X(**F("Phương thức", "N"), sort=modes, title=None,
                axis=alt.Axis(orient="top", labelAngle=0, labelColor=INK, labelFontSize=11.5,
                              labelExpr="replace(datum.label, ' Class', '')")),
        y=alt.Y(**F("Khu vực", "N"), sort=reg_order, title=None,
                axis=alt.Axis(labelColor=INK, labelFontSize=11.5, labelOverlap=False, labelLimit=140)))
    cells = base.mark_rect(cornerRadius=5, stroke="#FFFFFF", strokeWidth=2).encode(
        color=alt.condition("datum['Đủ dữ liệu']", alt.Color(**F("Tỷ lệ trễ thực tế", "Q"), legend=None,
                            scale=alt.Scale(domain=[0.3, 1], range=["#FFF1F0", ACCENT, "#9F1D17"])),
                            alt.value("#F3F3F3")),
        tooltip=[alt.Tooltip(**F("Tuyến", "N")), alt.Tooltip(**F("Mức rủi ro", "N")),
                 alt.Tooltip(**F("Tỷ lệ trễ thực tế", "Q"), format=".1%"),
                 alt.Tooltip(**F("Số đơn", "Q"), format=",")])
    text = base.mark_text(fontSize=10.5, font="Inter", fontWeight=500).encode(
        text=alt.Text(**F("Nhãn", "N")),
        color=alt.condition("datum['Chữ trắng']", alt.value("#FFFFFF"), alt.value(INK)))
    return clean(alt.layer(cells, text), 23 * 23).configure_axisY(grid=False)


# ============================================================ 4. GHÉP TRANG
def table_html(t):
    """Bảng xếp hạng dạng HTML để chỉnh được màu chữ tiêu đề, thanh % và nhãn mức rủi ro."""
    def bar(v, vmax, txt):
        if pd.isna(v):
            return '<span class="tb-muted">-</span>'
        w = max(0.0, min(1.0, v / vmax)) * 100
        return (f'<div class="tb-bar"><div class="tb-track"><div class="tb-fill" style="width:{w:.0f}%"></div></div>'
                f'<span>{txt}</span></div>')
    head = ("<tr><th>Hạng</th><th>Tuyến</th><th>Mức rủi ro</th><th class='r'>Số đơn</th><th>Trễ thực tế</th>"
            "<th class='r'>Trễ dự báo</th><th class='r'>Tiền rủi ro kỳ vọng (USD)</th><th>Điểm rủi ro</th></tr>")
    body = []
    for r in t.itertuples():
        hang = "-" if pd.isna(r.hang) else f"{int(r.hang)}"
        score = "-" if pd.isna(r.risk_score) else f"{r.risk_score:.2f}".replace(".", ",")
        body.append(
            f"<tr><td class='tb-rank'>{hang}</td><td class='tb-name'>{r.route_name}</td>"
            f"<td><span class='tier-pill tier-{r.risk_tier.split()[0]}'>{r.tier_vi}</span></td>"
            f"<td class='r'>{vn_int(r.n_orders)}</td>"
            f"<td>{bar(r.actual_late_rate, 1, vn_pct(r.actual_late_rate))}</td>"
            f"<td class='r'>{vn_pct(r.predicted_late_rate)}</td>"
            f"<td class='r'>{vn_int(r.expected_revenue_at_risk)}</td>"
            f"<td>{bar(r.risk_score, 1, score)}</td></tr>")
    return f'<div class="tb-wrap"><table class="tb">{head}{"".join(body)}</table></div>'



def alert_html(top):
    rows = []
    for i, r in enumerate(top.itertuples(), 1):
        rows.append(
            f'<div class="alert-row"><div class="alert-num">{i}</div><div class="alert-body">'
            f'<div class="alert-name">{r.order_region} - {r.shipping_mode}</div>'
            f'<div class="alert-meta">Trễ {vn_pct(r.actual_late_rate)} - {money_txt(r.expected_revenue_at_risk)} rủi ro</div>'
            f'</div><span class="tier-pill tier-{r.risk_tier}">{r.tier_vi}</span></div>')
    return "".join(rows)


def page_ranking():
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    _, region_market = load_mode_days()
    rr = prepare(load_route_risk(), region_market)

    # ---------- Bộ lọc ở thanh bên
    with st.sidebar:
        st.markdown('<div class="card-title" style="margin-top:18px">Bộ lọc</div>', unsafe_allow_html=True)
        markets = st.multiselect("Thị trường", sorted(rr["market"].dropna().unique()), placeholder="Tất cả thị trường")
        tiers = st.multiselect("Mức rủi ro", TIER_ORDER, format_func=lambda t: TIER_VI[t], placeholder="Tất cả mức")
    d_all = rr[rr["market"].isin(markets)] if markets else rr
    if tiers:
        d_all = d_all[d_all["risk_tier"].isin(tiers)]

    # ---------- Tiêu đề + chọn phương thức
    left, right = st.columns([1.5, 2], vertical_alignment="bottom")
    left.markdown('<div class="ov-title">Xếp hạng rủi ro tuyến</div>',
                  unsafe_allow_html=True)
    with right:
        mode = st.segmented_control("Phương thức", ["Tất cả"] + MODE_ORDER, default="Tất cả",
                                    label_visibility="collapsed", key="rank_mode") or "Tất cả"
    st.write("")
    d = d_all if mode == "Tất cả" else d_all[d_all["shipping_mode"] == mode]
    if d.empty:
        st.info("Không có tuyến nào khớp bộ lọc.")
        return
    ranked = d.dropna(subset=["risk_score"]).sort_values("risk_score", ascending=False)
    top = ranked.head(3)

    # ---------- Hàng 1: thẻ chỉ số
    high = d[d["risk_tier"] == "High"]
    share_orders = high["n_orders"].sum() / d["n_orders"].sum() if d["n_orders"].sum() else np.nan
    share_money = high["expected_revenue_at_risk"].sum() / d["expected_revenue_at_risk"].sum() \
        if d["expected_revenue_at_risk"].sum() else np.nan
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        kpi_card("card_rk1", "box", "Số tuyến", vn_int(len(d)),
                 f"{vn_int(len(ranked))} tuyến đủ dữ liệu xếp hạng", True, None, None)
    with c2:
        kpi_card("card_rk2", "late", "Tuyến mức Cao", vn_int(len(high)),
                 f"chiếm {vn_pct(share_orders, 0)} số đơn", False, None, None)
    with c3:
        kpi_card("card_rk3", "money", "Tiền rủi ro mức Cao",
                 money_txt(high["expected_revenue_at_risk"].sum()).replace(" triệu USD", "<small>triệu USD</small>"),
                 f"{vn_pct(share_money, 0)} tổng tiền rủi ro kỳ vọng", False, None, None)
    with c4:
        kpi_card("card_rk4", "delay", "Tỷ lệ giao trễ", vn_pct(weighted_rate(d), 1),
                 f"mô hình dự báo {vn_pct(weighted_rate(d, 'n_pred'), 1)}", True, None, None)

    # ---------- Hàng 2: bản đồ tuyến + cảnh báo
    a, b = st.columns([2.1, 1], gap="medium")
    with a:
        with st.container(key="card_map"):
            card_header(f"Bản đồ tuyến giao hàng - {mode}")
            st.altair_chart(route_map(by_region(d), top), width="stretch")
    with b:
        with st.container(key="card_alert"):
            card_header("Cảnh báo ưu tiên")
            if top.empty:
                st.caption("Các tuyến đang chọn không đủ dữ liệu để xếp hạng.")
            else:
                st.markdown(
                    f'<div class="alert-big"><b>{money_txt(top["expected_revenue_at_risk"].sum())}</b>'
                    f'tiền rủi ro kỳ vọng ở 3 tuyến này</div>' + alert_html(top) +
                    '<div class="alert-foot">Điểm rủi ro gồm 50% tỷ lệ trễ, 30% số ngày trễ '
                    'và 20% doanh thu bị ảnh hưởng.</div>', unsafe_allow_html=True)

    # ---------- Hàng 3: ma trận ưu tiên + bản đồ nhiệt
    x, y = st.columns([1.45, 1], gap="medium")
    with x:
        with st.container(key="card_matrix"):
            card_header("Ma trận ưu tiên")
            m = matrix_chart(d)
            if m is not None:
                st.altair_chart(m, width="stretch")
    with y:
        with st.container(key="card_heat"):
            card_header("Bản đồ nhiệt các tuyến")
            st.altair_chart(heatmap(d_all), width="stretch")

    # ---------- Hàng 4: bảng xếp hạng
    with st.container(key="card_table"):
        h1, h2 = st.columns([2, 1.4], vertical_alignment="center")
        with h1:
            card_header("Bảng xếp hạng chi tiết")
        with h2:
            sort_by = st.segmented_control("Xếp theo", ["Điểm rủi ro", "Tiền rủi ro", "Trong phương thức"],
                                           default="Điểm rủi ro", label_visibility="collapsed", key="rank_sort") \
                or "Điểm rủi ro"
        t = d.copy()
        if sort_by == "Tiền rủi ro":
            t = t.sort_values("expected_revenue_at_risk", ascending=False)
            t["hang"] = range(1, len(t) + 1)
        elif sort_by == "Trong phương thức":
            t["mode_order"] = t["shipping_mode"].map({m: i for i, m in enumerate(MODE_ORDER)})
            t = t.sort_values(["mode_order", "rank_in_mode"])
            t["hang"] = t["rank_in_mode"]
        else:
            t = t.sort_values("risk_score", ascending=False, na_position="last")
            t["hang"] = t["risk_rank"]
        st.markdown(table_html(t), unsafe_allow_html=True)
        show = t[["hang", "route_name", "tier_vi", "n_orders", "actual_late_rate", "predicted_late_rate",
                  "expected_revenue_at_risk", "risk_score"]]
        st.download_button("Tải bảng (CSV)", show.to_csv(index=False).encode("utf-8-sig"),
                           "xep_hang_tuyen.csv", "text/csv", icon=":material/download:")
