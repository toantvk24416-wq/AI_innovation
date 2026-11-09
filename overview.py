"""Trang Tổng quan hiệu suất giao hàng.

Cấu trúc file (đọc từ trên xuống):
  1. Màu sắc và CSS  -> quyết định "giao diện" (thẻ trắng bo góc, màu nhấn đỏ)
  2. Lấy dữ liệu     -> từ BigQuery (nếu đã cấu hình khóa) hoặc từ file CSV
  3. Hàm tính chỉ số -> tỷ lệ trễ, trễ trung bình... tính ở cấp ĐƠN HÀNG
  4. Hàm vẽ biểu đồ  -> mỗi hàm trả về một biểu đồ Altair
  5. page_overview() -> ghép tất cả thành trang
"""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent

# ============================================================ 1. MÀU SẮC VÀ CSS
ACCENT = "#EF4B4B"   # đỏ san hô: màu nhấn chính
SOFT = "#FAD2D1"     # đỏ nhạt
INK = "#1F1F1F"      # chữ chính
MUTED = "#8C8C8C"    # chữ phụ
LINE = "#E6E6E6"     # đường kẻ, cột nền
BG = "#F4F4F2"       # nền trang

TIER_ORDER = ["High", "Medium", "Low", "Insufficient data"]
TIER_VI = {"High": "Cao", "Medium": "Trung bình", "Low": "Thấp", "Insufficient data": "Thiếu dữ liệu"}
TIER_COLOR = {"High": ACCENT, "Medium": "#F5A3A1", "Low": "#CFCFCF", "Insufficient data": "#EDEDED"}
MODE_ORDER = ["First Class", "Second Class", "Same Day", "Standard Class"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp :is(h1, h2, h3, h4, p, label, li, input, textarea),
.stApp [data-testid="stMarkdownContainer"] {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: {BG}; }}
[data-testid="stSidebar"] {{ background: #FFFFFF; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1440px; }}

/* Thẻ trắng: mọi st.container(key="card_...") */
[class*="st-key-card"] {{
    background: #FFFFFF; border-radius: 20px; padding: 18px 20px 14px 20px;
    box-shadow: 0 1px 2px rgba(0,0,0,.04);
}}
[class*="st-key-card_kpi"] {{ min-height: 178px; }}
[class*="st-key-card_trend"] {{ min-height: 372px; }}
/* Thẻ đỏ nổi bật */
.st-key-accent_insight {{
    background: {ACCENT}; border-radius: 20px; padding: 22px 24px; min-height: 400px;
}}
.st-key-accent_insight * {{ color: #FFFFFF !important; }}

/* Thẻ xe tải */
[class*="st-key-card_truck"] {{ min-height: 372px; }}
.st-key-card_mode, .st-key-card_market {{ min-height: 400px; }}
.truck-big  {{ font-size:15px; color:{MUTED}; margin-top:6px; line-height:1.5; }}
.truck-big b {{ color:{INK}; }}
.truck-stats {{ display:flex; gap:10px; margin-top:12px; }}
.truck-stat {{ flex:1; background:{BG}; border-radius:14px; padding:10px 12px; }}
.truck-stat .v {{ font-size:18px; font-weight:600; color:{INK}; }}
.truck-stat .l {{ font-size:12px; color:{MUTED}; }}
.truck-stat.red {{ background:#FDECEC; }}
.truck-stat.red .v {{ color:{ACCENT}; }}

/* Menu bên trái: mỗi trang là một thanh có icon */
[class*="st-key-nav_"] {{ margin-bottom:-10px; }}
[class*="st-key-nav_"] button {{
    justify-content:flex-start; border:none; border-radius:12px; padding:10px 14px;
    background:transparent; color:{INK}; box-shadow:none; min-height:44px;
}}
[class*="st-key-nav_"] button > div {{ justify-content:flex-start; width:100%; }}
[class*="st-key-nav_"] button p {{ font-size:15px; font-weight:500; }}
[class*="st-key-nav_"] button:hover {{ background:{BG}; color:{INK}; border:none; }}
[class*="st-key-nav_"] button:focus:not(:active) {{ border:none; color:inherit; }}
[class*="st-key-nav_"] button[kind="primary"],
[class*="st-key-nav_"] button[kind="primary"]:hover {{ background:{ACCENT}; color:#FFFFFF; }}
[class*="st-key-nav_"] button[kind="primary"] * {{ color:#FFFFFF; }}
.nav-group {{ font-size:12px; color:{MUTED}; text-transform:uppercase; letter-spacing:.6px; margin:6px 0 10px 4px; }}

/* Tiêu đề trang */
.ov-title {{ font-size: 32px; line-height:1.15; font-weight: 600; color: {INK}; letter-spacing: -.5px; margin: 0; }}
.ov-sub   {{ color: {MUTED}; font-size: 14px; margin-top: 2px; }}
.ov-chip  {{ display:inline-block; background:#fff; border-radius:999px; padding:6px 14px;
             font-size:13px; color:{INK}; margin-left:6px; box-shadow:0 1px 2px rgba(0,0,0,.05); }}

/* Thẻ chỉ số (KPI) */
.kpi-head  {{ display:flex; align-items:center; gap:10px; }}
.kpi-icon  {{ width:34px; height:34px; border-radius:50%; background:{BG};
              display:flex; align-items:center; justify-content:center; }}
.kpi-label {{ color:{MUTED}; font-size:14px; }}
.kpi-value {{ font-size:34px; font-weight:600; color:{INK}; letter-spacing:-1px; margin:8px 0 2px 0; line-height:1.1; }}
.kpi-value small {{ font-size:16px; font-weight:500; color:{MUTED}; margin-left:2px; }}
.kpi-pill  {{ display:inline-block; font-size:12px; padding:2px 10px; border-radius:999px;
              background:#FDECEC; color:{ACCENT}; font-weight:500; }}
.kpi-pill.grey {{ background:{BG}; color:{MUTED}; }}

/* Tiêu đề thẻ */
.card-title {{ font-size:17px; font-weight:600; color:{INK}; margin-bottom:0; }}
.card-note  {{ font-size:12.5px; color:{MUTED}; margin-top:2px; }}

/* Thẻ đỏ "Điểm chính" */
.ins-kicker {{ font-size:17px; font-weight:600; opacity:.95; }}
.ins-big    {{ font-size:56px; font-weight:600; letter-spacing:-2px; line-height:1; margin:14px 0 6px 0; }}
.ins-big small {{ font-size:28px; letter-spacing:0; }}
.ins-lead   {{ font-size:15px; font-weight:500; opacity:.95; margin-bottom:18px; }}
.ins-item   {{ border-top:1px solid rgba(255,255,255,.35); padding:10px 0 4px 0; font-size:14px; line-height:1.45; }}
.ins-item b {{ font-weight:600; }}
.ins-row  {{ display:flex; align-items:baseline; gap:12px; border-top:1px solid rgba(255,255,255,.3); padding:12px 0 8px 0; }}
.ins-num  {{ font-size:24px; font-weight:600; min-width:84px; letter-spacing:-.5px; }}
.ins-lab  {{ font-size:13.5px; opacity:.9; }}

/* Chú thích donut */
.legend-row {{ display:flex; align-items:center; justify-content:space-between; font-size:13.5px;
               padding:6px 0; border-bottom:1px solid #F1F1F1; color:{INK}; }}
.legend-dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px; }}

/* Thanh bên */
.brand {{ display:flex; align-items:center; gap:10px; margin:4px 0 18px 0; }}
.brand-logo {{ width:38px; height:38px; border-radius:12px; background:{ACCENT}; color:#fff;
               display:flex; align-items:center; justify-content:center; font-weight:700; }}
.brand-name {{ font-weight:600; color:{INK}; line-height:1.15; }}
.brand-sub  {{ font-size:12px; color:{MUTED}; }}
</style>
"""

ICONS = {
    "box": '<path d="M3 7.5 12 3l9 4.5-9 4.5-9-4.5Zm0 0V17l9 4.5 9-4.5V7.5M12 12v9.5"/>',
    "late": '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
    "delay": '<rect x="3.5" y="5" width="17" height="15" rx="3"/><path d="M3.5 10h17M8 3v4M16 3v4"/>',
    "money": '<circle cx="12" cy="12" r="8.5"/><path d="M14.8 9.2c-.5-.9-1.6-1.4-2.8-1.4-1.6 0-2.8.8-2.8 2.1 0 2.9 5.8 1.5 5.8 4.3 0 1.3-1.3 2.1-3 2.1-1.3 0-2.5-.6-3-1.6M12 6v1.8M12 16.3V18"/>',
}


def icon(name):
    return (f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{INK}" '
            f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')


def apply_style():
    """Gọi một lần ở đầu app để áp dụng giao diện cho mọi trang."""
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="brand"><div class="brand-logo">DC</div>'
        '<div><div class="brand-name">DataCo Delivery Risk</div>'
        '<div class="brand-sub">Đánh giá rủi ro giao hàng</div></div></div>',
        unsafe_allow_html=True)


def sidebar_nav(pages):
    """Menu dạng thanh có icon. pages = {"Tên trang": (":material/icon:", hàm_vẽ_trang)}."""
    names = list(pages)
    if st.session_state.get("page") not in names:
        st.session_state["page"] = names[0]
    st.sidebar.markdown('<div class="nav-group">Menu</div>', unsafe_allow_html=True)
    for i, name in enumerate(names):
        st.sidebar.button(name, key=f"nav_{i}", icon=pages[name][0], width="stretch",
                          type="primary" if st.session_state["page"] == name else "secondary",
                          on_click=st.session_state.update, kwargs={"page": name})
    return pages[st.session_state["page"]][1]


def truck_svg(rate):
    """Xe tải: thùng hàng được tô đỏ theo tỷ lệ giao trễ (rate từ 0 đến 1)."""
    rate = 0 if pd.isna(rate) else max(0.0, min(1.0, float(rate)))
    x0, w = 14, 300                      # vị trí và chiều dài thùng hàng
    fill_w = w * rate
    label = vn_pct(rate, 1)
    return f"""
<svg viewBox="0 0 440 190" width="100%" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <pattern id="stripe" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
      <rect width="10" height="10" fill="{ACCENT}"/><rect width="4" height="10" fill="#F36F6F"/>
    </pattern>
    <clipPath id="box"><rect x="{x0}" y="18" width="{w}" height="118" rx="10"/></clipPath>
    <clipPath id="filled"><rect x="{x0}" y="18" width="{fill_w:.1f}" height="118"/></clipPath>
    <clipPath id="empty"><rect x="{x0 + fill_w:.1f}" y="18" width="{w - fill_w:.1f}" height="118"/></clipPath>
    <linearGradient id="cab" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#E9E9E9"/>
    </linearGradient>
  </defs>
  <ellipse cx="220" cy="176" rx="205" ry="7" fill="#000" opacity=".06"/>
  <!-- thùng hàng -->
  <rect x="{x0}" y="18" width="{w}" height="118" rx="10" fill="#EFEFEF" stroke="#DADADA"/>
  <g clip-path="url(#box)"><rect x="{x0}" y="18" width="{fill_w:.1f}" height="118" fill="url(#stripe)"/></g>
  <!-- chữ % đổi màu theo nền: trắng trên phần đỏ, đen trên phần xám -->
  <text x="{x0 + w / 2}" y="92" text-anchor="middle" font-family="Inter, sans-serif" font-size="40"
        font-weight="700" fill="#FFFFFF" clip-path="url(#filled)">{label}</text>
  <text x="{x0 + w / 2}" y="92" text-anchor="middle" font-family="Inter, sans-serif" font-size="40"
        font-weight="700" fill="{INK}" clip-path="url(#empty)">{label}</text>
  <rect x="{x0}" y="136" width="{w + 6}" height="10" rx="3" fill="#BDBDBD"/>
  <!-- cabin -->
  <path d="M326 52 h56 q10 0 16 9 l26 38 q4 6 4 13 v34 h-102 z" fill="url(#cab)" stroke="#CFCFCF"/>
  <path d="M342 62 h38 q6 0 9 5 l18 28 h-65 z" fill="#C9D3DC"/>
  <rect x="420" y="118" width="12" height="10" rx="2" fill="#FFC857"/>
  <rect x="326" y="128" width="104" height="18" rx="4" fill="#9E9E9E"/>
  <!-- bánh xe -->
  <g fill="#3A3A3A"><circle cx="70" cy="152" r="18"/><circle cx="112" cy="152" r="18"/>
     <circle cx="262" cy="152" r="18"/><circle cx="380" cy="152" r="18"/></g>
  <g fill="#BDBDBD"><circle cx="70" cy="152" r="7"/><circle cx="112" cy="152" r="7"/>
     <circle cx="262" cy="152" r="7"/><circle cx="380" cy="152" r="7"/></g>
</svg>"""


# ============================================================ 2. LẤY DỮ LIỆU
# Câu SQL lấy dữ liệu cho trang Tổng quan từ tầng Gold.
# Gộp về cấp ĐƠN HÀNG trước (giống route_scorecard), rồi tổng hợp theo
# tháng x thị trường x khu vực x phương thức. Kết quả chỉ vài nghìn dòng.
OVERVIEW_SQL = """
WITH orders AS (
  SELECT
    f.order_id,
    d.year_month,
    m.market,
    r.order_region,
    r.shipping_mode,
    MAX(f.late_delivery_risk)  AS is_late,
    LOGICAL_OR(f.is_canceled)  AS is_canceled,
    AVG(f.delay_days)          AS delay_days,
    SUM(f.order_item_total)    AS revenue
  FROM `dw-dataco-513010.gold.fact_order_items` AS f
  JOIN `dw-dataco-513010.gold.dim_route`  AS r USING (route_key)
  JOIN `dw-dataco-513010.gold.dim_market` AS m USING (market_key)
  JOIN `dw-dataco-513010.gold.dim_date`   AS d ON d.date_key = f.order_date_key
  GROUP BY 1, 2, 3, 4, 5
)
SELECT
  year_month, market, order_region, shipping_mode,
  COUNT(*)                                  AS n_orders,
  COUNTIF(NOT is_canceled)                  AS n_not_canceled,
  COUNTIF(is_late = 1)                      AS n_late,
  COUNTIF(is_canceled)                      AS n_canceled,
  SUM(IF(NOT is_canceled, delay_days, 0))   AS sum_delay_days,
  SUM(revenue)                              AS revenue,
  SUM(IF(is_late = 1, revenue, 0))          AS revenue_at_risk
FROM orders
GROUP BY 1, 2, 3, 4
ORDER BY 1, 2, 3, 4
"""

SUM_COLS = ["n_orders", "n_not_canceled", "n_late", "n_canceled", "sum_delay_days", "revenue", "revenue_at_risk"]


def _bigquery_client():
    """Trả về client BigQuery nếu đã khai báo khóa trong secrets, ngược lại trả về None."""
    try:
        if "gcp_service_account" not in st.secrets:
            return None
    except Exception:          # chưa có file secrets.toml
        return None
    from google.cloud import bigquery
    from google.oauth2 import service_account
    try:
        info = dict(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(info)
        return bigquery.Client(credentials=creds, project=info["project_id"])
    except Exception as e:     # khóa dán sai hoặc còn giá trị mẫu: dùng CSV thay vì làm sập cả trang
        print(f"Khóa BigQuery trong secrets không dùng được ({type(e).__name__}: {e}). Chuyển sang dữ liệu CSV.")
        return None


def _query_or_csv(sql, csv_name):
    """Đọc từ BigQuery nếu có khóa; khóa sai, thiếu quyền hoặc lỗi mạng thì đọc file CSV đi kèm app."""
    client = _bigquery_client()
    if client is not None:
        try:
            return client.query(sql).to_dataframe(create_bqstorage_client=False), "BigQuery"
        except Exception as e:
            print(f"Truy vấn BigQuery lỗi ({type(e).__name__}: {str(e)[:300]}). Chuyển sang {csv_name}.")
    return pd.read_csv(BASE / csv_name), "CSV"


@st.cache_data(ttl=600, show_spinner="Đang tải dữ liệu từ tầng Gold...")
def load_overview():
    df, source = _query_or_csv(OVERVIEW_SQL, "overview_monthly.csv")
    df["year_month"] = df["year_month"].astype(str)
    return df, source


@st.cache_data(ttl=600, show_spinner="Đang tải bảng xếp hạng tuyến...")
def load_route_risk():
    return _query_or_csv("SELECT * FROM `dw-dataco-513010.gold.route_risk_expected`", "route_risk.csv")[0]


@st.cache_data
def load_mode_days():
    opt = json.loads((BASE / "options.json").read_text(encoding="utf-8"))
    return opt.get("mode_days", {}), opt.get("region_market", {})


# ============================================================ 3. TÍNH CHỈ SỐ
def summarize(df, by=None):
    """Cộng các cột đếm rồi mới chia -> tỷ lệ đúng ở cấp đơn hàng (không lấy trung bình của tỷ lệ)."""
    s = df[SUM_COLS].sum().to_frame().T if by is None else df.groupby(by, as_index=False)[SUM_COLS].sum()
    nc = s["n_not_canceled"].where(s["n_not_canceled"] > 0)
    s["late_rate"] = s["n_late"] / nc
    s["avg_delay"] = s["sum_delay_days"] / nc
    s["cancel_rate"] = s["n_canceled"] / s["n_orders"]
    s["risk_share"] = s["revenue_at_risk"] / s["revenue"].where(s["revenue"] > 0)
    return s


def vn_int(x):
    return f"{x:,.0f}".replace(",", ".")


def vn_pct(x, d=1):
    return "-" if pd.isna(x) else f"{x * 100:.{d}f}".replace(".", ",") + "%"


def vn_num(x, d=2):
    return "-" if pd.isna(x) else f"{x:+.{d}f}".replace(".", ",")


def vn_money(x):
    if x >= 1e6:
        return f"{x / 1e6:.1f}".replace(".", ",") + "<small>triệu USD</small>"
    return vn_int(x)


def ym_label(ym):
    y, m = ym.split("-")
    return f"{m}/{y}"


# ============================================================ 4. BIỂU ĐỒ
def clean(chart, height):
    """Phong cách chung cho mọi biểu đồ: không khung, lưới mờ, chữ xám."""
    return (chart.properties(height=height)
            .configure_view(stroke=None)
            .configure_axis(grid=False, domain=False, ticks=False, labelColor=MUTED,
                            labelFont="Inter", titleColor=MUTED, titleFont="Inter",
                            labelFontSize=11.5, titleFontSize=11.5, titleFontWeight=400)
            .configure_axisY(grid=True, gridColor="#F0F0F0")
            .configure_legend(disable=True))


def F(name, kind):
    """Trường dữ liệu theo tên tiếng Việt (có dấu, có khoảng trắng). kind: Q số, N chữ, T thời gian."""
    return {"field": name, "type": {"Q": "quantitative", "N": "nominal", "T": "temporal"}[kind]}


def spark(df, field, label, highlight_last=True):
    """Cột nhỏ trong thẻ KPI (12 tháng gần nhất của khoảng đang lọc)."""
    d = df.tail(12)[["year_month", field]].rename(columns={"year_month": "Tháng", field: label}).reset_index(drop=True)
    d[label] = d[label].round(4)
    d["Tháng gần nhất"] = d.index == len(d) - 1
    return clean(alt.Chart(d).mark_bar(cornerRadius=3, size=9).encode(
        x=alt.X(**F("Tháng", "N"), axis=None, sort=None),
        y=alt.Y(**F(label, "Q"), axis=None, scale=alt.Scale(zero=False, padding=4)),
        color=alt.condition("datum['Tháng gần nhất']" if highlight_last else "false",
                            alt.value(ACCENT), alt.value(SOFT)),
        tooltip=[alt.Tooltip(**F("Tháng", "N")), alt.Tooltip(**F(label, "Q"), format=",.3~f")],
    ), 46)


def to_period(monthly, by_quarter):
    """Gộp dữ liệu theo tháng hoặc theo quý (cộng số đơn rồi mới chia)."""
    d = monthly.copy()
    d["date"] = pd.to_datetime(d["year_month"] + "-01")
    if by_quarter:
        d["date"] = d["date"].dt.to_period("Q").dt.start_time
    g = d.groupby("date", as_index=False)[SUM_COLS].sum()
    g["late_rate"] = g["n_late"] / g["n_not_canceled"].where(g["n_not_canceled"] > 0)
    return g


def trend_chart(d, by_quarter):
    """Đường tỷ lệ trễ: vùng tô mờ, chấm tròn, các vạch đỏ đứt nét thả xuống trục."""
    d = d.dropna(subset=["late_rate"])
    lo = max(0.0, round(d["late_rate"].min() - 0.10, 1))
    hi = min(1.0, round(d["late_rate"].max() + 0.05, 1))
    d = d[["date", "late_rate", "n_orders"]].rename(
        columns={"date": "Kỳ", "late_rate": "Tỷ lệ trễ", "n_orders": "Số đơn"}).reset_index(drop=True)
    d["Tỷ lệ trễ"] = d["Tỷ lệ trễ"].round(4)
    d["Đáy trục"] = lo
    d["Cao nhất"] = d.index == d["Tỷ lệ trễ"].idxmax()
    d["Nhãn"] = d["Tỷ lệ trễ"].map(lambda v: vn_pct(v))
    fmt = "Q%q/%y" if by_quarter else "%m/%y"
    x = alt.X(**F("Kỳ", "T"), title=None,
              axis=alt.Axis(format=fmt, labelAngle=0, tickCount=8 if not by_quarter else len(d)))
    y = alt.Y(**F("Tỷ lệ trễ", "Q"), title=None, axis=alt.Axis(format="%", tickCount=5),
              scale=alt.Scale(domain=[lo, hi]))
    yy = alt.Y(**F("Tỷ lệ trễ", "Q"))
    tip = [alt.Tooltip(**F("Kỳ", "T"), format="%m/%Y"),
           alt.Tooltip(**F("Tỷ lệ trễ", "Q"), format=".1%"),
           alt.Tooltip(**F("Số đơn", "Q"), format=",")]
    base = alt.Chart(d).encode(x=x)
    area = base.mark_area(interpolate="monotone", line=False, color=alt.Gradient(
        gradient="linear", x1=1, x2=1, y1=1, y2=0,
        stops=[alt.GradientStop(color="#FFFFFF", offset=0), alt.GradientStop(color="#FBD5D4", offset=1)])
    ).encode(y=y, y2=alt.Y2(field="Đáy trục"))
    drops = base.mark_rule(strokeDash=[3, 3], color=ACCENT, opacity=.45, strokeWidth=1).encode(
        y=yy, y2=alt.Y2(field="Đáy trục"))
    line = base.mark_line(interpolate="monotone", color=ACCENT, strokeWidth=2.5).encode(y=yy)
    dots = base.mark_circle(size=55 if not by_quarter else 90, color=ACCENT, opacity=1,
                            stroke="#FFFFFF", strokeWidth=2).encode(y=yy, tooltip=tip)
    halo = base.transform_filter("datum['Cao nhất']").mark_circle(size=380, color=ACCENT, opacity=.18).encode(y=yy)
    tag = base.transform_filter("datum['Cao nhất']").mark_text(
        dy=-18, color=ACCENT, fontWeight=600, font="Inter", fontSize=12).encode(y=yy, text=alt.Text(**F("Nhãn", "N")))
    return clean(alt.layer(area, drops, line, halo, dots, tag), 280)


def mode_chart(s, mode_days):
    d = s[["shipping_mode", "late_rate", "avg_delay", "n_orders"]].rename(columns={
        "shipping_mode": "Phương thức", "late_rate": "Tỷ lệ trễ", "avg_delay": "Trễ trung bình (ngày)",
        "n_orders": "Số đơn"})
    d = d.round({"Tỷ lệ trễ": 4, "Trễ trung bình (ngày)": 2})
    d["Phương thức và cam kết"] = d["Phương thức"] + d["Phương thức"].map(
        lambda x: f" - cam kết {mode_days[x]} ngày" if x in mode_days else "")
    d["Cao nhất"] = d["Tỷ lệ trễ"] == d["Tỷ lệ trễ"].max()
    d["Nhãn"] = d["Tỷ lệ trễ"].map(vn_pct)
    order = d.sort_values("Tỷ lệ trễ", ascending=False)["Phương thức và cam kết"].tolist()
    base = alt.Chart(d).encode(y=alt.Y(**F("Phương thức và cam kết", "N"), sort=order, title=None,
                                       axis=alt.Axis(labelLimit=260, labelFontSize=12.5, labelColor=INK, labelPadding=8)))
    track = base.mark_bar(cornerRadius=9, size=18, color="#F3F3F3").encode(
        x=alt.X(field="nen", type="quantitative", axis=None)).transform_calculate(nen="1")
    bars = base.mark_bar(cornerRadius=9, size=18).encode(
        x=alt.X(**F("Tỷ lệ trễ", "Q"), axis=None, scale=alt.Scale(domain=[0, 1])),
        color=alt.condition("datum['Cao nhất']", alt.value(ACCENT), alt.value("#F2A6A4")),
        tooltip=[alt.Tooltip(**F("Phương thức", "N")),
                 alt.Tooltip(**F("Tỷ lệ trễ", "Q"), format=".1%"),
                 alt.Tooltip(**F("Trễ trung bình (ngày)", "Q"), format="+.2f"),
                 alt.Tooltip(**F("Số đơn", "Q"), format=",")])
    text = base.mark_text(align="right", dx=-8, color="#FFFFFF", fontWeight=600, font="Inter", fontSize=12).encode(
        x=alt.X(**F("Tỷ lệ trễ", "Q")), text=alt.Text(**F("Nhãn", "N")))
    return clean(alt.layer(track, bars, text), 300)


def market_chart(s):
    d = s.sort_values("revenue_at_risk", ascending=False)[["market", "revenue_at_risk", "late_rate"]].rename(
        columns={"market": "Thị trường", "revenue_at_risk": "Doanh thu chịu rủi ro (USD)", "late_rate": "Tỷ lệ trễ"})
    money = "Doanh thu chịu rủi ro (USD)"
    d = d.round({money: 0, "Tỷ lệ trễ": 4})
    total = d[money].sum()
    d["Tỷ trọng"] = d[money] / total if total else 0
    d["Tỷ trọng"] = d["Tỷ trọng"].round(4)
    d["Nhãn"] = d["Tỷ trọng"].map(lambda x: vn_pct(x, 0))
    d["Cao nhất"] = d[money] == d[money].max()
    base = alt.Chart(d).encode(y=alt.Y(**F("Thị trường", "N"), sort=d["Thị trường"].tolist(), title=None,
                                       axis=alt.Axis(labelColor=INK, labelFontSize=12.5, labelPadding=8)))
    bars = base.mark_bar(cornerRadius=9, size=18).encode(
        x=alt.X(**F(money, "Q"), axis=None, scale=alt.Scale(domain=[0, d[money].max() * 1.18])),
        color=alt.condition("datum['Cao nhất']", alt.value(ACCENT), alt.value("#F2A6A4")),
        tooltip=[alt.Tooltip(**F("Thị trường", "N")),
                 alt.Tooltip(**F(money, "Q"), format=",.0f"),
                 alt.Tooltip(**F("Tỷ lệ trễ", "Q"), format=".1%")])
    text = base.mark_text(align="left", dx=6, color=INK, fontSize=12, font="Inter", fontWeight=600).encode(
        x=alt.X(**F(money, "Q")), text=alt.Text(**F("Nhãn", "N")))
    return clean(alt.layer(bars, text), 300)


def tier_donut(t):
    center = int(t.loc[t["risk_tier"] == "High", "n_routes"].sum())
    arc = alt.Chart(t).mark_arc(innerRadius=58, outerRadius=84, cornerRadius=4, padAngle=0.02).encode(
        theta=alt.Theta("n_routes:Q", stack=True),
        order=alt.Order("order:Q"),
        color=alt.Color("risk_tier:N", scale=alt.Scale(domain=list(TIER_COLOR), range=list(TIER_COLOR.values()))),
        tooltip=[alt.Tooltip("tier_vi:N", title="Mức"), alt.Tooltip("n_routes:Q", title="Số tuyến"),
                 alt.Tooltip("n_orders:Q", title="Số đơn", format=",")])
    num = alt.Chart(pd.DataFrame({"t": [str(center)]})).mark_text(
        fontSize=30, fontWeight=600, color=INK, font="Inter", dy=-6).encode(text="t:N")
    lab = alt.Chart(pd.DataFrame({"t": ["tuyến mức Cao"]})).mark_text(
        fontSize=11.5, color=MUTED, font="Inter", dy=16).encode(text="t:N")
    return clean(alt.layer(arc, num, lab), 190)


# ============================================================ 5. GHÉP TRANG
def kpi_card(key, icon_name, label, value, pill, pill_grey, spark_df, spark_field):
    with st.container(key=key):
        st.markdown(
            f'<div class="kpi-head"><div class="kpi-icon">{icon(icon_name)}</div>'
            f'<div class="kpi-label">{label}</div></div>'
            f'<div class="kpi-value">{value}</div>'
            f'<span class="kpi-pill{" grey" if pill_grey else ""}">{pill}</span>',
            unsafe_allow_html=True)
        if spark_df is not None and len(spark_df) > 1:
            st.altair_chart(spark(spark_df, spark_field, label), width="stretch")


def card_header(title, note=None):
    st.markdown(f'<div class="card-title">{title}</div>' + (f'<div class="card-note">{note}</div>' if note else ""),
                unsafe_allow_html=True)


def build_insights(f, by_mode, by_market, monthly, mode_days):
    """Điểm chính dạng "con số + vài chữ". Trả về (số lớn, nhãn số lớn, danh sách (số, nhãn))."""
    worst = by_mode.sort_values("late_rate", ascending=False).iloc[0]
    days = mode_days.get(worst["shipping_mode"])
    lead = f"{worst['shipping_mode']} giao trễ" + (f" - cam kết {days} ngày" if days is not None else "")
    items = []
    best = by_mode.sort_values("late_rate").iloc[0]
    if best["shipping_mode"] != worst["shipping_mode"]:
        items.append((vn_pct(best["late_rate"], 0), f"{best['shipping_mode']} - trễ ít nhất"))
    if len(by_market) > 1:
        mk = by_market.sort_values("revenue_at_risk", ascending=False).iloc[0]
        share = mk["revenue_at_risk"] / by_market["revenue_at_risk"].sum()
        items.append((vn_pct(share, 0), f"tiền rủi ro ở {mk['market']}"))
    if len(monthly) >= 12:
        recent, prev = monthly.tail(6), monthly.iloc[-12:-6]
        r1 = recent["n_late"].sum() / recent["n_not_canceled"].sum()
        r0 = prev["n_late"].sum() / prev["n_not_canceled"].sum()
        diff = (r1 - r0) * 100
        if abs(diff) <= 0.5:
            items.append(("Ổn định", "tỷ lệ trễ 6 tháng gần nhất"))
        else:
            arrow = "↑" if diff > 0 else "↓"
            items.append((f"{arrow}{abs(diff):.1f}".replace(".", ",") + "%", "tỷ lệ trễ 6 tháng gần nhất"))
    return worst["late_rate"], lead, items


def page_overview():
    df, source = load_overview()
    mode_days, region_market = load_mode_days()

    # ---------- Bộ lọc ở thanh bên (để trống = tất cả)
    months = sorted(df["year_month"].unique())
    with st.sidebar:
        st.markdown('<div class="card-title" style="margin-top:18px">Bộ lọc</div>', unsafe_allow_html=True)
        start, end = st.select_slider("Thời gian đặt hàng", options=months,
                                      value=(months[0], months[-1]), format_func=ym_label)
        markets = st.multiselect("Thị trường", sorted(df["market"].unique()), placeholder="Tất cả thị trường")
        modes = st.multiselect("Phương thức vận chuyển",
                               [m for m in MODE_ORDER if m in set(df["shipping_mode"])] or sorted(df["shipping_mode"].unique()),
                               placeholder="Tất cả phương thức")

    f = df[(df["year_month"] >= start) & (df["year_month"] <= end)]
    if markets:
        f = f[f["market"].isin(markets)]
    if modes:
        f = f[f["shipping_mode"].isin(modes)]

    # ---------- Tiêu đề
    left, right = st.columns([3, 2], vertical_alignment="center")
    left.markdown('<div class="ov-title">Tổng quan giao hàng</div>', unsafe_allow_html=True)
    scope = ", ".join(markets) if markets else "Tất cả thị trường"
    right.markdown(f'<div style="text-align:right"><span class="ov-chip">{ym_label(start)} - {ym_label(end)}</span>'
                   f'<span class="ov-chip">{scope}</span></div>', unsafe_allow_html=True)
    st.write("")

    if f.empty or f["n_not_canceled"].sum() == 0:
        st.info("Không có đơn hàng nào khớp bộ lọc. Hãy nới rộng bộ lọc ở thanh bên.")
        return

    tot = summarize(f).iloc[0]
    monthly = summarize(f, "year_month").sort_values("year_month")
    by_mode = summarize(f, "shipping_mode")
    by_market = summarize(f, "market")

    # ---------- Hàng 1: 4 thẻ chỉ số
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        kpi_card("card_kpi1", "box", "Số đơn hàng", vn_int(tot["n_orders"]),
                 f"{vn_int(tot['n_canceled'])} đơn hủy ({vn_pct(tot['cancel_rate'])})", True, monthly, "n_orders")
    with c2:
        kpi_card("card_kpi2", "late", "Số đơn giao trễ", vn_int(tot["n_late"]),
                 f"trên {vn_int(tot['n_not_canceled'])} đơn không hủy", False, monthly, "n_late")
    with c3:
        kpi_card("card_kpi3", "delay", "Chênh lệch giao hàng TB",
                 f"{vn_num(tot['avg_delay'])}<small>ngày</small>",
                 "so với ngày cam kết", True, monthly, "avg_delay")
    with c4:
        kpi_card("card_kpi4", "money", "Doanh thu chịu rủi ro", vn_money(tot["revenue_at_risk"]),
                 f"{vn_pct(tot['risk_share'])} tổng doanh thu", False, monthly, "revenue_at_risk")

    # ---------- Hàng 2: đường xu hướng + thẻ xe tải
    a, b = st.columns([2.1, 1], gap="medium")
    with a:
        with st.container(key="card_trend"):
            h1, h2 = st.columns([3, 1.1], vertical_alignment="center")
            with h1:
                card_header("Xu hướng tỷ lệ giao trễ")
            with h2:
                period = st.segmented_control("Kỳ", ["Tháng", "Quý"], default="Tháng",
                                              label_visibility="collapsed", key="period")
            by_q = period == "Quý"
            st.altair_chart(trend_chart(to_period(monthly, by_q), by_q), width="stretch")
    with b:
        with st.container(key="card_truck"):
            card_header("Tỷ lệ giao trễ")
            st.markdown(truck_svg(tot["late_rate"]), unsafe_allow_html=True)
            on_time = tot["n_not_canceled"] - tot["n_late"]
            st.markdown(
                f'<div class="truck-big">Cứ 10 đơn được giao thì có khoảng <b>{tot["late_rate"] * 10:.0f}</b> '
                f'đơn đến trễ so với cam kết.</div>'
                f'<div class="truck-stats">'
                f'<div class="truck-stat"><div class="v">{vn_int(on_time)}</div><div class="l">Đúng hạn</div></div>'
                f'<div class="truck-stat red"><div class="v">{vn_int(tot["n_late"])}</div><div class="l">Giao trễ</div></div>'
                f'<div class="truck-stat"><div class="v">{vn_int(tot["n_canceled"])}</div><div class="l">Bị hủy</div></div>'
                f'</div>', unsafe_allow_html=True)

    # ---------- Hàng 3: theo phương thức · theo thị trường · thẻ đỏ "Điểm chính"
    x, y, z = st.columns([1.25, 1.1, 1], gap="medium")
    with x:
        with st.container(key="card_mode"):
            card_header("Tỷ lệ trễ theo phương thức")
            st.altair_chart(mode_chart(by_mode, mode_days), width="stretch")
    with y:
        with st.container(key="card_market"):
            card_header("Doanh thu chịu rủi ro theo thị trường")
            st.altair_chart(market_chart(by_market), width="stretch")
    with z:
        big, lead, items = build_insights(f, by_mode, by_market, monthly, mode_days)
        pct_txt = vn_pct(big, 0)
        with st.container(key="accent_insight"):
            st.markdown(
                '<div class="ins-kicker">Điểm chính</div>'
                f'<div class="ins-big">{pct_txt[:-1]}<small>%</small></div>'
                f'<div class="ins-lead">{lead}</div>'
                + "".join(f'<div class="ins-row"><span class="ins-num">{v}</span>'
                          f'<span class="ins-lab">{t}</span></div>' for v, t in items),
                unsafe_allow_html=True)
