"""Trang Trợ lý dữ liệu: hỏi bằng tiếng Anh, AI viết SQL, chạy trên tầng Gold của BigQuery.

Luồng xử lý một câu hỏi:
  1. ask_sql()    : gửi câu hỏi + mô tả các bảng Gold cho AI (Groq, hoặc Gemini) -> nhận về câu SQL
  2. check_sql()  : chỉ cho phép SELECT trên đúng 4 bảng Gold (chặn xóa/sửa dữ liệu)
  3. run_sql()    : chạy trên BigQuery, có giới hạn dung lượng quét
     (nếu lỗi, gửi lỗi lại cho AI sửa 1 lần)
  4. ask_answer() : AI đọc kết quả và trả lời ngắn gọn bằng tiếng Anh
  5. Hiển thị câu trả lời + bảng số liệu + biểu đồ (nếu hợp) + câu SQL để kiểm chứng
"""
import json
import re

import altair as alt
import pandas as pd
import streamlit as st

from views.overview import ACCENT, INK, MUTED, _bigquery_client, clean

PROJECT = "dw-dataco-513010"
DATASET = f"{PROJECT}.gold"
ALLOWED_TABLES = {"wide_order_items", "route_scorecard", "route_scorecard_monthly", "route_risk_expected"}
MAX_BYTES = 1_000_000_000          # mỗi câu hỏi quét tối đa 1 GB (bảng lớn nhất chỉ vài chục MB)
MAX_ROWS = 500
TIMEOUT = 30                       # mỗi bước chờ tối đa 30 giây
# Mô hình AI: dùng Gemini trước (nếu có GEMINI_API_KEY), quá tải thì tự chuyển sang mô hình Gemini khác,
# rồi sang Groq (nếu có GROQ_API_KEY). Muốn đổi mô hình, thêm GEMINI_MODEL = "..." / GROQ_MODEL = "..." vào secrets.toml.
GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-flash-latest"]
GROQ_MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]   # llama-3.3-70b đã bị Groq gỡ
BUSY_WORDS = ("503", "unavailable", "overloaded", "high demand", "429", "resource_exhausted", "rate limit",
              "rate_limit", "timed out", "timeout", "504", "deadline", "500 internal", "internal error")
MAX_WAIT = 75                       # tổng thời gian tối đa cho một lần gọi AI (kể cả thử mô hình dự phòng)

SAMPLES = [
    "Which shipping mode has the highest late rate?",
    "Top 5 routes by expected revenue at risk",
    "Late rate by year",
    "Which product category is late the most?",
    "Compare late rates across customer segments",
    "How is the risk score calculated?",
]

# ------------------------------------------------------------------ mô tả dữ liệu cho AI
SCHEMA = f"""
You work with 4 tables in BigQuery, dataset `{DATASET}` (DataCo data, orders placed from 01/2015 to 01/2018, currency USD).

1) `{DATASET}.wide_order_items` - each row is ONE ORDER LINE (one product within an order), 180,519 rows.
   An order (order_id) can have several lines.
   Columns: order_item_id, order_id, route_name, shipping_mode, scheduled_shipping_days (promised days),
   market, order_region, order_country, product_id, product_name, category_name, department_name,
   customer_id, customer_segment, payment_type, quantity, unit_price, sales, discount_rate,
   order_item_total (line revenue after discount), order_datetime, order_date, order_year, order_month,
   order_year_month (string 'YYYY-MM'), order_day_of_week (1=Sunday..7=Saturday), order_is_weekend, order_hour,
   order_status, delivery_status, actual_shipping_days, delay_days (actual - promised, positive means late),
   shipping_datetime, is_canceled (BOOL), order_profit, late_delivery_risk (1 = late, 0 = not late).

2) `{DATASET}.route_scorecard` - each row is ONE ROUTE (destination region x shipping mode), 92 routes.
   Columns: route_key, route_name, order_region, shipping_mode, n_orders, n_orders_not_canceled, n_late_orders,
   n_canceled_orders, late_rate, avg_delay_days, cancel_rate, total_revenue, revenue_at_risk (revenue of late orders),
   is_reliable (at least 100 orders), risk_score (0-1), risk_rank, risk_rank_in_mode, risk_tier ('High','Medium','Low','Insufficient data').

3) `{DATASET}.route_scorecard_monthly` - each row is one route in one month.
   Columns: route_key, route_name, order_region, shipping_mode, year_month ('YYYY-MM'), n_orders, late_rate,
   avg_delay_days, cancel_rate, revenue_at_risk.

4) `{DATASET}.route_risk_expected` - each route with the XGBoost prediction results.
   Columns: route_name, order_region, shipping_mode, n_orders, n_scored_orders, actual_late_rate, predicted_late_rate,
   actual_revenue_at_risk, expected_revenue_at_risk (late probability x revenue), risk_score, risk_rank, risk_tier.

Values to know:
- shipping_mode: 'First Class', 'Second Class', 'Same Day', 'Standard Class'
- market: 'Africa', 'Europe', 'LATAM', 'Pacific Asia', 'USCA'
- customer_segment: 'Consumer', 'Corporate', 'Home Office'
- 23 order_region values, e.g. 'Western Europe', 'Central America', 'South of  USA' (note the DOUBLE space).
"""

GLOSSARY = """
Project terms (used to answer concept questions, no SQL needed):
- Shipping mode (shipping_mode) is the delivery SERVICE LEVEL, not a carrier. There are 4 levels,
  each with the number of days the business promises the customer: Same Day (delivered the same day, promise 0 days),
  First Class (promise 1 day), Second Class (promise 2 days), Standard Class (standard level, promise 4 days).
- In the data, First Class is always late because it really takes about 2 days while only 1 day is promised.
- Late delivery: actual shipping days greater than the promised days (late_delivery_risk = 1).
- Late rate: late orders divided by non-canceled orders, computed at order level.
- Route: a destination region combined with a shipping mode, e.g. "Western Europe - First Class". There are 92 routes
  (23 regions x 4 modes).
- Risk score (risk_score, 0-1): 50% late rate + 30% average days late + 20% revenue affected, each part normalised to 0-1.
- Risk tier: High if the score is at least 0.66; Medium if at least 0.33; Low if below 0.33; Insufficient data if the route has fewer than 100 orders.
- Revenue at risk: total revenue of late orders. Expected revenue at risk: the late probability predicted by the
  XGBoost model times the order value.
- Data: the DataCo Smart Supply Chain dataset (Kaggle), orders placed from 01/2015 to 01/2018, processed through the Bronze, Silver and Gold layers on BigQuery.
"""

RULES = f"""
SQL rules (BigQuery Standard SQL):
- Write exactly ONE SELECT statement (WITH is allowed). Never modify, delete or create tables.
- Always write the full table name inside backticks, e.g. `{DATASET}.wide_order_items`.
- Count orders with COUNT(DISTINCT order_id), NOT by counting rows.
- The late rate must be computed at ORDER level, with non-canceled orders as the denominator. Template:
    WITH o AS (
      SELECT order_id, <group columns>, MAX(late_delivery_risk) AS is_late, LOGICAL_OR(is_canceled) AS is_canceled,
             SUM(order_item_total) AS revenue
      FROM `{DATASET}.wide_order_items` GROUP BY order_id, <group columns>)
    SELECT <group columns>, COUNT(*) AS n_orders,
           ROUND(SAFE_DIVIDE(COUNTIF(is_late = 1), COUNTIF(NOT is_canceled)), 4) AS late_rate
    FROM o GROUP BY <group columns>
- NEVER compute the late rate with AVG(late_delivery_risk) over order lines (wrong: orders with several products are counted several times).
- Late rate by shipping mode, region or route (or overall): use `{DATASET}.route_scorecard`, add the counts first and divide afterwards:
    SELECT shipping_mode, SUM(n_orders) AS n_orders,
           ROUND(SAFE_DIVIDE(SUM(n_late_orders), SUM(n_orders_not_canceled)), 4) AS late_rate
    FROM `{DATASET}.route_scorecard` GROUP BY shipping_mode
  By market use the WITH o AS (...) template above on wide_order_items (route_scorecard has no market column).
- Questions about routes, risk score, ranking, money at risk: prefer route_scorecard or route_risk_expected.
  When ranking routes, only keep routes with risk_tier <> 'Insufficient data'.
- Name result columns in short English snake_case (e.g. shipping_mode, n_orders, late_rate, revenue_at_risk).
- Rates stay in the 0-1 form, rounded to 4 digits. Money rounded to 0 digits.
- Always include a sensible ORDER BY and a LIMIT of at most 50.

Return ONLY one JSON:
- If the question needs data: {{"sql": "<SQL statement>", "column_labels": {{"<column name in the SQL>": "<short English display name>"}}}}
  In "column_labels", give a short English display name for EVERY result column,
  e.g. {{"shipping_mode": "Shipping mode", "n_orders": "Orders", "late_rate": "Late rate"}}.
- If it is a concept question (e.g. "what is First Class", "how is the risk score computed"): answer from the Terms section, 2-3 short sentences:
  {{"sql": null, "reply": "<answer>"}}
- If it is a greeting or the data cannot answer: {{"sql": null, "reply": "<short answer, suggest suitable questions>"}}
- Write the "reply" calmly, without long dashes.
"""

ANSWER_RULES = """
You are a delivery analysis assistant. Based on the question and the result table, answer in English:
- At most 3 short sentences: go straight to the main number and one remark useful for operations.
- Write numbers in English style: comma for thousands, dot for decimals, % attached: 65,752 orders, 57.3%, 100.0%.
  A 0-1 rate becomes a percentage rounded to 1 digit. Money in USD.
- Calm, objective analyst tone. No emotional words such as "alarming", "urgent", "extremely".
- Useful context: First Class promises 1 day but really always takes about 2 days, so the problem lies in the promise
  rather than in operations. Mention it only when relevant.
- Use only numbers that are in the result table, do not invent any. If the table is empty, say that no matching data was found.
- No headings, no bullet points, no long dashes, only the short hyphen "-" if needed.
"""

PAGE_CSS = f"""
<style>
.st-key-sample_pills button {{ background:#FFFFFF; border:1px solid #EBD3D2; }}
.st-key-sample_pills button:hover {{ border-color:{ACCENT}; color:{ACCENT}; }}
[data-testid="stChatMessage"] {{ background:#FFFFFF; border-radius:18px; padding:14px 18px;
    box-shadow:0 1px 2px rgba(0,0,0,.04); margin-bottom:10px; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{ background:#FDECEC; }}
.chat-hello {{ background:#FFFFFF; border-radius:20px; padding:22px 24px; margin:8px 0 14px 0; }}
.chat-hello h4 {{ margin:0 0 4px 0; color:{INK}; font-weight:600; }}
.chat-hello p {{ margin:0; color:{MUTED}; font-size:14px; }}
.sql-note {{ font-size:12.5px; color:{MUTED}; }}
</style>
"""


# ------------------------------------------------------------------ 1. GỌI AI
def _secret(name):
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def providers():
    """Danh sách AI theo thứ tự thử: Gemini trước, Groq dự phòng (chỉ những cái có key)."""
    order = [name for name, key in (("Gemini", "GEMINI_API_KEY"), ("Groq", "GROQ_API_KEY")) if _secret(key)]
    if _secret("AI_PRIMARY") == "Groq" and "Groq" in order:      # muốn Groq chạy trước thì thêm AI_PRIMARY = "Groq"
        order = ["Groq"] + [p for p in order if p != "Groq"]
    return order


def provider():
    """Tên hiển thị dưới tiêu đề trang, vd "Gemini (dự phòng: Groq)"."""
    p = providers()
    if not p:
        return None
    return p[0] + (f" (fallback: {', '.join(p[1:])})" if len(p) > 1 else "")


@st.cache_resource
def _groq():
    from groq import Groq
    return Groq(api_key=_secret("GROQ_API_KEY"), timeout=TIMEOUT, max_retries=1)


@st.cache_resource
def _gemini():
    from google import genai
    from google.genai import types
    return genai.Client(api_key=_secret("GEMINI_API_KEY"), http_options=types.HttpOptions(timeout=TIMEOUT * 1000))


def _is_denied(e):
    """Khóa sai, hết hạn hoặc project chưa bật thanh toán: thử lại hay đổi mô hình cũng không được."""
    m = str(e).lower()
    return any(w in m for w in ("403", "permission_denied", "api key not valid", "api_key_invalid", "billing", "401"))


def _is_busy(e):
    m = str(e).lower()
    return not _is_denied(e) and any(w in m for w in BUSY_WORDS)


def _friendly(e):
    if _is_denied(e):
        if "free tier is not available" in str(e).lower() or "billing" in str(e).lower():
            return ("The Gemini key is valid but its Google project cannot use the free tier. "
                    "Enable billing for that project, create a key in another project, or use GROQ_API_KEY.")
        return "The AI key is invalid or lacks permission. Check the key in Secrets."
    if _is_busy(e):
        return "The AI is overloaded or out of free quota. Wait about a minute and ask again."
    return f"Could not call the AI: {str(e)[:300]}"


def _gemini_config(model, system, json_mode, fast):
    """fast=True: tắt hoặc giảm "suy nghĩ" cho nhanh (dòng 2.5 dùng thinking_budget, dòng 3 dùng thinking_level)."""
    from google.genai import types
    kw = dict(system_instruction=system, temperature=0.1,
              response_mime_type="application/json" if json_mode else "text/plain")
    if fast:
        try:
            kw["thinking_config"] = (types.ThinkingConfig(thinking_budget=0) if "2.5" in model
                                     else types.ThinkingConfig(thinking_level="low"))
        except Exception:
            pass
    return types.GenerateContentConfig(**kw)


def _call_gemini(system, prompt, json_mode, deadline):
    import time
    first = _secret("GEMINI_MODEL")
    models = ([first] if first else []) + [m for m in GEMINI_MODELS if m != first]
    last = None
    for model in models:
        if time.time() > deadline:
            break
        for fast in (True, False):              # lỗi 400 do cấu hình "suy nghĩ" -> thử lại không kèm cấu hình
            try:
                cfg = _gemini_config(model, system, json_mode, fast)
                return _gemini().models.generate_content(model=model, contents=prompt, config=cfg).text
            except Exception as e:
                last = e
                m = str(e).lower()
                if fast and ("400" in m or "invalid" in m or "thinking" in m):
                    continue
                break
        m = str(last).lower()
        if not (_is_busy(last) or "not found" in m or "404" in m or "400" in m or "invalid" in m):
            break                               # lỗi khác (sai key...) thì không thử tiếp
    raise last or TimeoutError("timeout: the AI took too long to answer")


def _call_groq(system, prompt, json_mode, deadline):
    models = [_secret("GROQ_MODEL")] if _secret("GROQ_MODEL") else GROQ_MODELS
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    last = None
    for model in models:
        import time
        if time.time() > deadline:
            break
        try:
            r = _groq().chat.completions.create(
                model=model, temperature=0.1, max_tokens=1200,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}], **extra)
            return r.choices[0].message.content
        except Exception as e:
            last = e
            m = str(e).lower()
            # mô hình bị gỡ (404), lỗi ép JSON (400 json_validate) hay quá tải: thử mô hình kế tiếp
            if _is_denied(e) or not (_is_busy(e) or "model" in m or "json_validate" in m or "400" in m):
                break
    raise last or TimeoutError("timeout: the AI took too long to answer")


def _generate(system, prompt, json_mode):
    """Gọi AI theo thứ tự providers(); cái trước lỗi thì chuyển sang cái sau."""
    order = providers()
    if not order:
        raise RuntimeError("No AI API key in secrets.toml.")
    import time
    deadline = time.time() + MAX_WAIT
    last = None
    for name in order:
        try:
            return (_call_gemini if name == "Gemini" else _call_groq)(system, prompt, json_mode, deadline)
        except Exception as e:
            last = e
    raise RuntimeError(_friendly(last))


@st.cache_data(ttl=3600, show_spinner=False)
def _ask_sql_cached(question, ctx, error, bad_sql):
    prompt = (f"Recent conversation context (use it if the new question continues the previous one):\n{ctx or '(none)'}\n\n"
              f"New question: {question}")
    if error:
        prompt += f"\n\nThe previous SQL failed:\n{bad_sql}\nError: {error}\nRewrite the SQL correctly."
    return _generate(SCHEMA + GLOSSARY + RULES, prompt, json_mode=True)


def ask_sql(question, history, error=None, bad_sql=None):
    """Trả về dict {"sql": ...} hoặc {"sql": None, "reply": ...}. Câu hỏi giống hệt sẽ lấy từ bộ nhớ đệm."""
    ctx = "".join(f"- Previous question: {h['q']}\n  SQL used: {h['sql']}\n" for h in history[-2:] if h.get("sql"))
    text = _ask_sql_cached(question.strip(), ctx, error, bad_sql)
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        out = json.loads(m.group(0)) if m else {"sql": None, "reply": text}
    return out if isinstance(out, dict) else {"sql": None, "reply": str(out)}


def clean_text(s):
    """Normalise the AI text: no long dashes, no space before % (57.3 % -> 57.3%)."""
    s = (s or "").replace("—", "-").replace("–", "-")
    s = re.sub(r"(?<=\d)\s+%", "%", s)           # "57.3 %" -> "57.3%"
    return s.replace("$", "\\$")                   # st.markdown would read two "$" as a LaTeX formula


@st.cache_data(ttl=3600, show_spinner=False)
def ask_answer(question, sql, df):
    table = df.head(30).to_csv(index=False)
    prompt = f"Question: {question}\n\nSQL that ran:\n{sql}\n\nResult ({len(df)} rows, first 30 rows at most):\n{table}"
    return clean_text(_generate(ANSWER_RULES + GLOSSARY, prompt, json_mode=False).strip())


# ------------------------------------------------------------------ 2. KIỂM TRA SQL
FORBIDDEN = re.compile(r"\b(INSERT|UPDATE|DELETE|MERGE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE|EXPORT|LOAD|"
                       r"CALL|DECLARE|SET|BEGIN|COMMIT|ROLLBACK|EXECUTE)\b", re.I)


def check_sql(sql):
    """Chỉ cho phép một câu SELECT đọc từ 4 bảng Gold. Sai thì báo lỗi (ValueError)."""
    s = re.sub(r"^```(?:sql)?|```$", "", (sql or "").strip(), flags=re.I).strip().rstrip(";").strip()
    no_str = re.sub(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"", "''", s)      # bỏ chuỗi để không bắt nhầm chữ
    no_str = re.sub(r"--[^\n]*|/\*.*?\*/", " ", no_str, flags=re.S)          # bỏ chú thích
    if not s:
        raise ValueError("Empty SQL.")
    if ";" in no_str:
        raise ValueError("Only one statement may run.")
    if not re.match(r"(?is)^\s*(WITH|SELECT)\b", no_str):
        raise ValueError("Only SELECT statements are allowed.")
    if FORBIDDEN.search(no_str):
        raise ValueError("The SQL contains a forbidden command.")
    scan = re.sub(r"(?is)EXTRACT\s*\([^()]*\)", " ", no_str)                 # EXTRACT(YEAR FROM x) không phải bảng
    ctes = {c.lower() for c in re.findall(r"(?i)(?:\bWITH|,)\s*([A-Za-z_]\w*)\s+AS\s*\(", scan)}
    refs = re.findall(r"(?i)\b(?:FROM|JOIN)\s+(`[^`]+`|[A-Za-z_][\w.\-]*)", scan)
    refs += [f"`{q}`" for q in re.findall(r"`([^`]*\.[^`]*)`", scan)]   # mọi tên bảng trong backtick
    if not refs:
        raise ValueError("No data table found in the SQL.")
    for ref in refs:
        name = ref.strip("`")
        if name.lower() in ctes or name.upper() == "UNNEST":
            continue
        parts = name.split(".")
        ok = (parts[-1] in ALLOWED_TABLES and
              (len(parts) == 1 or (len(parts) == 2 and parts[0] == "gold") or
               (len(parts) == 3 and parts[0] == PROJECT and parts[1] == "gold")))
        if not ok:
            raise ValueError(f"Table {name} is not in the allowed list.")
    return s


# ------------------------------------------------------------------ 3. CHẠY TRÊN BIGQUERY
@st.cache_data(ttl=3600, show_spinner=False)
def run_sql(sql):
    from google.cloud import bigquery
    client = _bigquery_client()
    if client is None:
        raise RuntimeError("No BigQuery key configured in secrets.toml.")
    job = client.query(sql, job_config=bigquery.QueryJobConfig(
        maximum_bytes_billed=MAX_BYTES, default_dataset=DATASET))   # quét quá 1 GB thì BigQuery tự từ chối
    return job.result(timeout=TIMEOUT).to_dataframe(create_bqstorage_client=False).head(MAX_ROWS)


def answer(question, history, step=lambda text: None):
    """Xử lý trọn một câu hỏi. step(text) dùng để báo bước đang chạy lên giao diện."""
    step("Writing the SQL query...")
    plan = ask_sql(question, history)
    if not plan.get("sql"):
        reply = plan.get("reply") or "I did not understand the question, please try again."
        return {"q": question, "text": clean_text(reply)}
    sql, error = plan["sql"], None
    for attempt in range(2):                    # lần đầu + 1 lần nhờ AI sửa lỗi
        try:
            sql = check_sql(sql)
            step("Running on BigQuery...")
            df = run_sql(sql)
            step("Summarising the result...")
            return {"q": question, "sql": sql, "df": df, "labels": plan.get("column_labels") or {},
                    "text": ask_answer(question, sql, df)}
        except RuntimeError:                    # lỗi gọi AI (quá tải, hết giờ) -> báo luôn, không thử lại
            raise
        except Exception as e:
            error = str(e)[:500]
            if attempt == 1:
                break
            step("The query failed, asking the AI to fix it...")
            plan = ask_sql(question, history, error=error, bad_sql=sql)
            if not plan.get("sql"):
                break
            sql = plan["sql"]
    return {"q": question, "sql": sql, "text": f"I could not run a query for this question. Error: {error}"}


# ------------------------------------------------------------------ 4. HIỂN THỊ
# Tên cột hay gặp -> tên tiếng Việt có dấu (dùng khi AI không trả về "ten_cot")
COL_EN = {
    "shipping_mode": "Shipping mode", "market": "Market", "order_region": "Region", "order_country": "Country",
    "route_name": "Route", "category_name": "Category", "department_name": "Department",
    "customer_segment": "Customer segment", "payment_type": "Payment type",
    "order_year": "Year", "year": "Year", "order_year_month": "Month", "year_month": "Month", "month": "Month",
    "quarter": "Quarter", "hour": "Order hour", "order_hour": "Order hour", "weekday": "Weekday", "is_weekend": "Weekend",
    "n_orders": "Orders", "late_orders": "Late orders", "n_late_orders": "Late orders",
    "canceled_orders": "Canceled orders", "n_canceled_orders": "Canceled orders",
    "late_rate": "Late rate", "cancel_rate": "Cancel rate",
    "revenue_at_risk": "Revenue at risk (USD)", "expected_revenue_at_risk": "Expected revenue at risk (USD)",
    "revenue": "Revenue (USD)", "risk_score": "Risk score", "risk_rank": "Rank", "rank": "Rank", "risk_tier": "Risk tier",
    "avg_delay_days": "Avg days late", "order_id": "Order ID",
}


def en_columns(df, labels):
    """Đổi tên cột kết quả sang tiếng Anh: ưu tiên tên AI đặt, sau đó bảng COL_EN, cuối cùng bỏ dấu gạch dưới."""
    labels = labels or {}
    def name(c):
        return labels.get(c) or COL_EN.get(c) or c.replace("_", " ").capitalize()
    out = df.rename(columns={c: name(c) for c in df.columns})
    if out.columns.duplicated().any():          # tránh trùng tên sau khi đổi
        out.columns = [f"{c} ({i})" if out.columns[:i].tolist().count(c) else c for i, c in enumerate(out.columns)]
    return out


def auto_chart(df, raw_cols):
    """Vẽ biểu đồ cột nếu kết quả có 1 cột chữ + ít nhất 1 cột số và 2-25 dòng.
    df đã đổi tên cột tiếng Anh; raw_cols là tên cột gốc (để nhận biết cột tỷ lệ)."""
    if df is None or not 2 <= len(df) <= 25:
        return None
    raw = dict(zip(df.columns, raw_cols))
    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(text_cols) != 1 or not num_cols:
        return None
    cat = text_cols[0]
    is_rate_name = lambda c: "rate" in raw[c].lower() or "rate" in c.lower()
    val = next((c for c in num_cols if is_rate_name(c)), num_cols[-1])
    d = df[[cat, val]].copy()
    d[cat] = d[cat].astype(str)
    is_rate = d[val].between(0, 1).all() and is_rate_name(val)
    d["Highest"] = d[val] == d[val].max()
    order = d.sort_values(val, ascending=False)[cat].tolist()
    rc = raw[cat]
    if any(k in rc for k in ("month", "year", "quarter")) or rc in ("hour", "weekday"):
        order = sorted(d[cat].tolist())
    bars = alt.Chart(d).mark_bar(cornerRadius=8, size=16).encode(
        y=alt.Y(field=cat, type="nominal", sort=order, title=None,
                axis=alt.Axis(labelColor=INK, labelLimit=220, labelOverlap=False)),
        x=alt.X(field=val, type="quantitative", title=None, axis=alt.Axis(format="%" if is_rate else "~s")),
        color=alt.condition("datum['Highest']", alt.value(ACCENT), alt.value("#F2A6A4")),
        tooltip=[alt.Tooltip(field=cat, type="nominal"),
                 alt.Tooltip(field=val, type="quantitative", format=".1%" if is_rate else ",.2f")])
    return clean(bars, max(150, 40 * len(d)))


def show_reply(msg):
    """Câu trả lời + biểu đồ (nếu hợp). Bảng số liệu và câu SQL vẫn lưu trong msg nhưng không hiển thị."""
    st.markdown(msg["text"])
    df = msg.get("df")
    if df is not None and len(df):
        chart = auto_chart(en_columns(df, msg.get("labels")), list(df.columns))
        if chart is not None:
            st.altair_chart(chart, width="stretch")


def page_chatbot():
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    st.markdown('<div class="ov-title">Data assistant</div>', unsafe_allow_html=True)
    st.write("")

    try:
        ready = provider() is not None and _bigquery_client() is not None
    except Exception:
        ready = False
    if not ready:
        st.warning("No AI API key (Groq or Gemini) or BigQuery key in .streamlit/secrets.toml. "
                   "Run `python scripts/make_secrets.py` and reopen the app.")
        return

    history = st.session_state.setdefault("chat", [])
    with st.sidebar:
        st.write("")
        if st.button("Clear conversation", icon=":material/delete:", width="stretch"):
            history.clear()
            st.rerun()

    question = st.chat_input("Example: What is the late rate of Second Class in Europe?") \
        or st.session_state.pop("picked_q", None)

    if not history and not question:
        st.markdown('<div class="chat-hello"><h4>What would you like to know about deliveries?</h4>'
                    '<p>Click a suggested question below or type your own at the bottom of the page.</p></div>',
                    unsafe_allow_html=True)

    def _pick():                                 # bấm một câu gợi ý -> dùng nó làm câu hỏi
        st.session_state["picked_q"] = st.session_state["sample_pills"]
        st.session_state["sample_pills"] = None

    st.pills("Suggestions", SAMPLES, selection_mode="single", key="sample_pills",
             label_visibility="collapsed", on_change=_pick)

    for msg in history:
        with st.chat_message("user"):
            st.markdown(msg["q"])
        with st.chat_message("assistant"):
            show_reply(msg)

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.status("Processing the question...", expanded=False) as status:
                try:
                    msg = answer(question, history, step=lambda t: status.update(label=t))
                    status.update(label="Done", state="complete")
                except Exception as e:
                    msg = {"q": question, "text": str(e) if isinstance(e, RuntimeError) else f"Error while processing: {e}"}
                    status.update(label="Not completed", state="error")
            show_reply(msg)
        history.append(msg)
      
