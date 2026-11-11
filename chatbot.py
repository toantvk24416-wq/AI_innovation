"""Trang Trợ lý dữ liệu: hỏi bằng tiếng Việt, AI viết SQL, chạy trên tầng Gold của BigQuery.

Luồng xử lý một câu hỏi:
  1. ask_sql()    : gửi câu hỏi + mô tả các bảng Gold cho AI (Groq, hoặc Gemini) -> nhận về câu SQL
  2. check_sql()  : chỉ cho phép SELECT trên đúng 4 bảng Gold (chặn xóa/sửa dữ liệu)
  3. run_sql()    : chạy trên BigQuery, có giới hạn dung lượng quét
     (nếu lỗi, gửi lỗi lại cho AI sửa 1 lần)
  4. ask_answer() : AI đọc kết quả và trả lời ngắn gọn bằng tiếng Việt
  5. Hiển thị câu trả lời + bảng số liệu + biểu đồ (nếu hợp) + câu SQL để kiểm chứng
"""
import json
import re

import altair as alt
import pandas as pd
import streamlit as st

from overview import ACCENT, INK, MUTED, _bigquery_client, clean

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
    "Phương thức vận chuyển nào có tỷ lệ giao trễ cao nhất?",
    "5 tuyến có tiền rủi ro kỳ vọng lớn nhất",
    "Tỷ lệ giao trễ theo năm",
    "Nhóm hàng nào bị trễ nhiều nhất?",
    "So sánh tỷ lệ trễ giữa các phân khúc khách hàng",
    "Điểm rủi ro được tính như thế nào?",
]

# ------------------------------------------------------------------ mô tả dữ liệu cho AI
SCHEMA = f"""
Bạn làm việc với 4 bảng trong BigQuery, dataset `{DATASET}` (dữ liệu DataCo, đơn đặt từ 01/2015 đến 01/2018, tiền tệ USD).

1) `{DATASET}.wide_order_items` - mỗi dòng là MỘT DÒNG HÀNG (một sản phẩm trong một đơn), 180.519 dòng.
   Một đơn (order_id) có thể có nhiều dòng hàng.
   Cột: order_item_id, order_id, route_name, shipping_mode, scheduled_shipping_days (số ngày cam kết),
   market, order_region, order_country, product_id, product_name, category_name, department_name,
   customer_id, customer_segment, payment_type, quantity, unit_price, sales, discount_rate,
   order_item_total (doanh thu dòng hàng sau giảm giá), order_datetime, order_date, order_year, order_month,
   order_year_month (chuỗi 'YYYY-MM'), order_day_of_week (1=Chủ nhật..7=Thứ bảy), order_is_weekend, order_hour,
   order_status, delivery_status, actual_shipping_days, delay_days (thực tế - cam kết, dương là trễ),
   shipping_datetime, is_canceled (BOOL), order_profit, late_delivery_risk (1 = giao trễ, 0 = không).

2) `{DATASET}.route_scorecard` - mỗi dòng là MỘT TUYẾN (khu vực nhận hàng x phương thức), 92 tuyến.
   Cột: route_key, route_name, order_region, shipping_mode, n_orders, n_orders_not_canceled, n_late_orders,
   n_canceled_orders, late_rate, avg_delay_days, cancel_rate, total_revenue, revenue_at_risk (doanh thu của đơn trễ),
   is_reliable (đủ >= 100 đơn), risk_score (0-1), risk_rank, risk_rank_in_mode, risk_tier ('High','Medium','Low','Insufficient data').

3) `{DATASET}.route_scorecard_monthly` - mỗi dòng là một tuyến trong một tháng.
   Cột: route_key, route_name, order_region, shipping_mode, year_month ('YYYY-MM'), n_orders, late_rate,
   avg_delay_days, cancel_rate, revenue_at_risk.

4) `{DATASET}.route_risk_expected` - mỗi tuyến kèm kết quả mô hình dự báo XGBoost.
   Cột: route_name, order_region, shipping_mode, n_orders, n_scored_orders, actual_late_rate, predicted_late_rate,
   actual_revenue_at_risk, expected_revenue_at_risk (xác suất trễ x doanh thu), risk_score, risk_rank, risk_tier.

Giá trị cần biết:
- shipping_mode: 'First Class', 'Second Class', 'Same Day', 'Standard Class'
- market: 'Africa', 'Europe', 'LATAM', 'Pacific Asia', 'USCA'
- customer_segment: 'Consumer', 'Corporate', 'Home Office'
- 23 order_region, ví dụ 'Western Europe', 'Central America', 'South of  USA' (có HAI dấu cách).
"""

GLOSSARY = """
Thuật ngữ của đồ án (dùng để trả lời câu hỏi khái niệm, không cần SQL):
- Phương thức vận chuyển (shipping_mode) là HẠNG DỊCH VỤ giao hàng, không phải hãng vận chuyển. Có 4 hạng,
  kèm số ngày giao doanh nghiệp cam kết với khách: Same Day (giao trong ngày, cam kết 0 ngày), First Class (cam kết 1 ngày),
  Second Class (cam kết 2 ngày), Standard Class (hạng tiêu chuẩn, cam kết 4 ngày).
- Trong dữ liệu, First Class luôn giao trễ vì thực tế mất khoảng 2 ngày trong khi chỉ cam kết 1 ngày.
- Giao trễ: số ngày giao thực tế lớn hơn số ngày cam kết (late_delivery_risk = 1).
- Tỷ lệ giao trễ: số đơn trễ chia cho số đơn không bị hủy, tính ở cấp đơn hàng.
- Tuyến: một khu vực nhận hàng kết hợp một phương thức vận chuyển, ví dụ "Western Europe - First Class". Có 92 tuyến
  (23 khu vực x 4 phương thức).
- Điểm rủi ro (risk_score, 0-1): 50% tỷ lệ trễ + 30% số ngày trễ trung bình + 20% doanh thu bị ảnh hưởng, mỗi phần chuẩn hóa về 0-1.
- Mức rủi ro: Cao nếu điểm từ 0,66; Trung bình nếu từ 0,33; Thấp nếu dưới 0,33; Thiếu dữ liệu nếu tuyến dưới 100 đơn.
- Doanh thu chịu rủi ro: tổng doanh thu của các đơn giao trễ. Doanh thu rủi ro kỳ vọng: xác suất trễ do mô hình
  XGBoost dự báo nhân với giá trị đơn.
- Dữ liệu: bộ DataCo Smart Supply Chain (Kaggle), đơn đặt từ 01/2015 đến 01/2018, xử lý qua các tầng Bronze, Silver, Gold trên BigQuery.
"""

RULES = f"""
Quy tắc viết SQL (BigQuery Standard SQL):
- Chỉ viết MỘT câu SELECT (được dùng WITH). Không bao giờ sửa, xóa, tạo bảng.
- Luôn ghi tên bảng đầy đủ trong dấu backtick, ví dụ `{DATASET}.wide_order_items`.
- Đếm số đơn hàng dùng COUNT(DISTINCT order_id), KHÔNG đếm số dòng.
- Tỷ lệ giao trễ phải tính ở cấp ĐƠN HÀNG, mẫu số là đơn KHÔNG bị hủy. Mẫu:
    WITH o AS (
      SELECT order_id, <các cột nhóm>, MAX(late_delivery_risk) AS is_late, LOGICAL_OR(is_canceled) AS is_canceled,
             SUM(order_item_total) AS revenue
      FROM `{DATASET}.wide_order_items` GROUP BY order_id, <các cột nhóm>)
    SELECT <các cột nhóm>, COUNT(*) AS so_don,
           ROUND(SAFE_DIVIDE(COUNTIF(is_late = 1), COUNTIF(NOT is_canceled)), 4) AS ty_le_tre
    FROM o GROUP BY <các cột nhóm>
- KHÔNG BAO GIỜ tính tỷ lệ trễ bằng AVG(late_delivery_risk) trên từng dòng hàng (sai vì đơn nhiều sản phẩm bị đếm nhiều lần).
- Tỷ lệ trễ theo phương thức, khu vực, tuyến (hoặc toàn bộ): dùng `{DATASET}.route_scorecard`, cộng số đếm rồi mới chia:
    SELECT shipping_mode, SUM(n_orders) AS so_don,
           ROUND(SAFE_DIVIDE(SUM(n_late_orders), SUM(n_orders_not_canceled)), 4) AS ty_le_tre
    FROM `{DATASET}.route_scorecard` GROUP BY shipping_mode
  Theo thị trường thì dùng mẫu WITH o AS (...) ở trên trên wide_order_items (vì route_scorecard không có cột market).
- Câu hỏi về tuyến, điểm rủi ro, xếp hạng, tiền rủi ro: ưu tiên route_scorecard hoặc route_risk_expected.
  Khi xếp hạng tuyến, chỉ lấy tuyến có risk_tier <> 'Insufficient data'.
- Đặt tên cột kết quả bằng tiếng Việt không dấu, ngắn gọn (vd: phuong_thuc, so_don, ty_le_tre, tien_rui_ro).
- Tỷ lệ để ở dạng 0-1, làm tròn 4 chữ số. Tiền làm tròn 0 chữ số.
- Luôn có ORDER BY hợp lý và LIMIT tối đa 50.

Trả về DUY NHẤT một JSON:
- Nếu câu hỏi cần dữ liệu: {{"sql": "<câu SQL>", "ten_cot": {{"<tên cột trong SQL>": "<tên tiếng Việt có dấu>"}}}}
  Trong "ten_cot", ghi tên hiển thị tiếng Việt có dấu, ngắn gọn cho TỪNG cột kết quả,
  vd {{"phuong_thuc": "Phương thức", "so_don": "Số đơn", "ty_le_tre": "Tỷ lệ trễ"}}.
- Nếu là câu hỏi khái niệm (vd "First Class là gì", "điểm rủi ro tính thế nào"): trả lời dựa trên phần Thuật ngữ, 2-3 câu ngắn:
  {{"sql": null, "reply": "<câu trả lời>"}}
- Nếu là lời chào hoặc dữ liệu không có thông tin để trả lời: {{"sql": null, "reply": "<trả lời ngắn, gợi ý câu hỏi phù hợp>"}}
- Câu trả lời trong "reply" viết bình tĩnh, không dùng dấu gạch dài.
"""

ANSWER_RULES = """
Bạn là trợ lý phân tích giao hàng. Dựa vào câu hỏi và bảng kết quả, trả lời bằng tiếng Việt:
- Tối đa 3 câu ngắn, đi thẳng vào con số chính và một nhận xét có ích cho người vận hành.
- Số viết kiểu Việt Nam: 65.752 đơn, 57,3%. Tỷ lệ 0-1 thì đổi ra % và làm tròn 1 chữ số. Tiền ghi USD.
- Giọng bình tĩnh, khách quan như người phân tích. Không dùng từ cảm xúc như "báo động", "khẩn cấp", "cực kỳ".
- Bối cảnh giúp giải thích: First Class cam kết giao 1 ngày nhưng thực tế luôn mất khoảng 2 ngày, nên vấn đề nằm ở mức cam kết
  chứ không hẳn ở vận hành. Chỉ nhắc khi liên quan.
- Chỉ dùng số có trong bảng kết quả, không bịa thêm. Nếu bảng rỗng thì nói không tìm thấy dữ liệu phù hợp.
- Không dùng tiêu đề, không gạch đầu dòng, không dùng dấu gạch dài, chỉ dùng gạch ngắn "-" nếu cần.
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
    return p[0] + (f" (dự phòng: {', '.join(p[1:])})" if len(p) > 1 else "")


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
            return ("Khóa Gemini hợp lệ nhưng project Google của khóa không dùng được gói miễn phí. "
                    "Hãy bật thanh toán cho project đó, tạo khóa ở project khác, hoặc dùng GROQ_API_KEY.")
        return "Khóa AI không hợp lệ hoặc không có quyền. Hãy kiểm tra lại khóa trong Secrets."
    if _is_busy(e):
        return "AI đang quá tải hoặc hết lượt miễn phí. Cậu đợi khoảng 1 phút rồi hỏi lại nhé."
    return f"Không gọi được AI: {str(e)[:300]}"


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
    raise last or TimeoutError("timeout: AI phản hồi quá lâu")


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
    raise last or TimeoutError("timeout: AI phản hồi quá lâu")


def _generate(system, prompt, json_mode):
    """Gọi AI theo thứ tự providers(); cái trước lỗi thì chuyển sang cái sau."""
    order = providers()
    if not order:
        raise RuntimeError("Chưa có API key của AI trong secrets.toml.")
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
    prompt = (f"Ngữ cảnh hội thoại gần đây (dùng nếu câu hỏi mới nói tiếp ý trước):\n{ctx or '(không có)'}\n\n"
              f"Câu hỏi mới: {question}")
    if error:
        prompt += f"\n\nCâu SQL trước bị lỗi:\n{bad_sql}\nLỗi: {error}\nHãy viết lại câu SQL đúng."
    return _generate(SCHEMA + GLOSSARY + RULES, prompt, json_mode=True)


def ask_sql(question, history, error=None, bad_sql=None):
    """Trả về dict {"sql": ...} hoặc {"sql": None, "reply": ...}. Câu hỏi giống hệt sẽ lấy từ bộ nhớ đệm."""
    ctx = "".join(f"- Câu hỏi trước: {h['q']}\n  SQL đã dùng: {h['sql']}\n" for h in history[-2:] if h.get("sql"))
    text = _ask_sql_cached(question.strip(), ctx, error, bad_sql)
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        out = json.loads(m.group(0)) if m else {"sql": None, "reply": text}
    return out if isinstance(out, dict) else {"sql": None, "reply": str(out)}


@st.cache_data(ttl=3600, show_spinner=False)
def ask_answer(question, sql, df):
    table = df.head(30).to_csv(index=False)
    prompt = f"Câu hỏi: {question}\n\nSQL đã chạy:\n{sql}\n\nKết quả ({len(df)} dòng, tối đa 30 dòng đầu):\n{table}"
    return _generate(ANSWER_RULES + GLOSSARY, prompt, json_mode=False).strip().replace("—", "-").replace("–", "-")


# ------------------------------------------------------------------ 2. KIỂM TRA SQL
FORBIDDEN = re.compile(r"\b(INSERT|UPDATE|DELETE|MERGE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE|EXPORT|LOAD|"
                       r"CALL|DECLARE|SET|BEGIN|COMMIT|ROLLBACK|EXECUTE)\b", re.I)


def check_sql(sql):
    """Chỉ cho phép một câu SELECT đọc từ 4 bảng Gold. Sai thì báo lỗi (ValueError)."""
    s = re.sub(r"^```(?:sql)?|```$", "", (sql or "").strip(), flags=re.I).strip().rstrip(";").strip()
    no_str = re.sub(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"", "''", s)      # bỏ chuỗi để không bắt nhầm chữ
    no_str = re.sub(r"--[^\n]*|/\*.*?\*/", " ", no_str, flags=re.S)          # bỏ chú thích
    if not s:
        raise ValueError("SQL rỗng.")
    if ";" in no_str:
        raise ValueError("Chỉ được chạy một câu lệnh.")
    if not re.match(r"(?is)^\s*(WITH|SELECT)\b", no_str):
        raise ValueError("Chỉ được dùng câu SELECT.")
    if FORBIDDEN.search(no_str):
        raise ValueError("Câu SQL chứa lệnh không được phép.")
    scan = re.sub(r"(?is)EXTRACT\s*\([^()]*\)", " ", no_str)                 # EXTRACT(YEAR FROM x) không phải bảng
    ctes = {c.lower() for c in re.findall(r"(?i)(?:\bWITH|,)\s*([A-Za-z_]\w*)\s+AS\s*\(", scan)}
    refs = re.findall(r"(?i)\b(?:FROM|JOIN)\s+(`[^`]+`|[A-Za-z_][\w.\-]*)", scan)
    refs += [f"`{q}`" for q in re.findall(r"`([^`]*\.[^`]*)`", scan)]   # mọi tên bảng trong backtick
    if not refs:
        raise ValueError("Không thấy bảng dữ liệu trong câu SQL.")
    for ref in refs:
        name = ref.strip("`")
        if name.lower() in ctes or name.upper() == "UNNEST":
            continue
        parts = name.split(".")
        ok = (parts[-1] in ALLOWED_TABLES and
              (len(parts) == 1 or (len(parts) == 2 and parts[0] == "gold") or
               (len(parts) == 3 and parts[0] == PROJECT and parts[1] == "gold")))
        if not ok:
            raise ValueError(f"Bảng {name} không nằm trong danh sách được phép.")
    return s


# ------------------------------------------------------------------ 3. CHẠY TRÊN BIGQUERY
@st.cache_data(ttl=3600, show_spinner=False)
def run_sql(sql):
    from google.cloud import bigquery
    client = _bigquery_client()
    if client is None:
        raise RuntimeError("Chưa cấu hình khóa BigQuery trong secrets.toml.")
    job = client.query(sql, job_config=bigquery.QueryJobConfig(
        maximum_bytes_billed=MAX_BYTES, default_dataset=DATASET))   # quét quá 1 GB thì BigQuery tự từ chối
    return job.result(timeout=TIMEOUT).to_dataframe(create_bqstorage_client=False).head(MAX_ROWS)


def answer(question, history, step=lambda text: None):
    """Xử lý trọn một câu hỏi. step(text) dùng để báo bước đang chạy lên giao diện."""
    step("Đang viết truy vấn SQL...")
    plan = ask_sql(question, history)
    if not plan.get("sql"):
        reply = plan.get("reply") or "Mình chưa hiểu câu hỏi, cậu thử hỏi lại nhé."
        return {"q": question, "text": reply.replace("—", "-").replace("–", "-")}
    sql, error = plan["sql"], None
    for attempt in range(2):                    # lần đầu + 1 lần nhờ AI sửa lỗi
        try:
            sql = check_sql(sql)
            step("Đang chạy trên BigQuery...")
            df = run_sql(sql)
            step("Đang tóm tắt kết quả...")
            return {"q": question, "sql": sql, "df": df, "labels": plan.get("ten_cot") or {},
                    "text": ask_answer(question, sql, df)}
        except RuntimeError:                    # lỗi gọi AI (quá tải, hết giờ) -> báo luôn, không thử lại
            raise
        except Exception as e:
            error = str(e)[:500]
            if attempt == 1:
                break
            step("Truy vấn bị lỗi, đang nhờ AI sửa...")
            plan = ask_sql(question, history, error=error, bad_sql=sql)
            if not plan.get("sql"):
                break
            sql = plan["sql"]
    return {"q": question, "sql": sql, "text": f"Mình chưa chạy được truy vấn cho câu này. Lỗi: {error}"}


# ------------------------------------------------------------------ 4. HIỂN THỊ
# Tên cột hay gặp -> tên tiếng Việt có dấu (dùng khi AI không trả về "ten_cot")
COL_VI = {
    "phuong_thuc": "Phương thức", "shipping_mode": "Phương thức", "thi_truong": "Thị trường", "market": "Thị trường",
    "khu_vuc": "Khu vực", "order_region": "Khu vực", "quoc_gia": "Quốc gia", "order_country": "Quốc gia",
    "tuyen": "Tuyến", "route_name": "Tuyến", "nhom_hang": "Nhóm hàng", "category_name": "Nhóm hàng",
    "bo_phan": "Bộ phận", "department_name": "Bộ phận", "phan_khuc": "Phân khúc khách hàng",
    "customer_segment": "Phân khúc khách hàng", "thanh_toan": "Hình thức thanh toán", "payment_type": "Hình thức thanh toán",
    "nam": "Năm", "order_year": "Năm", "thang": "Tháng", "order_year_month": "Tháng", "year_month": "Tháng",
    "quy": "Quý", "gio": "Giờ đặt hàng", "thu": "Thứ trong tuần", "cuoi_tuan": "Cuối tuần",
    "so_don": "Số đơn", "so_don_tre": "Số đơn trễ", "so_don_huy": "Số đơn hủy", "n_orders": "Số đơn",
    "ty_le_tre": "Tỷ lệ trễ", "late_rate": "Tỷ lệ trễ", "ty_le_huy": "Tỷ lệ hủy", "cancel_rate": "Tỷ lệ hủy",
    "tien_rui_ro": "Tiền rủi ro (USD)", "revenue_at_risk": "Doanh thu chịu rủi ro (USD)",
    "expected_revenue_at_risk": "Tiền rủi ro kỳ vọng (USD)", "doanh_thu": "Doanh thu (USD)",
    "risk_score": "Điểm rủi ro", "diem_rui_ro": "Điểm rủi ro", "risk_rank": "Hạng", "hang": "Hạng",
    "risk_tier": "Mức rủi ro", "muc_rui_ro": "Mức rủi ro", "so_ngay_tre_tb": "Số ngày trễ trung bình",
    "avg_delay_days": "Số ngày trễ trung bình", "ma_don": "Mã đơn", "order_id": "Mã đơn",
}


def vi_columns(df, labels):
    """Đổi tên cột kết quả sang tiếng Việt: ưu tiên tên AI đặt, sau đó bảng COL_VI, cuối cùng bỏ dấu gạch dưới."""
    labels = labels or {}
    def name(c):
        return labels.get(c) or COL_VI.get(c) or c.replace("_", " ").capitalize()
    out = df.rename(columns={c: name(c) for c in df.columns})
    if out.columns.duplicated().any():          # tránh trùng tên sau khi đổi
        out.columns = [f"{c} ({i})" if out.columns[:i].tolist().count(c) else c for i, c in enumerate(out.columns)]
    return out


def auto_chart(df, raw_cols):
    """Vẽ biểu đồ cột nếu kết quả có 1 cột chữ + ít nhất 1 cột số và 2-25 dòng.
    df đã đổi tên cột tiếng Việt; raw_cols là tên cột gốc (để nhận biết cột tỷ lệ)."""
    if df is None or not 2 <= len(df) <= 25:
        return None
    raw = dict(zip(df.columns, raw_cols))
    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(text_cols) != 1 or not num_cols:
        return None
    cat = text_cols[0]
    is_rate_name = lambda c: "ty_le" in raw[c] or "rate" in raw[c] or "Tỷ lệ" in c
    val = next((c for c in num_cols if is_rate_name(c)), num_cols[-1])
    d = df[[cat, val]].copy()
    d[cat] = d[cat].astype(str)
    is_rate = d[val].between(0, 1).all() and is_rate_name(val)
    d["Cao nhất"] = d[val] == d[val].max()
    order = d.sort_values(val, ascending=False)[cat].tolist()
    rc = raw[cat]
    if any(k in rc for k in ("thang", "month", "year", "quy")) or rc in ("nam", "gio", "thu"):
        order = sorted(d[cat].tolist())
    bars = alt.Chart(d).mark_bar(cornerRadius=8, size=16).encode(
        y=alt.Y(field=cat, type="nominal", sort=order, title=None,
                axis=alt.Axis(labelColor=INK, labelLimit=220, labelOverlap=False)),
        x=alt.X(field=val, type="quantitative", title=None, axis=alt.Axis(format="%" if is_rate else "~s")),
        color=alt.condition("datum['Cao nhất']", alt.value(ACCENT), alt.value("#F2A6A4")),
        tooltip=[alt.Tooltip(field=cat, type="nominal"),
                 alt.Tooltip(field=val, type="quantitative", format=".1%" if is_rate else ",.2f")])
    return clean(bars, max(150, 40 * len(d)))


def show_reply(msg):
    """Câu trả lời + biểu đồ (nếu hợp) + bảng số liệu + câu SQL đã chạy để người dùng kiểm chứng."""
    st.markdown(msg["text"])
    df = msg.get("df")
    if df is not None and len(df):
        view = vi_columns(df, msg.get("labels"))
        chart = auto_chart(view, list(df.columns))
        if chart is not None:
            st.altair_chart(chart, width="stretch")
        rates = {v: st.column_config.NumberColumn(v, format="percent") for c, v in zip(df.columns, view.columns)
                 if ("ty_le" in c or "rate" in c) and pd.api.types.is_numeric_dtype(df[c]) and df[c].between(0, 1).all()}
        with st.expander(f"Bảng số liệu ({len(df)} dòng)", expanded=True):
            st.dataframe(view, hide_index=True, width="stretch", column_config=rates)
    if msg.get("sql"):
        with st.expander("Câu SQL đã chạy", expanded=True):
            st.code(msg["sql"], language="sql")
            st.markdown('<div class="sql-note">Câu SQL được kiểm tra trước khi chạy: chỉ đọc (SELECT) '
                        'trên 4 bảng Gold, quét tối đa 1 GB.</div>', unsafe_allow_html=True)


def page_chatbot():
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    st.markdown('<div class="ov-title">Trợ lý dữ liệu</div>', unsafe_allow_html=True)
    st.write("")

    try:
        ready = provider() is not None and _bigquery_client() is not None
    except Exception:
        ready = False
    if not ready:
        st.warning("Chưa có API key của AI (Groq hoặc Gemini) hoặc khóa BigQuery trong file .streamlit/secrets.toml. "
                   "Hãy chạy `python make_secrets.py` rồi mở lại app.")
        return

    history = st.session_state.setdefault("chat", [])
    with st.sidebar:
        st.write("")
        if st.button("Xóa cuộc trò chuyện", icon=":material/delete:", width="stretch"):
            history.clear()
            st.rerun()

    question = st.chat_input("Ví dụ: Tỷ lệ trễ của Second Class ở châu Âu là bao nhiêu?") \
        or st.session_state.pop("picked_q", None)

    if not history and not question:
        st.markdown('<div class="chat-hello"><h4>Cậu muốn biết gì về tình hình giao hàng?</h4>'
                    '<p>Bấm một câu gợi ý bên dưới hoặc tự gõ câu hỏi ở cuối trang.</p></div>',
                    unsafe_allow_html=True)

    def _pick():                                 # bấm một câu gợi ý -> dùng nó làm câu hỏi
        st.session_state["picked_q"] = st.session_state["sample_pills"]
        st.session_state["sample_pills"] = None

    st.pills("Gợi ý", SAMPLES, selection_mode="single", key="sample_pills",
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
            with st.status("Đang xử lý câu hỏi...", expanded=False) as status:
                try:
                    msg = answer(question, history, step=lambda t: status.update(label=t))
                    status.update(label="Xong", state="complete")
                except Exception as e:
                    msg = {"q": question, "text": str(e) if isinstance(e, RuntimeError) else f"Có lỗi khi xử lý: {e}"}
                    status.update(label="Không hoàn thành", state="error")
            show_reply(msg)
        history.append(msg)
      
