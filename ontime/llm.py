"""Khung gọi LLM (Mục 5.4): phần mềm tính toán, AI chỉ diễn đạt.

Sáu bước cho mỗi lần soạn lời giải thích và tin nhắn nháp, chỉ bước 3 dùng AI:
  1. Lấy bản ghi JSON của đơn (xác suất trễ, ngày hẹn P80, lý do SHAP, hành động từ bảng luật)
  2. Che thông tin cá nhân bằng nhãn giữ chỗ [KHACH_HANG], [DIA_CHI], [SDT], [EMAIL]
  3. Gọi LLM (temperature thấp, tối đa 8 giây, thử lại 1 lần)
  4. Kiểm tra đầu ra: đúng khuôn JSON, tối đa 3 câu mỗi trường, không có cụm từ cấm,
     MỌI con số phải có trong bản ghi JSON (lệch một số là loại)
  5. Điền lại thông tin cá nhân trên máy chủ
  6. Nếu lỗi ở bất kỳ bước nào: dùng câu mẫu ghép từ JSON
"""
import concurrent.futures as cf
import json
import re

TIMEOUT_S = 8
BANNED = ("cam kết chắc chắn", "bồi thường", "chắc chắn giao")
PII_TAGS = {"customer_name": "[KHACH_HANG]", "customer_address": "[DIA_CHI]",
            "customer_phone": "[SDT]", "customer_email": "[EMAIL]"}

SYSTEM = """Bạn là trợ lý vận hành giao hàng. Chỉ dùng dữ liệu trong <du_lieu>.
1. Không tự tính, không làm tròn, không thêm con số ngoài <du_lieu>. Viết số đúng như trong <du_lieu>.
2. Không hẹn ngày giao khác ngay_hen_giao.
3. Không nêu lý do ngoài danh sách ly_do.
4. Thiếu dữ liệu thì trả lời: "Chưa đủ dữ liệu để giải thích đơn này."
5. Giữ nguyên các nhãn [KHACH_HANG], [DIA_CHI], [SDT], [EMAIL].
6. giai_thich viết cho nhân viên: vì sao đơn có rủi ro và nên làm gì (theo hanh_dong).
   tin_nhan_nhap viết cho khách: lịch sự, báo trước ngày giao mới, không nhắc xác suất hay lý do kỹ thuật.
Không dùng các cụm "cam kết chắc chắn", "bồi thường". Không dùng dấu gạch dài.
Trả về JSON {"giai_thich": "...", "tin_nhan_nhap": "..."}, mỗi trường tối đa 3 câu."""

_POOL = cf.ThreadPoolExecutor(max_workers=2)
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


# ------------------------------------------------------------------ bước 2 và 5: che, điền lại thông tin cá nhân
def mask(payload):
    """Trả về (bản ghi đã che, bảng ánh xạ nhãn -> giá trị thật). Bảng ánh xạ chỉ nằm trong bộ nhớ."""
    masked, mapping = dict(payload), {}
    for key, tag in PII_TAGS.items():
        if masked.get(key):
            mapping[tag] = str(masked[key])
            masked[key] = tag
    return masked, mapping


def unmask(text, mapping):
    for tag, value in mapping.items():
        text = text.replace(tag, value)
    return text


# ------------------------------------------------------------------ bước 4: kiểm tra đầu ra
def _norm(tok):
    """'76,6' -> '76.6', '1.234.567' -> '1234567', '2.0' -> '2' để so khớp số viết kiểu Việt hay kiểu Anh."""
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", tok):
        tok = tok.replace(".", "")
    tok = tok.replace(",", ".")
    try:
        return f"{float(tok):g}"
    except ValueError:
        return tok


def allowed_numbers(payload):
    return {_norm(t) for t in _NUM.findall(json.dumps(payload, ensure_ascii=False))}


def check_output(out, allowed):
    """Danh sách lỗi; rỗng nghĩa là đầu ra dùng được."""
    problems = []
    if not isinstance(out, dict) or not all(isinstance(out.get(k), str) and out[k].strip()
                                            for k in ("giai_thich", "tin_nhan_nhap")):
        return ["sai khuôn JSON"]
    for key in ("giai_thich", "tin_nhan_nhap"):
        text = out[key]
        if len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]) > 3:
            problems.append(f"{key} quá 3 câu")
        low = text.lower()
        problems += [f"có cụm từ cấm '{b}'" for b in BANNED if b in low]
        extra = sorted({t for t in _NUM.findall(text) if _norm(t) not in allowed})
        if extra:
            problems.append(f"{key} có số không có trong dữ liệu: {', '.join(extra)}")
    return problems


# ------------------------------------------------------------------ bước 3: gọi LLM
def call_llm(system, prompt, json_mode=True):
    """Gọi LLM qua các nhà cung cấp đã cấu hình cho chatbot (Gemini, Groq). Quá 8 giây thì thử lại 1 lần."""
    from chatbot import _generate          # dùng chung khóa API và cơ chế dự phòng nhà cung cấp với chatbot
    last = None
    for _ in range(2):
        try:
            return _POOL.submit(_generate, system, prompt, json_mode).result(timeout=TIMEOUT_S)
        except cf.TimeoutError:
            last = TimeoutError(f"LLM không phản hồi trong {TIMEOUT_S} giây")
        except Exception as e:      # chưa có khóa API, quá tải...: không thử lại
            raise e
    raise last


def compose(payload, template):
    """Chạy đủ 6 bước. template(payload) trả về câu mẫu dự phòng {"giai_thich", "tin_nhan_nhap"}."""
    masked, mapping = mask(payload)
    try:
        text = call_llm(SYSTEM, "<du_lieu>\n" + json.dumps(masked, ensure_ascii=False, indent=1) + "\n</du_lieu>")
        m = re.search(r"\{.*\}", text or "", re.S)
        out = json.loads(m.group(0)) if m else None
        problems = check_output(out, allowed_numbers(masked))
        if problems:
            raise ValueError("; ".join(problems))
        result = {k: unmask(out[k].replace("—", "-").replace("–", "-"), mapping) for k in ("giai_thich", "tin_nhan_nhap")}
        return {**result, "nguon": "LLM", "ghi_chu": "Đã qua kiểm tra số liệu và cụm từ cấm"}
    except Exception as e:
        result = {k: unmask(v, mapping) for k, v in template(masked).items()}
        return {**result, "nguon": "Câu mẫu", "ghi_chu": f"Dùng câu mẫu vì {str(e)[:160]}"}
