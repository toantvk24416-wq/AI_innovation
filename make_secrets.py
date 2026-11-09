"""Tạo file .streamlit/secrets.toml từ file khóa JSON của service account + Gemini API key.

Cách chạy (trong thư mục streamlit_app):  python make_secrets.py
Chỉ cần chạy một lần. File secrets.toml chứa khóa bí mật: KHÔNG đưa lên GitHub, KHÔNG gửi cho ai.
"""
import json
from pathlib import Path

key_path = Path(input("Kéo thả file khóa JSON vào đây (hoặc dán đường dẫn) rồi nhấn Enter: ").strip().strip('"').strip("'"))
info = json.loads(key_path.read_text(encoding="utf-8"))
gemini = input("Dán Gemini API key rồi nhấn Enter: ").strip()

lines = [f"GEMINI_API_KEY = {json.dumps(gemini)}", "", "[gcp_service_account]"]
lines += [f"{k} = {json.dumps(v)}" for k, v in info.items()]

out = Path(__file__).parent / ".streamlit" / "secrets.toml"
out.parent.mkdir(exist_ok=True)
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Đã tạo {out}")
print(f"Service account: {info.get('client_email')}  |  Project: {info.get('project_id')}")
