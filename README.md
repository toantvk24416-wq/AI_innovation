# OnTime AI - Cảnh báo sớm giao hàng trễ

Ứng dụng Streamlit cho đồ án **OnTime AI** (AI Innovation). OnTime AI giúp cửa hàng thương mại điện tử
**hẹn ngày giao sát thực tế**, phát hiện sớm đơn có nguy cơ trễ, giải thích lý do và giúp nhân viên chủ động báo khách.
Dữ liệu: bộ DataCo Smart Supply Chain (2015-2018), xử lý qua tầng Bronze, Silver, Gold trên BigQuery.

## Các màn hình và yêu cầu tương ứng

| Màn hình | Làm gì | Yêu cầu (Sheet PCCV / Mục 5) |
|---|---|---|
| Overview (Tổng quan) | Tỷ lệ trễ theo thời gian, phương thức, thị trường | Bản demo có sẵn |
| Route ranking (Xếp hạng tuyến) | Bản đồ tuyến, ma trận ưu tiên, bản đồ nhiệt, bảng 92 tuyến | Wireframe 9.1 (bảng tuyến), Mục 5.2 |
| New order prediction (Dự báo đơn mới) | Form nhập đơn, xác suất trễ, **ngày hẹn đề xuất P80**, lý do SHAP, việc nên làm | 9.2 (form nhập đơn), 9.4 (ngày hẹn đề xuất) |
| **Order assistant (Trợ lý đơn hàng)** | Hàng đợi đơn Cao/Trung bình xếp theo tiền rủi ro kỳ vọng, chi tiết đơn, tin nhắn nháp, nút Phê duyệt / Từ chối, nhật ký duyệt, 3 đơn mẫu để demo | 5.2, 5.5, wireframe 9.1 (hàng đợi, chi tiết đơn), 9.2 (3 đơn demo) |
| **Promise and evaluation (Ngày hẹn và đánh giá)** | Bảng P50/P80/P90, mô hình phân vị so với bảng phân vị, backtest các cách hẹn, Precision@K, SHAP toàn cục | 5.1, 6.1, 6.2 |
| **Savings estimate (Ước tính tiết kiệm)** | 3 kịch bản thấp/cơ sở/cao, tính theo mục 7.4 của báo cáo (chi phí 1 đơn trễ = phút xử lý + hoàn hàng) | 9.4, dùng cho 7.1 và 7.4 |
| Data assistant (Trợ lý dữ liệu) | Hỏi bằng tiếng Anh, AI viết SQL chỉ đọc trên tầng Gold, hiện câu trả lời + biểu đồ + bảng + câu SQL | 5.6, wireframe 9.1 (chatbot) |

Các hàm theo tab **Hợp đồng** của Sheet nằm trong `ontime/core.py`:

| Hàm | Đầu ra |
|---|---|
| `score_order(order)` | `{p_late, risk_level, rule_applied}`; First Class đi theo luật R1 |
| `recommend_promise(order)` | `{p50, p80, p90, promise_days, delta_vs_cu}` |
| `explain_order(order)` | `{p_late, reasons: [{rank, feature, label_vi, shap}]}` |
| `assist_order(order)` | `{giai_thich, hanh_dong, tin_nhan_nhap, nguon}` |
| `ask(question)` | Text-to-SQL (dùng `chatbot.answer`) |

`ontime/llm.py` là khung gọi LLM của Mục 5.4. Mỗi lần soạn đi qua các bước sau:
- che tên và địa chỉ khách bằng `[KHACH_HANG]`, `[DIA_CHI]` trước khi gửi;
- chờ tối đa 8 giây, quá thì thử lại 1 lần;
- kiểm tra đầu ra: đúng khuôn JSON, mỗi trường tối đa 3 câu, không có cụm từ cấm, **mọi con số phải có trong dữ liệu của đơn**;
- lỗi ở bất kỳ bước nào thì dùng câu mẫu.

Bảng luật hành động nằm ở `data/luat_hanh_dong.csv`. Đổi chính sách chỉ cần sửa file này, không cần sửa code:
- R1–R4 lấy theo tab Luật của Sheet;
- R5 (mức Trung bình: gửi tin xác nhận ngày giao P80) lấy theo bảng ở Mục 5.5.

## Kết quả (tập kiểm tra: đơn đặt từ 10/07/2017)

Dữ liệu dựng lại khớp tab Số liệu:
- 65.752 đơn, 62.897 đơn không hủy, 36.048 đơn trễ, 172.765 dòng hàng không hủy;
- ROC-AUC XGBoost **0,7838**;
- tập kiểm tra 12.581 đơn (Sheet ghi 12.580).

**Backtest các cách hẹn ngày giao (việc 6.1)**, tính trên 12.581 đơn:

| Cách hẹn | Trễ so với ngày hẹn | Hẹn thêm trung bình |
|---|---|---|
| Như cũ | 57,5% | 0 ngày |
| First Class hẹn 2 ngày | 42,4% | +0,15 ngày |
| Hẹn theo P50 | 34,1% | +0,55 ngày |
| **Hẹn theo P80** | **3,5%** | **+2,00 ngày** |
| Hẹn theo P90 | 0,0% | +2,19 ngày |

Số ngày giao thực tế trong DataCo là số nguyên từ 0 đến 6. Vì vậy hẹn theo P80 (làm tròn lên) phủ nhiều hơn 80% đơn: gần như hết trễ hẹn, nhưng ngày hẹn dài thêm khoảng 2 ngày. Ví dụ Second Class từ 2 lên khoảng 5,4 ngày. Nhóm cần cân nhắc cái giá này khi viết phần giá trị kinh doanh.

**Mô hình ngày hẹn (Mục 5.1):**
- Mô hình hồi quy phân vị (gradient boosting, hàm mất mát pinball) **không vượt** bảng phân vị lịch sử theo tuyến × phương thức. Pinball loss ở P80: mô hình 0,333, bảng 0,324.
- Theo nguyên tắc ở Mục 5.1, app dùng **bảng phân vị** để đề xuất ngày hẹn. Mô hình vẫn được lưu trong `models/`.

**Precision@K của hàng đợi (việc 6.2)**, mỗi ngày xử lý K đơn đầu hàng đợi:
- Tính cả First Class: Top 10 đúng 97%, Top 50 đúng 63% (tỷ lệ trễ nền 57,5%).
- Bỏ First Class (đã xử lý bằng luật): Top 10 đúng 81%, Top 50 đúng 52% (tỷ lệ trễ nền 49,9%).

## Chạy trên máy

```bash
pip install -r requirements.txt
streamlit run app.py
```

Không cấu hình gì thêm thì các trang dùng file CSV có sẵn. Muốn dùng BigQuery và chatbot, chạy `python make_secrets.py` để tạo
`.streamlit/secrets.toml` (khóa service account + `GEMINI_API_KEY` hoặc `GROQ_API_KEY`).
**Không đưa file này lên GitHub.** Không có khóa AI thì Trợ lý đơn hàng tự dùng câu mẫu.

## Dựng lại dữ liệu cho phần AI

Các file trong `data/` và `models/` được tạo bởi:

```bash
python scripts/build_artifacts.py "đường/dẫn/DataCoSupplyChainDataset.csv"
```

Script làm các việc sau:
- dịch tên nước từ tiếng Tây Ban Nha sang tiếng Anh như tầng Silver;
- kiểm chứng các số trong tab Số liệu;
- huấn luyện mô hình P50/P80/P90 và so với bảng phân vị;
- tính SHAP bằng `pred_contribs` của XGBoost, gộp các cột one-hot về 18 biến gốc;
- chạy backtest và Precision@K;
- ghi `order_queue.csv`, `promise_table.csv`, `evaluation.json`.

## Cấu trúc thư mục

```
app.py              điểm vào, menu và trang New order prediction
overview.py         trang Overview, giao diện chung
ranking.py          trang Route ranking
assistant.py        trang Order assistant
promise.py          trang Promise and evaluation
savings.py          trang Savings estimate
chatbot.py          trang Data assistant (Text-to-SQL)
ontime/             các hàm AI theo tab Hợp đồng, khung LLM, nhật ký duyệt
scripts/            script dựng dữ liệu
data/, models/      dữ liệu và mô hình đã dựng sẵn
best_pipe.joblib    mô hình XGBoost cảnh báo trễ (lưu lại để chạy với xgboost 3.4.1)
```

## Giới hạn

- Dữ liệu DataCo không phải dữ liệu Việt Nam. Khi triển khai cần huấn luyện lại bằng đơn của cửa hàng.
- Mô hình cảnh báo chỉ hơn cách đoán theo phương thức khoảng 0,036 ROC-AUC và bỏ sót khoảng 40% đơn trễ.
- Tỷ lệ trễ 100% của First Class phản ánh cách đặt hạn, chưa chứng minh First Class gây trễ.
- Hàng đợi demo dùng các đơn trong giai đoạn kiểm tra (07/2017 - 01/2018). Kết quả chưa được ghi ngược vào bảng `gold.order_late_predictions` trên BigQuery.
- Nhật ký duyệt lưu SQLite cạnh app. Trên Streamlit Cloud, nhật ký sẽ mất khi app khởi động lại.
- Chưa làm: bộ 30 câu hỏi kiểm thử chatbot (việc 6.3), ánh xạ cột sang file xuất GHN/GHTK (việc 7.2).
