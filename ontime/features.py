"""Phần dùng chung giữa script dựng dữ liệu và app: 18 biến đầu vào, mức rủi ro, SHAP và nhãn lý do tiếng Việt."""
import numpy as np
import pandas as pd
import xgboost as xgb

CAT_COLS = ["shipping_mode", "market", "order_region", "order_country",
            "category_name", "department_name", "customer_segment", "payment_type"]
NUM_COLS = ["scheduled_shipping_days", "quantity", "unit_price", "sales", "discount_rate",
            "order_item_total", "order_month", "order_day_of_week", "order_is_weekend", "order_hour"]
FEATURES = CAT_COLS + NUM_COLS
QUANTILES = {"p50": 0.5, "p80": 0.8, "p90": 0.9}
MODEL_VERSION = "xgb_2026-11"
DOW_VI = {1: "Chủ nhật", 2: "thứ Hai", 3: "thứ Ba", 4: "thứ Tư", 5: "thứ Năm", 6: "thứ Sáu", 7: "thứ Bảy"}


def risk_tier(p):
    """Mức rủi ro của đơn: Cao từ 60%, Trung bình từ 30%, Thấp dưới 30%."""
    p = np.asarray(p, dtype=float)
    out = np.where(p >= 0.6, "Cao", np.where(p >= 0.3, "Trung bình", "Thấp"))
    return out if out.ndim else str(out)


def reason_label(feature, r):
    """Từ điển đổi biến gốc thành câu tiếng Việt (quy tắc, không dùng LLM, để cùng yếu tố luôn cùng tên gọi)."""
    m, d = r["shipping_mode"], int(r["scheduled_shipping_days"])
    return {
        "shipping_mode": f"{m} hẹn giao {d} ngày, thường bị vượt",
        "scheduled_shipping_days": f"Hạn giao {d} ngày ngắn so với thực tế của tuyến",
        "order_hour": f"Đặt hàng lúc {int(r['order_hour'])}h",
        "order_country": f"Giao tới {r['order_country']}",
        "order_region": f"Khu vực nhận {r['order_region']}",
        "market": f"Thị trường {r['market']}",
        "category_name": f"Nhóm hàng {r['category_name']}",
        "department_name": f"Bộ phận {r['department_name']}",
        "customer_segment": f"Khách hàng nhóm {r['customer_segment']}",
        "payment_type": f"Thanh toán {r['payment_type']}",
        "order_month": f"Đặt trong tháng {int(r['order_month'])}",
        "order_day_of_week": f"Đặt vào {DOW_VI[int(r['order_day_of_week'])]}",
        "order_is_weekend": "Đặt vào cuối tuần" if r["order_is_weekend"] else "Đặt vào ngày thường",
        "quantity": f"Số lượng {int(r['quantity'])} sản phẩm",
        "unit_price": f"Đơn giá {r['unit_price']:.2f} USD",
        "sales": f"Doanh số dòng hàng {r['sales']:.2f} USD",
        "order_item_total": f"Giá trị dòng hàng {r['order_item_total']:.2f} USD",
        "discount_rate": f"Giảm giá {r['discount_rate'] * 100:.0f}%",
    }[feature]


def shap_by_feature(pipe, X):
    """SHAP chính xác từ XGBoost (pred_contribs, tương đương TreeExplainer), gộp các cột one-hot về 18 biến gốc."""
    pre, clf = pipe[:-1], pipe.steps[-1][1]
    names = list(pre.get_feature_names_out())
    contrib = clf.get_booster().predict(xgb.DMatrix(pre.transform(X[FEATURES]), feature_names=names),
                                        pred_contribs=True)[:, :-1]
    owner = []
    for n in names:
        kind, rest = n.split("__", 1)
        owner.append(rest if kind == "num" else next(c for c in CAT_COLS if rest.startswith(c + "_")))
    out = pd.DataFrame(contrib, columns=names).T.groupby(owner).sum().T.reindex(columns=FEATURES, fill_value=0)
    out.index = X.index
    return out


def top_reasons(row, shap_row, k=3):
    """Tối đa k yếu tố làm TĂNG rủi ro (SHAP dương lớn nhất), kèm nhãn tiếng Việt."""
    pos = shap_row[shap_row > 0].sort_values(ascending=False).head(k)
    return [{"rank": i + 1, "feature": f, "label_vi": reason_label(f, row), "shap": round(float(v), 3)}
            for i, (f, v) in enumerate(pos.items())]
