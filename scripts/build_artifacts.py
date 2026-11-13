"""Dựng dữ liệu cho các phần AI của OnTime AI (Mục 5) từ file DataCo gốc.

Cách chạy (một lần, trên máy có file dữ liệu gốc):
    python scripts/build_artifacts.py "đường/dẫn/DataCoSupplyChainDataset.csv"

Kết quả ghi vào thư mục data/ và models/ (app chỉ đọc các file này, không cần file gốc):
  models/promise_p50.joblib, promise_p80.joblib, promise_p90.joblib   mô hình hồi quy phân vị (5.1)
  data/promise_table.csv     bảng P50/P80/P90 theo tuyến x phương thức (mốc đối chiếu, 5.1)
  data/order_queue.csv       đơn trong giai đoạn kiểm tra: xác suất trễ, ngày hẹn, 3 lý do SHAP (5.2, 5.3)
  data/evaluation.json       số liệu đánh giá: AUC, độ phủ P80, pinball loss, backtest 3 cách hẹn (6.1), Precision@K (6.2)

Quy ước giống mô hình cảnh báo ở mục 3.4: chỉ dùng 18 biến biết lúc đặt hàng, chia theo thời gian tại 10/07/2017.
"""
import json

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ontime.features import (CAT_COLS, FEATURES, MODEL_VERSION, QUANTILES, risk_tier,  # noqa: E402
                             shap_by_feature, top_reasons)

ROOT = Path(__file__).resolve().parents[1]
SPLIT = pd.Timestamp("2017-07-10")
# Tầng Silver của đồ án dịch tên nước sang tiếng Anh; file gốc ghi bằng tiếng Tây Ban Nha.
ES_EN = {
    "Estados Unidos": "United States", "Francia": "France", "México": "Mexico", "Alemania": "Germany",
    "Brasil": "Brazil", "Reino Unido": "United Kingdom", "Italia": "Italy", "España": "Spain",
    "República Dominicana": "Dominican Republic", "Turquía": "Turkey", "Filipinas": "Philippines",
    "Panamá": "Panama", "Países Bajos": "Netherlands", "Nueva Zelanda": "New Zealand", "Irán": "Iran",
    "Egipto": "Egypt", "Marruecos": "Morocco", "SudAfrica": "South Africa", "Rusia": "Russia",
    "República Democrática del Congo": "Democratic Republic of the Congo", "Irak": "Iraq", "Ucrania": "Ukraine",
    "Suecia": "Sweden", "Tailandia": "Thailand", "Arabia Saudí": "Saudi Arabia", "Perú": "Peru",
    "Japón": "Japan", "Pakistán": "Pakistan", "Bélgica": "Belgium", "Polonia": "Poland",
    "Corea del Sur": "South Korea", "Malasia": "Malaysia", "Haití": "Haiti", "Argelia": "Algeria",
    "Irlanda": "Ireland", "Bangladés": "Bangladesh", "Rumania": "Romania", "Singapur": "Singapore",
    "Myanmar (Birmania)": "Myanmar", "Noruega": "Norway", "Suiza": "Switzerland", "Finlandia": "Finland",
    "Dinamarca": "Denmark", "Camerún": "Cameroon", "Kazajistán": "Kazakhstan", "Costa de Marfil": "Côte d'Ivoire",
    "Sudán": "Sudan", "Kenia": "Kenya", "Bielorrusia": "Belarus", "Zimbabue": "Zimbabwe",
    "República Checa": "Czech Republic", "Hungría": "Hungary", "Afganistán": "Afghanistan", "Camboya": "Cambodia",
    "Trinidad y Tobago": "Trinidad and Tobago", "Lituania": "Lithuania", "Uzbekistán": "Uzbekistan",
    "Martinica": "Martinique", "Jordania": "Jordan", "Níger": "Niger", "Libia": "Libya",
    "Kirguistán": "Kyrgyzstan", "Croacia": "Croatia", "Ruanda": "Rwanda", "Benín": "Benin",
    "Moldavia": "Moldova", "Azerbaiyán": "Azerbaijan", "Siria": "Syria", "Papúa Nueva Guinea": "Papua New Guinea",
    "Turkmenistán": "Turkmenistan", "Bosnia y Herzegovina": "Bosnia and Herzegovina", "Gabón": "Gabon",
    "Sierra Leona": "Sierra Leone", "Guadalupe": "Guadeloupe", "Túnez": "Tunisia", "Yibuti": "Djibouti",
    "Taiwán": "Taiwan", "Emiratos Árabes Unidos": "United Arab Emirates", "Grecia": "Greece",
    "Etiopía": "Ethiopia", "República Centroafricana": "Central African Republic", "Eslovaquia": "Slovakia",
    "Líbano": "Lebanon", "Lesoto": "Lesotho", "República del Congo": "Republic of the Congo",
    "Botsuana": "Botswana", "Belice": "Belize", "Chipre": "Cyprus", "Luxemburgo": "Luxembourg",
    "Guayana Francesa": "French Guiana", "Surinam": "Suriname", "Omán": "Oman", "Tayikistán": "Tajikistan",
    "Eslovenia": "Slovenia", "Macedonia": "North Macedonia", "Sudán del Sur": "South Sudan", "Bután": "Bhutan",
    "Suazilandia": "Eswatini", "República de Gambia": "Gambia", "Baréin": "Bahrain",
    "Sáhara Occidental": "Western Sahara", "Guinea Ecuatorial": "Equatorial Guinea",
}


def check(label, got, expected):
    ok = got == expected if not isinstance(expected, float) else abs(got - expected) < 5e-4
    print(f"  [{'OK' if ok else 'LỆCH'}] {label}: {got} (Sheet: {expected})")
    return ok


# ------------------------------------------------------------------ 1. Đọc và chuẩn hóa
def load(raw_path):
    opt = json.loads((ROOT / "data" / "options.json").read_text(encoding="utf-8"))
    d = pd.read_csv(raw_path, encoding="latin-1")
    for c in ["Order Region", "Order Country", "Market", "Shipping Mode", "Category Name", "Department Name",
              "Customer Segment", "Type"]:
        d[c] = d[c].str.rstrip()          # file gốc có dấu cách thừa ở cuối, vd "West of USA "
    d["order_country"] = d["Order Country"].replace(ES_EN)
    bad = d[[c not in opt["region_country"].get(r, []) for r, c in zip(d["Order Region"], d["order_country"])]]
    if len(bad):
        raise SystemExit("Chưa dịch được tên nước: " + ", ".join(sorted(bad["Order Country"].unique())[:20]))
    ts = pd.to_datetime(d["order date (DateOrders)"], format="%m/%d/%Y %H:%M")
    out = pd.DataFrame({
        "order_id": d["Order Id"], "order_item_id": d["Order Item Id"], "order_ts": ts,
        "shipping_mode": d["Shipping Mode"], "market": d["Market"], "order_region": d["Order Region"],
        "order_country": d["order_country"], "category_name": d["Category Name"],
        "department_name": d["Department Name"], "customer_segment": d["Customer Segment"],
        "payment_type": d["Type"], "scheduled_shipping_days": d["Days for shipment (scheduled)"],
        "quantity": d["Order Item Quantity"], "unit_price": d["Order Item Product Price"], "sales": d["Sales"],
        "discount_rate": d["Order Item Discount Rate"], "order_item_total": d["Order Item Total"],
        "order_month": ts.dt.month, "order_day_of_week": (ts.dt.dayofweek + 1) % 7 + 1,
        "actual_days": d["Days for shipping (real)"], "is_late": d["Late_delivery_risk"],
        # Giống is_canceled của tầng Gold: gồm cả đơn CANCELED và SUSPECTED_FRAUD
        "is_canceled": d["Delivery Status"].eq("Shipping canceled"), "product_name": d["Product Name"],
        # Thông tin cá nhân chỉ dùng để minh họa bước che dữ liệu ở 5.4 (DataCo là bộ dữ liệu công khai trên Kaggle)
        "customer_name": (d["Customer Fname"].fillna("") + " " + d["Customer Lname"].fillna("")).str.strip(),
        "customer_address": d["Customer Street"].fillna("") + ", " + d["Customer City"].fillna(""),
    })
    out["order_is_weekend"] = out["order_day_of_week"].isin([1, 7]).astype(int)
    out["order_hour"] = ts.dt.hour
    out["route_name"] = out["order_region"] + " - " + out["shipping_mode"]
    return out


def verify(d, pipe):
    """Tái tạo các số đã chốt trong tab Số liệu để chắc dữ liệu dựng lại khớp tầng Gold."""
    print("Kiểm chứng với tab Số liệu:")
    o = d.groupby("order_id").agg(late=("is_late", "max"), canceled=("is_canceled", "any"))
    nc = o[~o["canceled"]]
    check("Số đơn", len(o), 65752)
    check("Đơn không hủy", len(nc), 62897)
    check("Đơn trễ", int(nc["late"].sum()), 36048)
    lines = d[~d["is_canceled"]]
    check("Dòng hàng không hủy", len(lines), 172765)
    test = lines[lines["order_ts"] >= SPLIT]
    check("Số đơn tập kiểm tra", test["order_id"].nunique(), 12580)
    auc = roc_auc_score(test["is_late"], pipe.predict_proba(test[FEATURES])[:, 1])
    check("ROC-AUC XGBoost trên tập kiểm tra", round(auc, 4), 0.7838)
    return auc


# ------------------------------------------------------------------ 2. Mô hình ngày hẹn giao (5.1)
def quantile_model(alpha):
    pre = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                                    encoded_missing_value=-1), CAT_COLS)],
                            remainder="passthrough", verbose_feature_names_out=False)
    reg = HistGradientBoostingRegressor(loss="quantile", quantile=alpha, max_iter=300, learning_rate=0.05,
                                        max_leaf_nodes=31, categorical_features=list(range(len(CAT_COLS))),
                                        random_state=42)
    return Pipeline([("pre", pre), ("reg", reg)])


def sort_quantiles(p):
    """Bảo đảm P50 <= P80 <= P90 cho từng đơn."""
    q = np.sort(np.column_stack([p["p50"], p["p80"], p["p90"]]), axis=1)
    return {"p50": q[:, 0], "p80": q[:, 1], "p90": q[:, 2]}


def promise_models(train, test):
    models, pred = {}, {}
    for k, a in QUANTILES.items():
        models[k] = quantile_model(a).fit(train[FEATURES], train["actual_days"])
        pred[k] = models[k].predict(test[FEATURES])
    pred = sort_quantiles(pred)

    # Mốc đối chiếu: phân vị lịch sử theo tuyến x phương thức trên tập huấn luyện
    table = (train.groupby(["route_name", "order_region", "shipping_mode"])["actual_days"]
             .quantile(list(QUANTILES.values())).unstack().reset_index())
    table.columns = ["route_name", "order_region", "shipping_mode", "p50", "p80", "p90"]
    table = table.merge(train.groupby("route_name").size().rename("n_lines").reset_index(), on="route_name")
    mode_q = train.groupby("shipping_mode")["actual_days"].quantile(list(QUANTILES.values())).unstack()
    base = test[["route_name", "shipping_mode"]].merge(table, on=["route_name", "shipping_mode"], how="left")
    for k, a in QUANTILES.items():
        base[k] = base[k].fillna(test["shipping_mode"].map(mode_q[a]).reset_index(drop=True))

    y = test["actual_days"].to_numpy()
    res = {"pinball": {}, "coverage_p80": {}}
    for name, p in (("model", pred), ("table", {k: base[k].to_numpy() for k in QUANTILES})):
        res["pinball"][name] = {k: round(float(mean_pinball_loss(y, p[k], alpha=a)), 4) for k, a in QUANTILES.items()}
        cov = pd.Series(y <= np.ceil(p["p80"] - 1e-9)).groupby(test["shipping_mode"].to_numpy()).mean()
        res["coverage_p80"][name] = {m: round(float(v), 4) for m, v in cov.items()}
        res["coverage_p80"][name]["All"] = round(float(np.mean(y <= np.ceil(p["p80"] - 1e-9))), 4)
    better = res["pinball"]["model"]["p80"] < res["pinball"]["table"]["p80"]
    res["chosen"] = "model" if better else "table"
    print(f"Mô hình ngày hẹn: pinball P80 mô hình {res['pinball']['model']['p80']} "
          f"so với bảng {res['pinball']['table']['p80']} -> dùng {res['chosen']}")
    return models, table, res


# ------------------------------------------------------------------ 3. Hàng đợi và đánh giá (5.2, 5.3, 6.1, 6.2)
def build_queue(test, pipe, models, table, chosen):
    t = test.copy()
    t["p_late"] = pipe.predict_proba(t[FEATURES])[:, 1]
    if chosen == "model":
        q = sort_quantiles({k: models[k].predict(t[FEATURES]) for k in QUANTILES})
    else:
        q = t[["route_name"]].merge(table, on="route_name", how="left")
    for k in QUANTILES:
        t[k] = np.round(np.asarray(q[k], dtype=float), 2)
    t["exp_rev_at_risk"] = t["p_late"] * t["order_item_total"]
    sh = shap_by_feature(pipe, t)

    # Gộp dòng hàng thành đơn: lý do lấy từ dòng hàng có xác suất trễ cao nhất
    lead_idx = t.groupby("order_id")["p_late"].idxmax()
    lead = t.loc[lead_idx].set_index("order_id")
    g = t.groupby("order_id").agg(
        order_value=("order_item_total", "sum"), exp_rev_at_risk=("exp_rev_at_risk", "sum"),
        n_items=("order_item_id", "size"), actual_days=("actual_days", "max"), is_late=("is_late", "max"),
        products=("product_name", lambda s: "; ".join(dict.fromkeys(s))))
    q = lead[["order_ts", "customer_name", "customer_address", "route_name", "shipping_mode", "market",
              "order_region", "order_country", "category_name", "customer_segment", "payment_type",
              "scheduled_shipping_days", "order_hour", "p_late", "p50", "p80", "p90"]].join(g)
    q["risk_tier"] = risk_tier(q["p_late"])
    q["promise_days"] = np.ceil(q["p80"] - 1e-9).astype(int)
    q.loc[q["shipping_mode"] == "First Class", "promise_days"] = np.maximum(
        q.loc[q["shipping_mode"] == "First Class", "promise_days"], 2)                # luật R1
    q["reasons_json"] = [json.dumps(top_reasons(t.loc[i], sh.loc[i]), ensure_ascii=False) for i in lead_idx.loc[q.index]]
    q = q.reset_index()
    q["order_ts"] = q["order_ts"].dt.strftime("%Y-%m-%d %H:%M")
    q["model_version"] = MODEL_VERSION
    return q, sh


def backtest(q):
    """6.1: so 3 cách hẹn giao trên đơn của tập kiểm tra (tỷ lệ trễ so với ngày hẹn và số ngày hẹn thêm)."""
    ways = {
        "Current": q["scheduled_shipping_days"],
        "First Class 2 days": np.where(q["shipping_mode"] == "First Class", 2, q["scheduled_shipping_days"]),
        "Promise P50": np.ceil(q["p50"] - 1e-9),
        "Promise P80": q["promise_days"],
        "Promise P90": np.ceil(q["p90"] - 1e-9),
    }
    rows = []
    for name, days in ways.items():
        days = pd.Series(np.asarray(days), index=q.index)
        late = q["actual_days"] > days
        for mode, idx in list(q.groupby("shipping_mode").groups.items()) + [("All", q.index)]:
            rows.append({"way": name, "mode": mode, "n_orders": int(len(idx)),
                         "late_vs_promise": round(float(late[idx].mean()), 4),
                         "avg_promise_days": round(float(days[idx].mean()), 2),
                         "avg_extra_days": round(float((days[idx] - q.loc[idx, "scheduled_shipping_days"]).mean()), 2)})
    return rows


def precision_at_k(q, ks=(10, 20, 50)):
    """6.2: mỗi ngày nhân viên xử lý K đơn đầu hàng đợi thì bao nhiêu % thật sự trễ."""
    d = q.assign(day=pd.to_datetime(q["order_ts"]).dt.date)
    out = {}
    for scope, sub in (("All modes", d), ("Excluding First Class (handled by rule)", d[d["shipping_mode"] != "First Class"])):
        out[scope] = {}
        for sort_col, sort_name in (("exp_rev_at_risk", "by expected revenue at risk"), ("p_late", "by late probability")):
            for k in ks:
                top = sub.sort_values(sort_col, ascending=False).groupby("day").head(k)
                out[scope][f"Top {k} per day, {sort_name}"] = round(float(top["is_late"].mean()), 4)
        out[scope]["Baseline late rate (random pick)"] = round(float(sub["is_late"].mean()), 4)
        out[scope]["Days"] = int(sub["day"].nunique())
    return out


def main(raw_path):
    pipe = joblib.load(ROOT / "models" / "best_pipe.joblib")
    d = load(raw_path)
    auc = verify(d, pipe)
    lines = d[~d["is_canceled"]]
    train, test = lines[lines["order_ts"] < SPLIT].reset_index(drop=True), lines[lines["order_ts"] >= SPLIT].reset_index(drop=True)

    models, table, promise_eval = promise_models(train, test)
    (ROOT / "models").mkdir(exist_ok=True)
    for k, m in models.items():
        joblib.dump(m, ROOT / "models" / f"promise_{k}.joblib", compress=3)
    table.round(2).to_csv(ROOT / "data" / "promise_table.csv", index=False)

    q, sh = build_queue(test, pipe, models, table, promise_eval["chosen"])
    keep = ["order_id", "order_ts", "customer_name", "customer_address", "route_name", "shipping_mode", "market",
            "order_region", "order_country", "category_name", "customer_segment", "payment_type", "products",
            "n_items", "order_value", "scheduled_shipping_days", "order_hour", "p_late", "risk_tier",
            "exp_rev_at_risk", "p50", "p80", "p90", "promise_days", "reasons_json", "model_version",
            "actual_days", "is_late"]
    q[keep].round({"order_value": 2, "exp_rev_at_risk": 2, "p_late": 4}).to_csv(
        ROOT / "data" / "order_queue.csv", index=False, encoding="utf-8")

    sample = sh.sample(min(3000, len(sh)), random_state=42)
    shap_global = sample.abs().mean().sort_values(ascending=False).round(4)
    ev = {
        "auc_test": round(float(auc), 4), "split_date": str(SPLIT.date()),
        "n_test_orders": int(q["order_id"].nunique()), "promise": promise_eval,
        "backtest": backtest(q), "precision_at_k": precision_at_k(q),
        "shap_global": shap_global.to_dict(), "model_version": MODEL_VERSION,
    }
    (ROOT / "data" / "evaluation.json").write_text(json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Backtest (All):")
    for r in ev["backtest"]:
        if r["mode"] == "All":
            print(f"  {r['way']}: late vs promise {r['late_vs_promise']:.1%}, extra promised days {r['avg_extra_days']}")
    print("SHAP toàn cục:", shap_global.head(5).to_dict())
    print("Đã ghi data/ và models/.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
