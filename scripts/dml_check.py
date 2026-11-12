"""Kiểm tra bằng Double Machine Learning: một phương thức giao có nhanh hơn Standard Class không?

Chạy:  python dml_check.py <file_gold.(pkl|csv|xlsx)> [--fast]
Bảng Gold cần có các cột: order_id, order_status, shipping_mode, actual_shipping_days, order_region, market,
customer_segment, payment_type, category_name, order_day_of_week, order_hour, order_month, order_year,
order_total_value, order_item_count.
Cần: pip install pandas numpy scikit-learn  (không cần thư viện DoubleML; ước lượng tự tính theo công thức mục 3 của tài liệu DML)

Kết quả in ra: mỗi cặp (Second Class, First Class, Same Day so với Standard Class) gồm so trung bình, hồi quy,
DML, kiểm tra positivity, phép thử giả (xáo nhãn), đổi mô hình, và (riêng Second Class) theo thị trường và theo năm.
Mốc ngày trong phép thử giả là bản đơn giản hóa: nhãn xáo độc lập với biến kiểm soát nên m(Z) là hằng số, dùng lại phần dư của Y.
"""
import sys, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
FAST = "--fast" in sys.argv
TREES = 30 if FAST else 100
CATS = ["region", "market", "seg", "pay", "cat"]
NUMS = ["dow", "hr", "mon", "yr", "val", "n"]


def load(path):
    if path.endswith(".pkl"): return pd.read_pickle(path)
    if path.endswith(".csv"): return pd.read_csv(path)
    return pd.read_excel(path)


def prep(base, treat):
    g = base[base.shipping_mode.isin([treat, "Standard Class"])]
    o = g.groupby("order_id").agg(mode=("shipping_mode", "first"), days=("actual_shipping_days", "max"),
        region=("order_region", "first"), market=("market", "first"), seg=("customer_segment", "first"),
        pay=("payment_type", "first"), cat=("category_name", "first"), dow=("order_day_of_week", "first"),
        hr=("order_hour", "first"), mon=("order_month", "first"), yr=("order_year", "first"),
        val=("order_total_value", "max"), n=("order_item_count", "max")).reset_index()
    o["D"] = (o["mode"] == treat).astype(int)
    return o


def X_of(o): return pd.get_dummies(o[CATS + NUMS], columns=CATS).values.astype(float)


def learners(kind):
    if kind == "RF":
        return (RandomForestRegressor(TREES, min_samples_leaf=20, n_jobs=-1, random_state=0),
                RandomForestClassifier(TREES, min_samples_leaf=20, n_jobs=-1, random_state=0))
    if kind == "HGB":
        return (HistGradientBoostingRegressor(max_iter=150, random_state=0), HistGradientBoostingClassifier(max_iter=150, random_state=0))
    return (LassoCV(cv=3, n_jobs=-1, max_iter=2000), LogisticRegression(penalty="l1", solver="saga", C=0.5, max_iter=300))


def dml(o, kind="RF", seed=0, X=None):
    """theta = sum(D~ * Y~) / sum(D~^2); chia chéo 5 phần; sai số chuẩn theo công thức bền (robust)."""
    X = X_of(o) if X is None else X
    Y, D = o.days.values.astype(float), o.D.values
    if kind == "Lasso": X = StandardScaler().fit_transform(X)
    rY, rD, p = np.zeros(len(o)), np.zeros(len(o)), np.zeros(len(o))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        l, m = learners(kind)
        rY[te] = Y[te] - l.fit(X[tr], Y[tr]).predict(X[te])
        p[te] = m.fit(X[tr], D[tr]).predict_proba(X[te])[:, 1]
        rD[te] = D[te] - p[te]
    th = (rD * rY).sum() / (rD ** 2).sum()
    e = rY - th * rD
    se = np.sqrt((rD ** 2 * e ** 2).sum()) / (rD ** 2).sum()
    return th, se, rY, p


fmt = lambda t, s: f"{t:+.3f} ({t - 1.96 * s:+.3f} .. {t + 1.96 * s:+.3f})"

if __name__ == "__main__":
    df = load(sys.argv[1])
    base = df[~df.order_status.isin(["CANCELED", "SUSPECTED_FRAUD"])]
    for treat in ["Second Class", "First Class", "Same Day"]:
        o = prep(base, treat); X = X_of(o); D = o.D.values; Y = o.days.values.astype(float)
        th, se, rY, p = dml(o, "RF", 0, X)
        print(f"\n=== {treat} so với Standard Class: n={len(o):,} đơn, {D.mean()*100:.1f}% là {treat}")
        d = Y[D == 1].mean() - Y[D == 0].mean()
        sd = np.sqrt(Y[D == 1].var(ddof=1) / D.sum() + Y[D == 0].var(ddof=1) / (len(D) - D.sum()))
        b = np.linalg.lstsq(np.column_stack([np.ones(len(o)), D, X]), Y, rcond=None)[0]
        print(f"Số ngày TB {Y[D==1].mean():.2f} so với {Y[D==0].mean():.2f}")
        print(f"So trung bình: {fmt(d, sd)} | hồi quy có kiểm soát: {b[1]:+.3f} | DML (RF): {fmt(th, se)}")
        cnt = o.groupby(["region", "market"]).D.agg(["sum", "count"])
        print(f"Positivity: xác suất chọn từ {p.min():.3f} đến {p.max():.3f}; ô vùng x thị trường thiếu một nhóm: "
              f"{int(((cnt['sum'] == 0) | (cnt['sum'] == cnt['count'])).sum())}/{len(cnt)}")
        rng = np.random.default_rng(1); pl = []
        for _ in range(100 if FAST else 300):
            Ds = rng.permutation(D); rD = Ds - Ds.mean(); t = (rD * rY).sum() / (rD ** 2).sum()
            e = rY - t * rD; s = np.sqrt((rD ** 2 * e ** 2).sum()) / (rD ** 2).sum(); pl.append((t, abs(t) < 1.96 * s))
        pl = np.array(pl)
        print(f"Phép thử giả: theta TB {pl[:,0].mean():+.4f}, SD {pl[:,0].std():.4f}, khoảng tin cậy chứa 0 ở {pl[:,1].mean()*100:.0f}% số lần")
        for k in (["HGB"] if FAST else ["HGB", "Lasso"]): print(f"Đổi mô hình {k}: {fmt(*dml(o, k, 0, X)[:2])}")
        if treat == "Second Class" and not FAST:
            for col, name in (("market", "thị trường"), ("yr", "năm")):
                print(f"Theo {name}:")
                for key, g in o.groupby(col): print(f"  {key}: n={len(g):,}  {fmt(*dml(g.reset_index(drop=True), 'RF', 0)[:2])}")
