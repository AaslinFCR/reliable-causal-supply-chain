"""Next-7-day demand forecasting with past-only features and yearly refits.

Target: 7-day demand divided by the trailing 28-day level (ratio target).
Primary model: histogram gradient-boosted trees (LightGBM if installed, otherwise
scikit-learn's HistGradientBoostingRegressor, the same leaf-wise histogram GBDT family).
Comparators: seasonal naive, 28-day moving average, Random Forest.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from . import config as C

try:  # optional
    import lightgbm as lgb  # noqa
    HAVE_LGB = True
except Exception:  # pragma: no cover
    HAVE_LGB = False

FEATURES = ["sum7", "sum14", "ma28x7", "sd28", "lag1", "lag7", "ly_week", "r7_28", "r7_ly",
            "fest_next7", "fest_max7", "days_to_fest", "promo_next7", "promo_last7",
            "dow", "doy_s", "doy_c", "month", "region", "product"]


def build_frame(world):
    """One row per (series, origin day t). Features use data up to t-1; target = demand[t:t+7]."""
    D = world["demand"].astype(float); S, T = D.shape
    dts = world["dates"]
    rows = []
    for s, (r, p) in enumerate(world["series"]):
        d = pd.Series(D[s])
        prev = d.shift(1)
        f = pd.DataFrame({
            "s": s, "t": np.arange(T),
            "sum7": prev.rolling(7).sum(), "sum14": prev.rolling(14).sum(),
            "ma28x7": prev.rolling(28).mean() * 7, "sd28": prev.rolling(28).std(),
            "lag1": prev, "lag7": d.shift(7),
            "ly_week": d.shift(364).rolling(7).sum().shift(-6),   # same week last year (t-364..t-358)
        })
        f["r7_28"] = f.sum7 / f.ma28x7
        f["r7_ly"] = f.sum7 / d.shift(371).rolling(7).sum()
        fk = pd.Series(world["fest_known"][s]); pr = pd.Series(world["promo"][s])
        f["fest_next7"] = fk[::-1].rolling(7, min_periods=1).mean()[::-1].values
        f["fest_max7"] = fk[::-1].rolling(7, min_periods=1).max()[::-1].values
        f["days_to_fest"] = world["fest_days"][s]
        f["promo_next7"] = pr[::-1].rolling(7, min_periods=1).sum()[::-1].values  # promos are planned
        f["promo_last7"] = pr.shift(1).rolling(7).sum()
        f["dow"] = dts.dayofweek; f["month"] = dts.month
        f["doy_s"] = np.sin(2 * np.pi * dts.dayofyear / 365.25)
        f["doy_c"] = np.cos(2 * np.pi * dts.dayofyear / 365.25)
        f["region"] = C.REGIONS.index(r); f["product"] = C.PRODUCTS.index(p)
        f["y"] = d[::-1].rolling(7).sum()[::-1].values  # demand t..t+6
        rows.append(f)
    df = pd.concat(rows, ignore_index=True)
    df["date"] = dts[df.t.values]
    return df


class GBDT:
    name = "GBDT"

    def __init__(self, seed=C.SEED):
        if HAVE_LGB:
            self.m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31,
                                       min_child_samples=40, reg_lambda=5, subsample=0.8,
                                       subsample_freq=1, colsample_bytree=0.8,
                                       random_state=seed, verbose=-1)
            self.name = "LightGBM"
        else:
            self.m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.04, max_leaf_nodes=31,
                                                   min_samples_leaf=40, l2_regularization=5.0,
                                                   categorical_features=[FEATURES.index("region"),
                                                                         FEATURES.index("product")],
                                                   random_state=seed)

    def fit(self, X, y):
        # target = 7-day demand relative to the past 28-day level (scale-free, robust to trend)
        self.m.fit(X[FEATURES], y / X.ma28x7.values); return self

    def predict(self, X):
        return (self.m.predict(X[FEATURES]) * X.ma28x7.values).clip(0)


class RF:
    name = "RandomForest"

    def __init__(self, seed=C.SEED):
        self.m = RandomForestRegressor(n_estimators=150, max_depth=18, min_samples_leaf=3,
                                       max_features=0.6, n_jobs=-1, random_state=seed)

    def _X(self, X):
        Z = X[FEATURES].copy()
        Z["ly_week"] = Z.ly_week.fillna(Z.ma28x7); Z["r7_ly"] = Z.r7_ly.fillna(1.0)
        return Z.fillna(0)

    def fit(self, X, y):
        self.m.fit(self._X(X), y / X.ma28x7.values); return self

    def predict(self, X):
        return (self.m.predict(self._X(X)) * X.ma28x7.values).clip(0)


def split_masks(df, year):
    """Train origins end so targets finish by 30 June of year-1; calibration = Jul-Dec of year-1."""
    tr_end = pd.Timestamp(f"{year-1}-06-30") - pd.Timedelta(days=6)
    ca_start = pd.Timestamp(f"{year-1}-07-01"); ca_end = pd.Timestamp(f"{year-1}-12-31") - pd.Timedelta(days=6)
    ok = df.y.notna() & df.ma28x7.notna()
    tr = ok & (df.date <= tr_end)
    ca = ok & (df.date >= ca_start) & (df.date <= ca_end)
    te = df.ma28x7.notna() & (df.date.dt.year == year)
    return tr, ca, te


def fit_year(df, year, models=("GBDT",)):
    tr, ca, te = split_masks(df, year)
    out = {}
    for name in models:
        m = (GBDT() if name == "GBDT" else RF()).fit(df[tr], df.y[tr].values)
        out[name] = dict(model=m, cal=m.predict(df[ca]), test=m.predict(df[te]))
    out["masks"] = (tr, ca, te)
    out["naive"] = dict(cal=df.sum7[ca].values, test=df.sum7[te].values)
    out["ma28"] = dict(cal=df.ma28x7[ca].values, test=df.ma28x7[te].values)
    return out


def point_metrics(y, yhat):
    m = ~np.isnan(y)
    y, yhat = y[m], yhat[m]
    e = y - yhat
    wape = np.abs(e).sum() / y.sum()
    return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), WAPE=100 * wape,
                Accuracy=100 * (1 - wape), Bias=100 * e.sum() / y.sum(),
                R2=1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())
