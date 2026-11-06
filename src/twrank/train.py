"""Rolling point-in-time LightGBM rankers.

Every ``REFIT_WEEKS`` weekly trade days from ``FIRST_SCORE`` on, one model per
horizon is refit on *all* weekly rows whose label had fully closed before the
cutoff (``end_h < cutoff``; an expanding window with an exact embargo).  That
model then scores every trading day until the next cutoff.  Each score is
therefore strictly out-of-sample.

Models are trained on weekly rows only (as in the earlier study this repo
replicates) but score every trading day, because the monthly schedules in
``genesis/FIAT-rebalance.md`` trade on arbitrary days of the month.

Output: ``data/derived/scores.parquet`` with columns date, stock, s21, s63
(raw model outputs) and ``results/models/ic.csv`` (out-of-sample rank IC per
refit segment and horizon), ``results/models/importance.csv``.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config
from .config import RESULTS_DIR
from .features import FEATURES, HORIZONS, load_table

FIRST_SCORE = "2004-01-01"
REFIT_WEEKS = 26
TREES = 400
LGB_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 63,
    "seed": 42,
    "deterministic": True,
    "verbosity": -1,
}
MODEL_DIR = RESULTS_DIR / "models"


def _fit(X, y, device: str):
    import lightgbm as lgb

    params = dict(LGB_PARAMS, device_type=device)
    if device == "cpu":
        params["num_threads"] = 16
    ds = lgb.Dataset(X, y, feature_name=FEATURES, free_raw_data=True)
    return lgb.train(params, ds, num_boost_round=TREES)


def cutoffs(table: pd.DataFrame) -> list[pd.Timestamp]:
    weeks = np.sort(table.loc[table["weekly"], "date"].unique())
    first = np.searchsorted(weeks, np.datetime64(FIRST_SCORE))
    return [pd.Timestamp(weeks[i]) for i in range(first, len(weeks), REFIT_WEEKS)]


def _ic(sub: pd.DataFrame, col: str, label: str) -> float:
    v = sub.dropna(subset=[label])
    if v.empty:
        return float("nan")
    v = v.assign(sr=v.groupby("date")[col].rank(pct=True))
    per_day = v.groupby("date").apply(lambda x: x["sr"].corr(x[label]), include_groups=False)
    return float(per_day.mean())


def train_all(device: str = "cpu", log=print) -> pd.DataFrame:
    t0 = time.time()
    table = load_table()
    X = table[FEATURES].to_numpy(dtype="float32")
    dates = table["date"].to_numpy()
    weekly = table["weekly"].to_numpy()
    cuts = cutoffs(table)
    bounds = [*cuts, pd.Timestamp("2100-01-01")]
    log(f"table {len(table):,} rows; {len(cuts)} refits {cuts[0].date()}..{cuts[-1].date()}")

    pred = {h: np.full(len(table), np.nan, dtype="float32") for h in HORIZONS}
    ic_rows = []
    imp = {h: np.zeros(len(FEATURES)) for h in HORIZONS}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for k, c in enumerate(cuts):
        seg = (dates >= np.datetime64(c)) & (dates < np.datetime64(bounds[k + 1]))
        row = {"cutoff": c.date(), "segment_end": (bounds[k + 1] - pd.Timedelta(days=1)).date()
               if k + 1 < len(cuts) else pd.Timestamp(dates.max()).date()}
        for h in HORIZONS:
            y = table[f"y{h}"].to_numpy()
            end = table[f"end{h}"].to_numpy()
            tr = weekly & (end < np.datetime64(c)) & np.isfinite(y)
            t1 = time.time()
            try:
                model = _fit(X[tr], y[tr], device)
            except Exception as exc:  # GPU unavailable: fall back once, permanently
                log(f"device {device} failed ({str(exc)[:80]}); using cpu")
                device = "cpu"
                model = _fit(X[tr], y[tr], device)
            imp[h] += model.feature_importance("gain")
            pred[h][seg] = model.predict(X[seg])
            sub = pd.DataFrame({"date": dates[seg & weekly], "s": pred[h][seg & weekly],
                                "y": y[seg & weekly]})
            row[f"n_train{h}"] = int(tr.sum())
            row[f"ic{h}"] = round(_ic(sub, "s", "y"), 4)
            row[f"sec{h}"] = round(time.time() - t1, 1)
        ic_rows.append(row)
        log(f"refit {k + 1}/{len(cuts)} {c.date()} n={row['n_train21']:,}/{row['n_train63']:,} "
            f"IC21={row['ic21']:.3f} IC63={row['ic63']:.3f} [{device}] {row['sec21']}+{row['sec63']}s")

    out = table[["date", "stock"]].copy()
    for h in HORIZONS:
        out[f"s{h}"] = pred[h]
    out = out.dropna(subset=[f"s{h}" for h in HORIZONS])
    out.to_parquet(scores_path(), index=False)
    ic = pd.DataFrame(ic_rows)
    ic.to_csv(MODEL_DIR / "ic.csv", index=False)
    fi = pd.DataFrame({f"gain{h}": imp[h] / imp[h].sum() for h in HORIZONS}, index=FEATURES)
    fi.sort_values("gain21", ascending=False).round(4).to_csv(MODEL_DIR / "importance.csv")
    log(f"saved {scores_path()} ({len(out):,} rows) in {time.time() - t0:.0f}s; "
        f"mean IC21 {ic['ic21'].mean():.4f} IC63 {ic['ic63'].mean():.4f}")
    return ic


def scores_path():
    return config.derived_dir() / "scores.parquet"


def load_scores() -> pd.DataFrame:
    return pd.read_parquet(scores_path())
