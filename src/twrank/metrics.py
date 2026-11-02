"""Performance statistics on daily return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS_PER_YEAR as ANN


def nav(ret: pd.Series) -> pd.Series:
    return (1 + ret.fillna(0.0)).cumprod()


def drawdown(ret: pd.Series) -> pd.Series:
    v = nav(ret)
    return v / v.cummax() - 1


def summary(ret: pd.Series, turnover: pd.Series | None = None, cost: pd.Series | None = None,
            weights: pd.DataFrame | None = None) -> dict:
    r = ret.dropna()
    if len(r) < 2:
        return {}
    v = nav(r)
    years = len(r) / ANN
    sd = r.std()
    out = {
        "start": str(r.index[0].date()),
        "end": str(r.index[-1].date()),
        "years": round(years, 2),
        "total": float(v.iloc[-1] - 1),
        "cagr": float(v.iloc[-1] ** (1 / years) - 1),
        "vol": float(sd * np.sqrt(ANN)),
        "sharpe": float(r.mean() / sd * np.sqrt(ANN)) if sd > 0 else float("nan"),
        "mdd": float((v / v.cummax() - 1).min()),
    }
    out["calmar"] = out["cagr"] / -out["mdd"] if out["mdd"] < 0 else float("nan")
    if turnover is not None:
        out["turnover"] = float(turnover.loc[r.index].mean() * ANN)
    if cost is not None:
        out["cost"] = float(cost.loc[r.index].mean() * ANN)
    if weights is not None and len(weights.columns):
        w = weights.loc[r.index]
        n = (w > 1e-6).sum(axis=1)
        out["positions"] = float(n[n > 0].mean()) if (n > 0).any() else 0.0
        out["exposure"] = float(w.sum(axis=1).mean())
    return out


def yearly(ret: pd.Series) -> pd.Series:
    r = ret.dropna()
    return (1 + r).groupby(r.index.year).prod() - 1


def bench_returns(px: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Daily returns of a price series aligned to ``index`` (missing days = 0)."""
    p = px.reindex(px.index.union(index)).ffill().reindex(index)
    return p.pct_change(fill_method=None).fillna(0.0)
