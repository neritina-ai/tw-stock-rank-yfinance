"""Price/volume features, labels and the tradable universe.

Alignment convention (used everywhere downstream): a row dated ``d`` is a
*trade day*.  Its features and eligibility use data through the close of the
previous trading day ``d-1`` only, and a trade decided from them executes at
``d``'s open.  The label for row ``d`` is the open-to-open return a position
entered at ``d``'s open can capture: ``open(d+h) / open(d) - 1`` for
``h`` = 21 or 63 trading days, ranked among the stocks eligible at ``d``.
``y_end`` is the date of ``open(d+h)``; a model trained at cutoff ``c`` only
uses rows with ``y_end < c`` so every label it sees had fully closed.

Stock features are converted to cross-sectional percentiles among eligible
stocks each day, so the model only ever compares candidates with each other.

Output: ``data/derived/table.parquet``, one row per (trading day, eligible
stock), with 26 ranked features, both labels and their end dates, and a flag
for the weekly trade days the models are trained on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DERIVED_DIR, INDEX_TICKER
from .data import load_index, load_panel

HORIZONS = (21, 63)
LOOKBACK = 252  # longest feature window, in trading days

# Tradable universe (see README): 20-day median turnover above NT$10m, and at
# least 100 bars in the last 130 trading days so that features exist.
MIN_TURNOVER = 10e6
TURNOVER_WIN = 20
MIN_BARS, BARS_WIN = 100, 130

FEATURES = [
    "ret5", "ret21", "ret63", "ret126", "mom_12_1",
    "nh52", "nl52", "magap20", "magap60", "magap120", "tstat63",
    "vol21", "vol63", "dvol63", "skew63", "max1_21", "min1_21", "rng21",
    "beta63", "amihud21", "ldv63", "tratio", "volcv21",
    "clv21", "ogap21", "ups63",
]

TABLE = DERIVED_DIR / "table.parquet"


def calendar() -> pd.DatetimeIndex:
    """Trading days: dates with a TAIEX bar and a reasonably full stock cross-section."""
    close = load_panel("close")
    n = close.notna().sum(axis=1)
    idx = load_index().index
    days = close.index[(n >= 0.5 * n.rolling(20, min_periods=1).median()) & close.index.isin(idx)]
    return pd.DatetimeIndex(days)


def week_starts(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First trading day of every ISO week; a week with no session has no entry."""
    iso = days.isocalendar()
    s = pd.Series(days, index=days)
    return pd.DatetimeIndex(s.groupby([iso.year.to_numpy(), iso.week.to_numpy()]).first())


def _panels(days: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    return {k: load_panel(k).reindex(days).astype("float64") for k in
            ("open", "high", "low", "close", "volume", "turnover")}


def eligible(days: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Boolean (trade day x stock): tradable at ``d``'s open using data through ``d-1``."""
    days = calendar() if days is None else days
    P = _panels(days)
    to = P["turnover"]
    med = to.rolling(TURNOVER_WIN, min_periods=TURNOVER_WIN // 2).median()
    bars = P["close"].notna().rolling(BARS_WIN, min_periods=1).sum()
    ok = (med > MIN_TURNOVER) & (bars >= MIN_BARS) & P["close"].notna()
    return ok.shift(1, fill_value=False).astype(bool)


def raw_features(P: dict[str, pd.DataFrame], mkt_close: pd.Series):
    """Yield ``(name, matrix)`` with information through each row's own close."""
    ac, ao, ah, al = P["close"], P["open"], P["high"], P["low"]
    to = P["turnover"].where(P["turnover"] > 0)
    vol = P["volume"]
    r = ac.pct_change(fill_method=None)

    def mom(n, skip=0):
        return ac.shift(skip) / ac.shift(n) - 1

    yield "ret5", mom(5)
    yield "ret21", mom(21)
    yield "ret63", mom(63)
    yield "ret126", mom(126)
    yield "mom_12_1", mom(252, 21)
    yield "nh52", ac / ac.rolling(252, min_periods=126).max()
    yield "nl52", ac / ac.rolling(252, min_periods=126).min()
    for n in (20, 60, 120):
        yield f"magap{n}", ac / ac.rolling(n, min_periods=n // 2).mean() - 1
    r63 = r.rolling(63, min_periods=40)
    yield "tstat63", r63.mean() / r63.std()
    yield "vol21", r.rolling(21, min_periods=15).std()
    yield "vol63", r63.std()
    yield "dvol63", r.where(r < 0).rolling(63, min_periods=15).std()
    yield "skew63", r63.skew()
    yield "max1_21", r.rolling(21, min_periods=15).max()
    yield "min1_21", r.rolling(21, min_periods=15).min()
    rng = ((ah - al) / ac).where(ah >= al)
    yield "rng21", rng.rolling(21, min_periods=15).mean()

    rm = mkt_close.pct_change(fill_method=None)
    rm_mat = pd.DataFrame(np.broadcast_to(rm.to_numpy()[:, None], ac.shape),
                          index=ac.index, columns=ac.columns)
    exy = (r * rm_mat).rolling(63, min_periods=40).mean()
    ex = r63.mean()
    ey = rm.rolling(63, min_periods=40).mean()
    varm = rm.rolling(63, min_periods=40).var(ddof=0)
    yield "beta63", (exy - ex.mul(ey, axis=0)).div(varm.replace(0, np.nan), axis=0)

    yield "amihud21", (r.abs() / to).rolling(21, min_periods=15).mean()
    yield "ldv63", np.log(to.rolling(63, min_periods=30).median())
    yield "tratio", (to.rolling(21, min_periods=10).median()
                     / to.rolling(126, min_periods=60).median())
    v21 = vol.rolling(21, min_periods=15)
    yield "volcv21", v21.std() / v21.mean()
    hl = (ah - al).where((ah - al) > 0)
    yield "clv21", ((ac - al) / hl).clip(0, 1).rolling(21, min_periods=15).mean()
    yield "ogap21", (ao / ac.shift(1) - 1).rolling(21, min_periods=15).mean()
    yield "ups63", (r > 0).astype(float).where(r.notna()).rolling(63, min_periods=40).mean()


def build_table(log=print) -> pd.DataFrame:
    days = calendar()
    P = _panels(days)
    E = eligible(days)
    mkt = load_index()["Close"].reindex(days).ffill()
    rix, cix = np.nonzero(E.to_numpy())
    stocks = E.columns.to_numpy()
    out: dict[str, np.ndarray] = {
        "date": days.to_numpy()[rix],
        "stock": stocks[cix],
    }
    weekly = days.isin(week_starts(days))
    out["weekly"] = weekly[rix]
    Ev = E.to_numpy()
    for name, m in raw_features(P, mkt):
        shifted = m.shift(1)  # row d sees data through d-1
        ranked = shifted.where(Ev).rank(axis=1, pct=True)
        out[name] = ranked.to_numpy(dtype="float32")[rix, cix]
    log(f"features: {len(FEATURES)} x {len(rix):,} eligible rows")

    ao = P["open"]
    pos = np.arange(len(days))
    for h in HORIZONS:
        y = ao.shift(-h) / ao - 1
        yr = y.where(Ev).rank(axis=1, pct=True)
        out[f"y{h}"] = yr.to_numpy(dtype="float32")[rix, cix]
        end = np.where(pos + h < len(days), days.to_numpy()[np.minimum(pos + h, len(days) - 1)],
                       np.datetime64("NaT"))
        out[f"end{h}"] = end[rix]
    table = pd.DataFrame(out)
    table.to_parquet(TABLE, index=False)
    log(f"saved {TABLE} ({len(table):,} rows, {days[0].date()}..{days[-1].date()})")
    return table


def load_table() -> pd.DataFrame:
    return pd.read_parquet(TABLE)


__all__ = ["FEATURES", "HORIZONS", "INDEX_TICKER", "build_table", "calendar", "eligible",
           "load_table", "week_starts"]
