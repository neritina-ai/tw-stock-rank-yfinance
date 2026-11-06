"""Scores -> trade schedule -> target weights.

Score.  Each model's raw output is ranked among the eligible stocks each day
(percentile, 1 = best); the ensemble score is the mean of the 21-day and the
63-day ranks.  The *hold score* used for the buffer is an exponential moving
average of the score with a span of ``5 x smooth_weeks`` trading days (the
earlier study smoothed its weekly score with a 4-week span; 20 trading days is
the same decay expressed per day, so the same definition works for any
schedule).

Schedule.  Weekly: the first trading day of every ISO week (a closed Monday
moves to the next session; a fully closed week merges into the next week).
Monthly on day ``D``: the first trading day on or after the ``D``-th of each
month.  Holidays always postpone, never advance: a typhoon day is often
announced the evening before, so a rule that trades early is not executable.

Book.  At a trade day the current holdings that still rank inside the top
``band_mult x top_n`` by hold score are kept; the book is then filled up to
``top_n`` names from the top of the score ranking, equal weight.  When the
TAIEX closed at or below its 120-day EMA on the previous session, the total
exposure is halved.  Weights are only re-issued when the names or the exposure
change, or at least every 56 calendar days to undo drift.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache

import numpy as np
import pandas as pd

from .data import load_index
from .features import calendar, week_starts

REFRESH_DAYS = 56


@dataclass(frozen=True)
class Spec:
    id: str
    label: str
    score: str = "ens"  # "ens" | "s21" | "s63"
    schedule: str = "weekly"  # "weekly" | "monthly"
    day: int = 1  # day of month for the monthly schedule
    top_n: int = 30
    band_mult: float = 4.0
    smooth_weeks: float = 4.0
    regime: str = "half"  # "half" | "none"
    regime_check: str = "trade"  # "trade" | "weekly" | "daily"
    note: str = ""
    tags: tuple = field(default_factory=tuple)

    def with_(self, **kw) -> Spec:
        return replace(self, **kw)

    def params(self) -> dict:
        d = asdict(self)
        for k in ("id", "label", "note", "tags"):
            d.pop(k)
        return d


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _raw_scores() -> dict[str, pd.DataFrame]:
    from .train import load_scores

    s = load_scores()
    out = {}
    for h in ("s21", "s63"):
        w = s.pivot(index="date", columns="stock", values=h)
        out[h] = w.rank(axis=1, pct=True).astype("float64")
    return out


@lru_cache(maxsize=8)
def score_panel(kind: str) -> pd.DataFrame:
    r = _raw_scores()
    if kind == "ens":
        return (r["s21"] + r["s63"]) / 2
    return r[kind]


@lru_cache(maxsize=32)
def hold_panel(kind: str, smooth_weeks: float) -> pd.DataFrame:
    s = score_panel(kind)
    if smooth_weeks <= 1:
        return s
    span = 5 * smooth_weeks
    return s.ewm(span=span, min_periods=1).mean().where(s.notna())


@lru_cache(maxsize=4)
def risk_on() -> pd.Series:
    """True when TAIEX closed above its 120-day EMA on the previous session."""
    days = calendar()
    px = load_index()["Close"].reindex(days).ffill()
    return (px > px.ewm(span=120).mean()).shift(1, fill_value=True).astype(bool)


# ---------------------------------------------------------------------------
# schedules
# ---------------------------------------------------------------------------


def monthly_dates(days: pd.DatetimeIndex, day: int) -> pd.DatetimeIndex:
    """First session on or after the ``day``-th of each month (postpone only)."""
    months = pd.period_range(days[0], days[-1], freq="M")
    targets = pd.DatetimeIndex([pd.Timestamp(m.year, m.month, day) for m in months])
    pos = days.searchsorted(targets)
    pos = pos[pos < len(days)]
    return pd.DatetimeIndex(sorted(set(days[pos])))


def trade_dates(spec: Spec, start, end) -> pd.DatetimeIndex:
    days = calendar()
    if spec.schedule == "weekly":
        d = week_starts(days)
    elif spec.schedule == "monthly":
        d = monthly_dates(days, spec.day)
    else:
        raise ValueError(spec.schedule)
    return d[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]


def _check_dates(spec: Spec, start, end) -> pd.DatetimeIndex:
    days = calendar()
    if spec.regime == "none" or spec.regime_check == "trade":
        return pd.DatetimeIndex([])
    d = week_starts(days) if spec.regime_check == "weekly" else days
    return d[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]


# ---------------------------------------------------------------------------
# book construction
# ---------------------------------------------------------------------------


def select(score_row: pd.Series, hold_row: pd.Series, held: list[str],
           spec: Spec) -> list[str] | None:
    """The book after one trade day: keep holdings inside the band, fill from the top.

    ``score_row`` ranks today's candidates, ``hold_row`` is the smoothed score the
    buffer uses.  Returns None when there is no score at all (no trade).
    """
    s = score_row.dropna().sort_values(ascending=False, kind="stable")
    if s.empty:
        return None
    ks = hold_row.dropna().sort_values(ascending=False, kind="stable")
    band_set = set(ks.index[: int(round(spec.top_n * spec.band_mult))])
    keep = [x for x in held if x in band_set]
    kept = set(keep)
    adds = [x for x in s.index if x not in kept][: spec.top_n - len(keep)]
    return keep + adds


@dataclass
class Book:
    """Target weights plus the per-trade-day record the viewer needs."""

    weights: dict  # date -> {stock: weight}; one entry per order day
    trades: list  # one dict per scheduled trade day
    held_after: list  # holdings after the last trade day


def build_book(spec: Spec, start, end, held: list[str] | None = None,
               record: bool = False) -> Book:
    score = score_panel(spec.score)
    hold = hold_panel(spec.score, spec.smooth_weeks)
    on = risk_on()
    tdays = trade_dates(spec, start, end)
    checks = _check_dates(spec, start, end)
    events = tdays.union(checks)

    held = list(held or [])
    last_scale = -1.0
    last_emit = None
    weights: dict = {}
    trades: list = []
    tset = set(tdays)
    for d in events:
        if d not in score.index:
            continue
        scale = 0.5 if (spec.regime == "half" and not on.loc[d]) else 1.0
        if d in tset:
            new_held = select(score.loc[d], hold.loc[d], held, spec)
            if new_held is None:
                continue
            changed = set(new_held) != set(held)
            stale = last_emit is None or (d - last_emit).days >= REFRESH_DAYS
            if changed or abs(scale - last_scale) > 1e-9 or stale:
                weights[d] = {x: scale / len(new_held) for x in new_held} if new_held else {}
                last_scale, last_emit = scale, d
            if record:
                trades.append({"date": d, "held": new_held, "scale": scale,
                               "added": [x for x in new_held if x not in set(held)],
                               "dropped": [x for x in held if x not in set(new_held)]})
            held = new_held
        elif abs(scale - last_scale) > 1e-9 and held:
            weights[d] = {x: scale / len(held) for x in held}
            last_scale, last_emit = scale, d
    return Book(weights=weights, trades=trades, held_after=held)


def weights_frame(book: Book) -> pd.DataFrame:
    """Sparse target-weight frame: one row per order day, NaN = 0 within the row."""
    if not book.weights:
        return pd.DataFrame()
    rows = {d: (w if w else {"__cash__": 0.0}) for d, w in book.weights.items()}
    W = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    return W.drop(columns="__cash__", errors="ignore")
