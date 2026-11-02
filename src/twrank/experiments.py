"""Experiment registry, walk-forward stages and result files.

Every experiment is a :class:`~twrank.portfolio.Spec` run continuously from
``START`` to the end of the data; the walk-forward stages are reported as
slices of that run.  The scores are strictly out-of-sample at every date
(rolling refits), so a fixed-parameter run is already a walk-forward run as far
as the model is concerned.  ``wfv_select`` additionally re-picks the portfolio
parameters at each stage from the training window only.

Files per experiment (``results/exp/<id>/``)::

    summary.json     spec, full-period and per-stage metrics, yearly returns,
                     the same for 0050 (total return) and TAIEX
    curve.parquet    daily net return, cost, turnover, exposure
    positions.parquet  end-of-day weight of every holding (long format)
    trades.parquet   every scheduled trade day: holdings, adds, drops, exposure
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from . import backtest, metrics
from .config import RESULTS_DIR
from .data import load_bench, load_index, load_tr, read_lock
from .features import calendar
from .portfolio import Spec, build_book, weights_frame

START = "2006-01-01"  # first walk-forward test stage
TRAIN_START = "2004-01-01"  # first out-of-sample score; selection windows start here
EXP_DIR = RESULTS_DIR / "exp"

STAGES = [
    ("2006-01-01", "2007-12-31"),
    ("2008-01-01", "2009-12-31"),
    ("2010-01-01", "2011-12-31"),
    ("2012-01-01", "2013-12-31"),
    ("2014-01-01", "2015-12-31"),
    ("2016-01-01", "2017-12-31"),
    ("2018-01-01", "2019-12-31"),
    ("2020-01-01", "2021-12-31"),
    ("2022-01-01", "2023-12-31"),
    ("2024-01-01", None),  # through the end of the data
]
#: the six stages of the earlier study (its last one ended 2026-04-19)
ORIGINAL_STAGES = STAGES[4:]

BASE = Spec("base", "基本款", note="每週換股、前 30 名、續抱 4×30、4 週平滑、跌破 120 日 EMA 降半倉")

EXPERIMENTS: list[Spec] = [
    BASE,
    BASE.with_(id="h21", label="只用 21 日", score="s21", note="只用 21 日模型的排名"),
    BASE.with_(id="h63", label="只用 63 日", score="s63", note="只用 63 日模型的排名"),
    BASE.with_(id="nohalf", label="不降半倉", regime="none", note="永遠滿倉"),
]

WFV_GRID = {"band_mult": [4.0, 8.0], "smooth_weeks": [4.0, 8.0], "top_n": [30, 50]}
WFV = Spec("wfv", "每段挑參數", note="每段只用訓練期（2004 年起到該段開始）在續抱 4/8 倍、平滑 4/8 週、"
           "30/50 檔中挑 Sharpe 最高的組合，原封不動用在測試期")


def get(exp_id: str) -> Spec:
    for e in [*all_experiments(), WFV]:
        if e.id == exp_id:
            return e
    raise KeyError(exp_id)


def all_experiments() -> list[Spec]:
    return list(EXPERIMENTS)


def viewer_experiments() -> list[Spec]:
    """Experiments shown in the viewer: the runnable ones plus the walk-forward pick."""
    out = all_experiments()
    if (EXP_DIR / WFV.id / "summary.json").exists():
        out.insert(1, WFV)
    return out


def end_date() -> pd.Timestamp:
    return calendar()[-1]


def stage_bounds(i: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    a, b = STAGES[i]
    return pd.Timestamp(a), (pd.Timestamp(b) if b else end_date())


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------


def run_spec(spec: Spec, start=START, end=None, record: bool = False):
    end = end or end_date()
    book = build_book(spec, start, end, record=record)
    res = backtest.run(weights_frame(book), start, end)
    return book, res


def bench_series(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    return {
        "0050": metrics.bench_returns(load_bench(), index),
        "TR": metrics.bench_returns(load_tr(), index),
        "TAIEX": metrics.bench_returns(load_index()["Close"], index),
    }


def evaluate(res: backtest.Result) -> dict:
    """Full-period, per-stage and yearly statistics for strategy and benchmarks."""
    idx = res.ret.index
    bench = bench_series(idx)

    def block(a, b):
        sl = (idx >= a) & (idx <= b)
        r = res.ret[sl]
        out = {"strategy": metrics.summary(r, res.turnover, res.cost, res.weights)}
        for k, s in bench.items():
            out[k] = metrics.summary(s[sl])
        return out

    full = block(idx[0], idx[-1])
    stages = []
    for i in range(len(STAGES)):
        a, b = stage_bounds(i)
        if a > idx[-1]:
            continue
        st = block(a, b)
        st["stage"] = i + 1
        stages.append(st)
    yr = pd.DataFrame({"strategy": metrics.yearly(res.ret),
                       **{k: metrics.yearly(s) for k, s in bench.items()}})
    return {
        "full": full,
        "stages": stages,
        "yearly": {int(y): {k: float(v) for k, v in row.items()} for y, row in yr.iterrows()},
    }


def save(spec: Spec, book, res: backtest.Result, extra: dict | None = None) -> dict:
    d = EXP_DIR / spec.id
    d.mkdir(parents=True, exist_ok=True)
    ev = evaluate(res)
    lock = read_lock()
    summary = {"id": spec.id, "label": spec.label, "note": spec.note, "spec": asdict(spec),
               "data": {k: lock.get(k) for k in ("digest", "dates", "symbols")}, **ev,
               **(extra or {})}
    (d / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1,
                                               default=str), encoding="utf-8")
    w = res.weights
    curve = pd.DataFrame({"ret": res.ret, "cost": res.cost, "turnover": res.turnover,
                          "exposure": w.sum(axis=1) if len(w.columns) else 0.0})
    curve.to_parquet(d / "curve.parquet")
    if len(w.columns):
        long = w.where(w > 1e-9).stack().rename("weight").reset_index()
        long.columns = ["date", "stock", "weight"]
        long["weight"] = long["weight"].astype("float32")
        long.to_parquet(d / "positions.parquet", index=False)
    if book.trades:
        tr = pd.DataFrame(book.trades)
        for c in ("held", "added", "dropped"):
            tr[c] = tr[c].map(lambda x: " ".join(x))
        tr.to_parquet(d / "trades.parquet", index=False)
    return summary


def load_summary(exp_id: str) -> dict:
    return json.loads((EXP_DIR / exp_id / "summary.json").read_text(encoding="utf-8"))


def load_curve(exp_id: str) -> pd.DataFrame:
    return pd.read_parquet(EXP_DIR / exp_id / "curve.parquet")


def load_trades(exp_id: str) -> pd.DataFrame:
    tr = pd.read_parquet(EXP_DIR / exp_id / "trades.parquet")
    for c in ("held", "added", "dropped"):
        tr[c] = tr[c].map(lambda s: s.split() if s else [])
    return tr


def load_positions(exp_id: str) -> pd.DataFrame:
    return pd.read_parquet(EXP_DIR / exp_id / "positions.parquet")


# ---------------------------------------------------------------------------
# walk-forward parameter selection
# ---------------------------------------------------------------------------


def _grid(space: dict) -> list[dict]:
    import itertools

    keys = list(space)
    return [dict(zip(keys, v, strict=True)) for v in itertools.product(*space.values())]


def wfv_select(base: Spec, space: dict = WFV_GRID, log=print):
    """Pick parameters per stage on [TRAIN_START, stage start) by Sharpe; run them frozen.

    The test run carries the holdings across stage boundaries (one continuous
    book whose parameters change at each boundary).
    """
    rows = []
    held: list[str] = []
    W_parts = []
    trades = []
    end = end_date()
    for i in range(len(STAGES)):
        a, b = stage_bounds(i)
        best = None
        for p in _grid(space):
            spec = base.with_(**p)
            _, res = run_spec(spec, TRAIN_START, a - pd.Timedelta(days=1))
            sh = metrics.summary(res.ret).get("sharpe", np.nan)
            if best is None or (np.isfinite(sh) and sh > best[1] + 1e-12):
                best = (p, sh)
        p, sh = best
        spec = base.with_(**p)
        book = build_book(spec, a, min(b, end), held=held, record=True)
        held = book.held_after
        W_parts.append(weights_frame(book))
        trades += book.trades
        rows.append({"stage": i + 1, "test": f"{a.date()}..{min(b, end).date()}",
                     **p, "train_sharpe": round(float(sh), 3)})
        log(f"stage {i + 1}: {p} train Sharpe {sh:.3f}")
    W = pd.concat(W_parts).fillna(0.0)
    W = W[~W.index.duplicated(keep="last")].sort_index()
    from .portfolio import Book

    book = Book(weights={}, trades=trades, held_after=held)
    res = backtest.run(W, START, end)
    sel = pd.DataFrame(rows)
    return sel, book, res
