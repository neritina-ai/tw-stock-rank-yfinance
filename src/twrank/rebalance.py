"""Weekly versus monthly rebalancing (``genesis/FIAT-rebalance.md``).

Three sweeps, all on the same out-of-sample scores and the same walk-forward
stages as the base experiment:

``grid``
    schedule (weekly, or monthly on day 1..28) x book size (5..30), every
    other setting as in the base experiment.
``settings``
    for the weekly schedule and the recommended monthly day and size: the
    buffer (band multiple), the hold-score smoothing and when the TAIEX
    120-day EMA is checked (only on trade days, every week, every day).
``wf``
    the honest version of picking a day: at every stage, choose the monthly
    day and size with the best training-window Sharpe (2004 to the stage
    start) and use it frozen in the test stage.

Every configuration's daily net returns are kept so stage and yearly
statistics, and the consistency of a day's rank across years, can be computed
later without re-running anything.
"""

from __future__ import annotations

import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import experiments as X
from . import metrics
from .config import RESULTS_DIR
from .portfolio import Spec

OUT = RESULTS_DIR / "rebalance"
DAYS = list(range(1, 29))
SIZES = [5, 6, 7, 8, 10, 12, 15, 20, 25, 30]
BANDS = [2.0, 3.0, 4.0, 6.0, 8.0]
SMOOTHS = [1.0, 2.0, 4.0, 8.0, 13.0]
CHECKS = ["trade", "weekly", "daily"]


def _key(spec: Spec) -> str:
    p = spec.params()
    sched = "W" if p["schedule"] == "weekly" else f"M{p['day']:02d}"
    return (f"{sched}_n{p['top_n']}_b{p['band_mult']:g}_s{p['smooth_weeks']:g}_"
            f"{p['regime']}_{p['regime_check']}_{p['score']}")


def _warm():
    from .portfolio import hold_panel, risk_on, score_panel

    score_panel("ens")
    hold_panel("ens", 4.0)
    risk_on()
    from .backtest import market

    market()


def _run_one(spec: Spec):
    _, res = X.run_spec(spec)
    m = metrics.summary(res.ret, res.turnover, res.cost, res.weights)
    row = {"key": _key(spec), **spec.params(), **{k: m[k] for k in
           ("cagr", "sharpe", "mdd", "vol", "turnover", "cost", "positions", "exposure")}}
    for i in range(len(X.STAGES)):
        a, b = X.stage_bounds(i)
        r = res.ret[(res.ret.index >= a) & (res.ret.index <= b)]
        s = metrics.summary(r, res.turnover, res.cost)
        row[f"s{i + 1}_sharpe"] = s.get("sharpe")
        row[f"s{i + 1}_cagr"] = s.get("cagr")
        row[f"s{i + 1}_mdd"] = s.get("mdd")
        row[f"s{i + 1}_turnover"] = s.get("turnover")
    return row, res.ret.astype("float32").rename(row["key"])


def run_many(specs: list[Spec], name: str, workers: int | None = None, log=print) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, min(12, (os.cpu_count() or 4) - 4))
    rows, rets = [], []
    with ProcessPoolExecutor(max_workers=workers, initializer=_warm) as ex:
        for k, (row, ret) in enumerate(ex.map(_run_one, specs, chunksize=2)):
            rows.append(row)
            rets.append(ret)
            if (k + 1) % 25 == 0 or k + 1 == len(specs):
                log(f"{name}: {k + 1}/{len(specs)}")
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / f"{name}.csv", index=False)
    pd.concat(rets, axis=1).to_parquet(OUT / f"{name}_returns.parquet")
    return grid


def grid_specs() -> list[Spec]:
    out = []
    for n in SIZES:
        out.append(X.BASE.with_(id=f"W_n{n}", top_n=n))
        for d in DAYS:
            out.append(X.BASE.with_(id=f"M{d:02d}_n{n}", schedule="monthly", day=d, top_n=n))
    return out


def settings_specs(size: int) -> list[Spec]:
    """Buffer x smoothing x EMA-check timing, weekly and on every monthly day."""
    out = []
    bases = [X.BASE.with_(top_n=size)]
    bases += [X.BASE.with_(schedule="monthly", day=d, top_n=size) for d in DAYS]
    for base in bases:
        for b, s, c in itertools.product(BANDS, SMOOTHS, CHECKS):
            if base.schedule == "weekly" and c == "weekly":
                continue  # identical to checking on trade days
            out.append(base.with_(id=_key(base), band_mult=b, smooth_weeks=s, regime_check=c))
    return out


def load(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.read_csv(OUT / f"{name}.csv"), pd.read_parquet(OUT / f"{name}_returns.parquet"))


# ---------------------------------------------------------------------------
# walk-forward choice of day and size
# ---------------------------------------------------------------------------


def wf_choice(name: str = "grid", refresh: bool = False, log=print) -> pd.DataFrame:
    """Per stage, pick (monthly day, size) by training Sharpe; report the test stage.

    Training window: 2004-01 to the stage start, on returns of runs that
    started in 2004 (separate from the 2006+ runs in ``grid``).
    """
    specs = [X.BASE.with_(id=f"M{d:02d}_n{n}", schedule="monthly", day=d, top_n=n)
             for d in DAYS for n in SIZES]
    specs += [X.BASE.with_(id=f"W_n{n}", top_n=n) for n in SIZES]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "train_returns.parquet"
    if path.exists() and not refresh:
        tr = pd.read_parquet(path)
    else:
        workers = max(1, min(12, (os.cpu_count() or 4) - 4))
        with ProcessPoolExecutor(max_workers=workers, initializer=_warm) as ex:
            rets = list(ex.map(_run_train, specs, chunksize=2))
        tr = pd.concat(rets, axis=1)
        tr.to_parquet(path)
    _, test = load(name)
    rows = []
    for i in range(len(X.STAGES)):
        a, b = X.stage_bounds(i)
        win = tr[tr.index < a]
        sh = win.mean() / win.std() * np.sqrt(252)
        for fam, cols in (("monthly", [c for c in sh.index if c.startswith("M")]),
                          ("weekly", [c for c in sh.index if c.startswith("W")])):
            best = sh[cols].idxmax()
            t = test.loc[(test.index >= a) & (test.index <= b), _test_key(best)]
            s = metrics.summary(t.astype("float64"))
            rows.append({"stage": i + 1, "family": fam, "pick": best,
                         "train_sharpe": float(sh[best]), "test_sharpe": s.get("sharpe"),
                         "test_cagr": s.get("cagr"), "test_mdd": s.get("mdd")})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "wf_choice.csv", index=False)
    log(out.to_string())
    return out


def _test_key(spec_id: str) -> str:
    if spec_id.startswith("W"):
        n = int(spec_id.split("_n")[1])
        return _key(X.BASE.with_(top_n=n))
    d, n = spec_id[1:3], spec_id.split("_n")[1]
    return _key(X.BASE.with_(schedule="monthly", day=int(d), top_n=int(n)))


def _run_train(spec: Spec) -> pd.Series:
    _, res = X.run_spec(spec, start=X.TRAIN_START, end=X.end_date())
    return res.ret.astype("float32").rename(spec.id)


def save_json(obj, name: str) -> None:
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str),
                            encoding="utf-8")


# ---------------------------------------------------------------------------
# experiments added to the viewer (chosen from the sweeps; see SUMMARY.md)
# ---------------------------------------------------------------------------

REC_DAY = 11  # best day on the day-smoothed Sharpe map at 20 names (days 9-15 all similar)
REC_SIZE = 20  # on the plateau of both schedules (weekly 12-20, monthly 20-30)

REBALANCE_EXPERIMENTS = [
    X.BASE.with_(id="w20", label="每週 20 檔", top_n=REC_SIZE, tags=("上線",),  # live.plan's default
                 note="建議設定：每週換股、前 20 名，其餘同基本款"),
    X.BASE.with_(id="m11n20", label=f"每月 {REC_DAY} 日 20 檔", schedule="monthly", day=REC_DAY,
                 top_n=REC_SIZE, note=f"每月換股的建議：每月 {REC_DAY} 日（休市順延）、前 20 名，其餘同基本款"),
    X.BASE.with_(id="m11n30", label=f"每月 {REC_DAY} 日 30 檔", schedule="monthly", day=REC_DAY,
                 note=f"只把基本款改成每月 {REC_DAY} 日換股"),
    X.BASE.with_(id="w8", label="每週 8 檔", top_n=8,
                 note="接近臨界值的集中組合：每週換股、前 8 名"),
]
