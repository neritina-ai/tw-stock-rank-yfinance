"""Going live (``genesis/FIAT-deploy.md``): capital, lot sizes, costs, settlement.

How a capital level is held near its size.  An account that compounds for
twenty years outgrows any small starting capital, and lot-size friction
fades with it.  So every calendar year is run on its own: on the first trade
day of the year the account starts from cash with exactly the capital being
studied, buys the strategy's current book, and trades it until the session
before the next year's first trade day.  The yearly returns are chained into
one series for the statistics.  Every account therefore stays within the
range a year of returns can move it (roughly 0.6x to 1.6x of the capital).

The reference ("frictionless") runs the same yearly windows with fractional
shares, proportional fees at the same discount, no minimum fee and no
slippage, so the gap isolates what whole shares, lots, minimum fees and the
odd-lot market cost.  The book (which names, what exposure) is the same
point-in-time book as in the weight backtest.

Outputs under ``results/deploy/``: ``grid.csv`` (one row per experiment x
capital x scheme x discount x slippage, full period and since 2021),
``yearly.csv`` (yearly returns behind the gaps), ``settlement.csv`` and
``oddlot.json`` / ``oddlot_trades.csv`` (what the odd-lot market data says).
"""

from __future__ import annotations

import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd

from . import experiments as X
from . import lots, metrics
from .config import RESULTS_DIR
from .portfolio import build_book

OUT = RESULTS_DIR / "deploy"
SPECS = ["w20", "m11n20"]
CAPITALS = [1e5, 3e5, 5e5, 1e6, 2e6, 5e6]
FINE_CAPITALS = [1e5, 1.5e5, 2e5, 3e5, 4e5, 5e5, 7e5, 1e6, 1.5e6, 2e6, 3e6, 5e6, 1e7, 2e7]
DISCOUNTS = [1.0, 0.6, 0.28]
SLIPS = [0.001, 0.003, 0.005]
SCHEMES = ["A", "B"]
ODD_ERA = 2021  # first full year of intraday odd-lot trading (it started 2020-10-26)


# ---------------------------------------------------------------------------
# books and yearly windows
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def book(spec_id: str):
    spec = X.get(spec_id)
    return build_book(spec, X.START, X.end_date(), record=True)


@lru_cache(maxsize=4)
def windows(spec_id: str) -> list[tuple[int, pd.Timestamp, pd.Timestamp, dict]]:
    """(year, first day, last day, orders) for every calendar year of the book."""
    b = book(spec_id)
    tdays = pd.DatetimeIndex([t["date"] for t in b.trades])
    held = {t["date"]: t for t in b.trades}
    days = lots.prices().days
    end = X.end_date()
    out = []
    years = sorted(set(tdays.year))
    for k, y in enumerate(years):
        first = tdays[tdays.year == y][0]
        nxt = tdays[tdays.year == years[k + 1]][0] if k + 1 < len(years) else None
        last = days[days.searchsorted(nxt) - 1] if nxt is not None else end
        orders = {d: dict(w) for d, w in b.weights.items() if first < d <= last}
        t = held[first]
        orders[first] = {s: t["scale"] / len(t["held"]) for s in t["held"]} if t["held"] else {}
        out.append((y, first, last, dict(sorted(orders.items()))))
    return out


def ensure_splits(log=print) -> None:
    """Split history of every stock the studied books ever hold (fetched once, resumable)."""
    from .data import fetch_splits
    from .universe import load_universe

    tick_of = dict(zip(load_universe()["stock_id"], load_universe()["ticker"], strict=False))
    names = sorted({s for e in SPECS for t in book(e).trades for s in t["held"]})
    fetch_splits([tick_of[s] for s in names if s in tick_of], log=log)


def run_chain(spec_id: str, capital: float, scheme: str, costs: lots.Costs,
              keep_orders: bool = False):
    """Chained daily returns over the yearly windows (+ order rows if asked)."""
    rets, rows = [], []
    for y, a, b, orders in windows(spec_id):
        sim = lots.simulate(orders, a, b, capital, scheme, costs)
        rets.append(sim.ret)
        if keep_orders:
            o = sim.orders.copy()
            o["year"] = y
            o["window_start"] = o["date"] == a
            rows.append(o)
    ret = pd.concat(rets).sort_index()
    return ret, (pd.concat(rows, ignore_index=True) if keep_orders else None)


def _stats(ret: pd.Series, since: int | None = None) -> dict:
    r = ret[ret.index.year >= since] if since else ret
    m = metrics.summary(r)
    return {"cagr": m["cagr"], "sharpe": m["sharpe"], "mdd": m["mdd"], "vol": m["vol"]}


def _one(args) -> tuple[dict, pd.Series]:
    spec_id, capital, scheme, discount, slip, *minfee = args
    costs = lots.Costs(discount=discount, slip_odd=slip)
    if minfee and not minfee[0]:
        costs = lots.Costs(discount=discount, slip_odd=slip, min_fee_lot=0.0, min_fee_odd=0.0)
    ret, _ = run_chain(spec_id, capital, scheme, costs)
    row = {"exp": spec_id, "capital": capital, "scheme": scheme, "discount": discount,
           "slip": slip, "min_fee": bool(minfee[0]) if minfee else True}
    for k, v in _stats(ret).items():
        row[k] = v
    for k, v in _stats(ret, ODD_ERA).items():
        row[f"{k}_2021"] = v
    yr = metrics.yearly(ret).rename(f"{spec_id}|{capital:g}|{scheme}|{discount:g}|{slip:g}")
    return row, yr


def _warm():
    lots.prices()
    for s in SPECS:
        windows(s)


def sweep(combos: list[tuple], workers: int = 8, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, yrs = [], []
    with ProcessPoolExecutor(max_workers=workers, initializer=_warm) as ex:
        for k, (row, yr) in enumerate(ex.map(_one, combos, chunksize=2)):
            rows.append(row)
            yrs.append(yr)
            if (k + 1) % 50 == 0 or k + 1 == len(combos):
                log(f"  {k + 1}/{len(combos)}")
    return pd.DataFrame(rows), pd.concat(yrs, axis=1)


def empirical_slip() -> float:
    p = OUT / "oddlot.json"
    if not p.exists():
        raise FileNotFoundError("run `twrank odd-lot` first (odd-lot market estimate)")
    return float(json.loads(p.read_text(encoding="utf-8"))["slip_estimate"])


def run(workers: int = 8, log=print) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_splits(log)
    emp = empirical_slip()
    slips = sorted({0.0, *SLIPS, round(emp, 5)})
    combos = []
    for s in SPECS:
        combos += [(s, c, "frac", d, 0.0) for c in [1e6] for d in DISCOUNTS]
        combos += [(s, c, sch, d, sl) for c, sch, d, sl in
                   itertools.product(CAPITALS, SCHEMES, DISCOUNTS, slips)]
        # a finer capital grid for the threshold, at the middle discount
        combos += [(s, c, sch, 0.6, sl) for c, sch, sl in
                   itertools.product(FINE_CAPITALS, SCHEMES, slips) if c not in CAPITALS]
    log(f"deploy sweep: {len(combos)} runs x {len(windows(SPECS[0]))} yearly windows")
    grid, yearly = sweep(combos, workers, log)
    ref = grid[grid["scheme"] == "frac"].set_index(["exp", "discount"])
    for col in ("cagr", "cagr_2021"):
        base = grid.apply(lambda r, c=col: ref.loc[(r["exp"], r["discount"]), c], axis=1)
        grid[f"gap{col[4:]}"] = base - grid[col]
    grid.to_csv(OUT / "grid.csv", index=False)
    yearly.index.name = "year"
    yearly.to_csv(OUT / "yearly.csv")
    decompose(emp, workers, log)
    settlement(log=log)
    return grid


def decompose(emp: float, workers: int = 8, log=print) -> pd.DataFrame:
    """Split the gap: whole shares alone, then minimum fees, then odd-lot slippage (6 折).

    The slippage step uses the measured slippage, or 0.1% when the measurement
    rounds to nothing (so the step still shows what slippage would cost).
    """
    dslip = emp if emp >= SLIPS[0] else SLIPS[0]
    combos = [(s, c, sch, 0.6, 0.0, False) for s in SPECS for c in CAPITALS for sch in SCHEMES]
    log(f"decomposition: {len(combos)} runs without minimum fees or slippage")
    g, _ = sweep(combos, workers, log)
    full = pd.read_csv(OUT / "grid.csv")
    rows = []
    for _, r in g.iterrows():
        def pick(slip, r=r):
            m = full[(full["exp"] == r["exp"]) & (full["capital"] == r["capital"])
                     & (full["scheme"] == r["scheme"]) & (full["discount"] == 0.6)
                     & np.isclose(full["slip"], slip)]
            return m.iloc[0]
        ref = full[(full["exp"] == r["exp"]) & (full["scheme"] == "frac")
                   & (full["discount"] == 0.6)].iloc[0]
        fee = pick(0.0) if (np.isclose(full["slip"], 0.0)).any() else None
        slip = pick(dslip)
        for sfx in ("", "_2021"):
            k = f"cagr{sfx}"
            whole = ref[k] - r[k]
            fee_part = (r[k] - fee[k]) if fee is not None else np.nan
            rows.append({"exp": r["exp"], "capital": r["capital"], "scheme": r["scheme"],
                         "period": "2021-" if sfx else "2006-", "slip": dslip,
                         "whole_shares": whole,
                         "min_fee": fee_part,
                         "slippage": (fee[k] - slip[k]) if fee is not None else np.nan,
                         "total": ref[k] - slip[k]})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "decompose.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# settlement cash
# ---------------------------------------------------------------------------


def settlement(log=print) -> pd.DataFrame:
    """Order-day cash flows at the middle cost case (6 折, empirical slippage)."""
    emp = empirical_slip()
    parts = []
    for s in SPECS:
        for c, sch in itertools.product(CAPITALS, SCHEMES):
            _, o = run_chain(s, c, sch, lots.Costs(discount=0.6, slip_odd=emp), keep_orders=True)
            o.insert(0, "exp", s)
            o.insert(1, "capital", c)
            o.insert(2, "scheme", sch)
            parts.append(o)
    df = pd.concat(parts, ignore_index=True)
    df["pay"] = df["buy"] + 0.0 - df["sell"] + df["fee"] + df["tax"]  # >0: the account pays at T+2
    df["pay_pct"] = df["pay"] / df["nav_plan"]
    df["shortfall_pct"] = (-df["cash_after"]).clip(lower=0) / df["nav_plan"]
    df["cash_before_pct"] = df["cash_before"] / df["nav_plan"]
    df["prev_scale"] = df.groupby(["exp", "capital", "scheme", "year"])["scale"].shift(1)
    df.to_csv(OUT / "settlement.csv", index=False)
    log(f"settlement: {len(df):,} order days")
    return df


# ---------------------------------------------------------------------------
# odd-lot market data
# ---------------------------------------------------------------------------


def oddlot_trades(spec_ids=SPECS) -> pd.DataFrame:
    """Every buy and sell of the experiments on days with odd-lot data, with both quotes."""
    from . import oddlot

    rows = []
    for s in spec_ids:
        for t in book(s).trades:
            if t["date"] < oddlot.ODD_START:
                continue
            for x in t["added"]:
                rows.append({"exp": s, "date": t["date"], "stock": x, "side": "buy"})
            for x in t["dropped"]:
                rows.append({"exp": s, "date": t["date"], "stock": x, "side": "sell"})
    tr = pd.DataFrame(rows)
    q = oddlot.quotes(pd.DatetimeIndex(tr["date"].unique()))
    m = tr.merge(q, on=["date", "stock"], how="left")
    m["vwap"] = m["value_odd"] / m["volume_odd"]
    sign = np.where(m["side"] == "buy", 1.0, -1.0)
    m["slip_first"] = sign * (m["first"] / m["open"] - 1)
    m["slip_vwap"] = sign * (m["vwap"] / m["open"] - 1)
    m["spread"] = (m["ask"] - m["bid"]) / ((m["ask"] + m["bid"]) / 2)
    # the odd-lot premium at one moment: last odd-lot print against the whole-lot close
    m["premium_close"] = m["last"] / m["close"] - 1
    m["regime"] = pd.cut(m["date"], [pd.Timestamp("2020-10-25"), pd.Timestamp("2022-12-18"),
                                     pd.Timestamp("2024-12-01"), pd.Timestamp("2100-01-01")],
                         labels=["3 分鐘", "1 分鐘", "5 秒"])
    return m


def _q(s: pd.Series) -> dict:
    s = s.dropna()
    return {"n": int(len(s)), "mean": float(s.mean()), "median": float(s.median()),
            "p10": float(s.quantile(0.1)), "p90": float(s.quantile(0.9)),
            "trimmed_mean": float(s.clip(s.quantile(0.01), s.quantile(0.99)).mean())}


def oddlot_report(log=print) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    m = oddlot_trades()
    m.to_csv(OUT / "oddlot_trades.csv", index=False)
    ok = m.dropna(subset=["first", "open"])
    out = {"trades": int(len(m)), "with_quotes": int(len(ok)),
           "lot_from_yahoo": int((ok["lot_source"] == "yahoo").sum()),
           "no_odd_trade": int(m["open"].notna().sum() - len(ok))}
    out["first"] = {side: _q(g["slip_first"]) for side, g in ok.groupby("side")}
    out["first"]["all"] = _q(ok["slip_first"])
    out["vwap"] = {side: _q(g["slip_vwap"]) for side, g in ok.groupby("side")}
    out["vwap"]["all"] = _q(ok["slip_vwap"])
    out["by_regime"] = {str(k): _q(g["slip_first"]) for k, g in ok.groupby("regime", observed=True)}
    out["spread"] = _q(ok["spread"])
    out["premium_close"] = _q(ok["premium_close"])
    out["premium_close_by_regime"] = {str(k): _q(g["premium_close"])
                                      for k, g in ok.groupby("regime", observed=True)}
    out["odd_value"] = _q(ok["value_odd"])
    # capacity: one buy of capital/20 against the day's odd-lot value of that stock
    # (scheme A sends all of it to the odd-lot market, scheme B only the part under 1,000 shares)
    cap = {}
    buys = ok[ok["side"] == "buy"]
    for c in CAPITALS:
        n = np.floor(c / 20 / buys["open"])
        for sch, sh in (("A", n), ("B", np.mod(n, 1000))):
            r = sh * buys["open"] / buys["value_odd"]
            cap[f"{sch}|{c:g}"] = {"median_share": float(r.median()),
                                   "over_10pct": float((r > 0.10).mean()),
                                   "over_25pct": float((r > 0.25).mean()),
                                   "over_100pct": float((r > 1.0).mean())}
    out["capacity"] = cap
    # the estimate used as the odd-lot slippage of the backtest: mean cost of the
    # first odd-lot print against the whole-lot open, buys and sells pooled,
    # under the current 5-second matching (1% tails trimmed)
    cur = ok[ok["regime"] == "5 秒"]["slip_first"]
    out["slip_estimate"] = round(max(float(cur.clip(cur.quantile(0.01),
                                                    cur.quantile(0.99)).mean()), 0.0), 4)
    (OUT / "oddlot.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    log(f"odd-lot: {out['with_quotes']}/{out['trades']} trades with quotes; "
        f"slippage estimate {out['slip_estimate']:.2%}")
    return out
