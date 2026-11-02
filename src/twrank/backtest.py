"""Daily backtester: trade at the open, costs on traded notional.

A target-weight frame ``W`` has one row per order day; the row is executed at
that day's open (weights missing from the row are sold).  Between order days
the weights drift with prices.

Daily P&L = overnight return on the previous close's weights, then the open
trade, then open-to-close return on the post-trade weights.  Buys pay 0.1425%,
sells pay 0.1425% + 0.3% tax.  A stock with no bar that day cannot be traded
and is carried flat.  An order to buy a stock that opens at its limit-up
(or to sell one that opens at its limit-down) is skipped and the drifted weight
is kept; the gap is measured on adjusted prices so ex-dividend days do not look
like limit moves.  Intraday limit locks, slippage and lot sizes are not
modelled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BUY_COST, SELL_COST
from .data import load_panel
from .features import calendar

LIMIT_CHANGE = pd.Timestamp("2015-06-01")  # TWSE/TPEx daily limit 7% -> 10%


@dataclass
class Result:
    ret: pd.Series  # daily net return
    cost: pd.Series  # daily cost as a fraction of NAV
    turnover: pd.Series  # buys + sells as a fraction of NAV
    weights: pd.DataFrame  # end-of-day weights (columns: stocks ever targeted)

    @property
    def nav(self) -> pd.Series:
        return (1 + self.ret).cumprod()


class Market:
    """Arrays for the backtest, restricted on demand to a set of columns."""

    def __init__(self):
        days = calendar()
        self.days = days
        self.open = load_panel("open").reindex(days)
        self.close = load_panel("close").reindex(days)

    def arrays(self, cols: list[str], i0: int, i1: int):
        ao = self.open[cols].to_numpy(dtype="float64")
        ac = self.close[cols].to_numpy(dtype="float64")
        prev_c = np.vstack([np.full((1, len(cols)), np.nan), ac[:-1]])
        # carry the last close across missing days for the overnight leg
        prev_c = pd.DataFrame(prev_c).ffill().to_numpy()
        sl = slice(i0, i1)
        return ao[sl], ac[sl], prev_c[sl]


_MARKET: Market | None = None


def market() -> Market:
    global _MARKET
    if _MARKET is None:
        _MARKET = Market()
    return _MARKET


def run(W: pd.DataFrame, start, end, limit_guard: bool = True) -> Result:
    m = market()
    days = m.days
    i0 = days.searchsorted(pd.Timestamp(start))
    i1 = days.searchsorted(pd.Timestamp(end), side="right")
    idx = days[i0:i1]
    cols = sorted(W.columns) if len(W) else []
    if not cols:
        z = pd.Series(0.0, index=idx)
        return Result(z, z.copy(), z.copy(), pd.DataFrame(index=idx))
    ao, ac, pc = m.arrays(cols, i0, i1)
    T, N = ao.shape

    traded = np.isfinite(ao) & np.isfinite(ac)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_on = np.where(traded & np.isfinite(pc), ao / pc - 1, 0.0)
        r_id = np.where(traded, ac / ao - 1, 0.0)
        gap = ao / pc - 1
    lim = np.where(idx < LIMIT_CHANGE, 0.07, 0.10)[:, None]
    if limit_guard:
        no_buy = ~traded | (np.isfinite(gap) & (gap >= lim - 0.002))
        no_sell = ~traded | (np.isfinite(gap) & (gap <= -lim + 0.002))
    else:
        no_buy = no_sell = ~traded

    Wr = W.reindex(columns=cols)
    order_pos = idx.get_indexer(Wr.index)
    orders = {}
    for p, row in zip(order_pos, Wr.to_numpy(dtype="float64"), strict=True):
        if p >= 0:
            orders[p] = np.nan_to_num(row, nan=0.0)

    held = np.zeros(N)
    rets = np.empty(T)
    costs = np.zeros(T)
    tos = np.zeros(T)
    eod = np.empty((T, N))
    for t in range(T):
        pv_on = 1.0 + held @ r_on[t]
        w = held * (1.0 + r_on[t]) / pv_on
        tgt = orders.get(t)
        if tgt is not None:
            new = tgt.copy()
            inc = new > w + 1e-12
            dec = new < w - 1e-12
            blk = (inc & no_buy[t]) | (dec & no_sell[t])
            new[blk] = w[blk]
            dw = new - w
            buy = dw[dw > 0].sum()
            sell = -dw[dw < 0].sum()
            costs[t] = buy * BUY_COST + sell * SELL_COST
            tos[t] = buy + sell
            w = new
        pv_id = 1.0 + w @ r_id[t]
        rets[t] = pv_on * pv_id - 1.0 - costs[t]
        held = w * (1.0 + r_id[t]) / pv_id
        eod[t] = held
    return Result(
        ret=pd.Series(rets, index=idx),
        cost=pd.Series(costs, index=idx),
        turnover=pd.Series(tos, index=idx),
        weights=pd.DataFrame(eod, index=idx, columns=cols),
    )
