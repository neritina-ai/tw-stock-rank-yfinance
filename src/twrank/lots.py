"""Integer-share simulation: real share counts, lot sizes, minimum fees, odd-lot slippage.

``backtest.run`` trades weights: any fraction of a share, costs proportional
to the traded value, fills at the open.  This module trades an account in
New Taiwan dollars instead:

* **Planning** uses what is known the evening before: the previous close.
  Target shares = weight x NAV / previous close, rounded down to whole shares.
* **Execution** at the trade day's open.  Whole lots (1,000 shares) fill at
  the open; odd lots fill at the open moved against us by ``slip_odd``
  (intraday odd-lot trading has its own book and its first match is at 9:10).
  An order to buy a stock that opens at its limit-up, or to sell one that
  opens at its limit-down, or in a stock without a bar that day, is skipped.
* **Schemes**: ``A`` sends every share as odd-lot orders (1..999 shares each);
  ``B`` sends the multiple-of-1,000 part of each order as whole lots and the
  rest as one odd-lot order; ``frac`` is the reference: fractional shares,
  proportional fees, no minimum, no slippage.
* **Costs** per order: fee = max(minimum, floor(value x 0.1425% x discount)),
  sells also pay floor(value x 0.3%) securities tax.
* **Cash** that buys do not use stays in the account (no interest).  Sells and
  buys of one day settle together at T+2, so cash may dip below zero when the
  open gaps away from the previous close or a sell is blocked; the depth of
  that dip is the settlement buffer the account needs (``shortfall``).

Prices.  Yahoo's raw prices are back-adjusted for later splits, and Yahoo
books stock dividends as splits, so the traded price is ``raw x split
factor`` (``data.split_factor``).  Positions are valued with the adjusted
(total return) prices, exactly as in ``backtest.run``; cash dividends and the
odd shares left by stock dividends are moved into cash before each trade day.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import LIMIT_CHANGE
from .data import load_panel, split_factor
from .features import calendar

FEE_RATE = 0.001425
TAX_RATE = 0.003
LOT = 1000
ODD_MAX = 999  # an odd-lot order carries 1..999 shares


@dataclass(frozen=True)
class Costs:
    discount: float = 1.0  # fee multiplier: 0.6 is a 40% discount ("6 折")
    min_fee_lot: float = 20.0  # assumption: common minimum for whole-lot orders
    min_fee_odd: float = 1.0  # assumption: common minimum for odd-lot orders
    slip_odd: float = 0.0  # one-way odd-lot fill versus the whole-lot open


@dataclass
class Sim:
    nav: pd.Series  # end-of-day account value (NT$)
    ret: pd.Series  # daily return
    cash: pd.Series  # end-of-day cash (NT$)
    orders: pd.DataFrame  # one row per order day: flows, fees, shortfall
    fills: pd.DataFrame | None = None  # one row per stock and order day (record=True)
    shares: dict = field(default_factory=dict)  # final holdings {stock: shares}


class Prices:
    """Arrays shared by every simulation (loaded once per process)."""

    def __init__(self):
        days = calendar()
        self.days = days
        adj_c = load_panel("close").reindex(days)
        self.adj_open = load_panel("open").reindex(days)
        self.adj_close = adj_c
        self.adj_close_ff = adj_c.ffill()
        self.raw_open = load_panel("raw_open").reindex(days)
        self.raw_close_ff = load_panel("raw_close").reindex(days).ffill()
        # split factor of each session: raw x factor = price in that day's shares
        self.factor = split_factor(days).reindex(columns=adj_c.columns).fillna(1.0)

    def block(self, cols: list[str], i0: int, i1: int) -> dict[str, np.ndarray]:
        sl = slice(i0, i1)

        def arr(df):
            return df[cols].to_numpy("float64")

        def prev(a):
            return np.vstack([np.full((1, a.shape[1]), np.nan), a[:-1]])[sl]

        k = arr(self.factor)[sl]
        ao = arr(self.adj_open)[sl]
        ac = arr(self.adj_close)[sl]
        acf = arr(self.adj_close_ff)
        po = arr(self.raw_open)[sl] * k
        prev_acf = prev(acf)
        # previous close in today's shares (the exchange's reference price after a split)
        prev_px = prev(arr(self.raw_close_ff)) * k
        traded = np.isfinite(ao) & np.isfinite(ac) & np.isfinite(po)
        gap = ao / prev_acf - 1
        lim = np.where(self.days[sl] < LIMIT_CHANGE, 0.07, 0.10)[:, None]
        with np.errstate(invalid="ignore"):
            no_buy = ~traded | (np.isfinite(gap) & (gap >= lim - 0.002))
            no_sell = ~traded | (np.isfinite(gap) & (gap <= -lim + 0.002))
        return {"adj_open": ao, "adj_close_ff": acf[sl], "px_open": po, "factor": k,
                "prev_adj": prev_acf, "prev_px": prev_px, "no_buy": no_buy,
                "no_sell": no_sell}


_PRICES: Prices | None = None


def prices() -> Prices:
    global _PRICES
    if _PRICES is None:
        _PRICES = Prices()
    return _PRICES


def order_fees(shares: np.ndarray, price: np.ndarray, sell: np.ndarray, scheme: str,
               c: Costs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fee, tax, odd-lot share count and number of orders for absolute share counts.

    ``price`` is the whole-lot price; odd-lot shares pay ``price x (1 +/- slip)``.
    Returns (fee, tax, notional actually paid or received, odd shares).
    """
    n = np.abs(shares)
    rate = FEE_RATE * c.discount
    if scheme == "frac":
        val = n * price
        return val * rate, np.where(sell, val * TAX_RATE, 0.0), val, np.zeros_like(n)
    if scheme == "A":
        lots = np.zeros_like(n)
    elif scheme == "B":
        lots = np.floor(n / LOT) * LOT
    else:
        raise ValueError(scheme)
    odd = n - lots
    p_odd = price * np.where(sell, 1 - c.slip_odd, 1 + c.slip_odd)
    v_lot = lots * price
    full, rest = np.floor(odd / ODD_MAX), np.mod(odd, ODD_MAX)
    v_full = ODD_MAX * p_odd
    v_rest = rest * p_odd

    def fee(v, minimum):
        return np.maximum(minimum, np.floor(v * rate))

    f = np.where(lots > 0, fee(v_lot, c.min_fee_lot), 0.0)
    f += full * fee(v_full, c.min_fee_odd) + np.where(rest > 0, fee(v_rest, c.min_fee_odd), 0.0)
    t = np.zeros_like(n)
    if sell.any():
        t = np.where(sell, np.floor(v_lot * TAX_RATE) + full * np.floor(v_full * TAX_RATE)
                     + np.floor(v_rest * TAX_RATE), 0.0)
    return f, t, v_lot + full * v_full + v_rest, odd


NO_FRICTION = Costs()


def simulate(orders: dict, start, end, capital: float, scheme: str = "B",
             costs: Costs = NO_FRICTION, plan_at: str = "close", record: bool = False) -> Sim:
    """Run an account from cash through ``[start, end]``.

    ``orders`` maps an order day to target weights ``{stock: weight}`` (stocks
    missing from a row are sold).  ``plan_at="open"`` sizes the orders with the
    open prices instead of the previous close; it exists only to check the
    fractional scheme against ``backtest.run``.
    """
    P = prices()
    days = P.days
    i0 = days.searchsorted(pd.Timestamp(start))
    i1 = days.searchsorted(pd.Timestamp(end), side="right")
    idx = days[i0:i1]
    cols = sorted({s for w in orders.values() for s in w})
    T, N = len(idx), len(cols)
    if T == 0:
        raise ValueError("empty window")
    A = P.block(cols, i0, i1) if N else None
    pos = {idx.get_loc(d): w for d, w in orders.items() if d in idx}

    sh = np.zeros(N)  # shares actually held
    u = np.zeros(N)  # adjusted units: value = u x adjusted price
    k_last = np.ones(N)  # split factor at the last sweep
    cash = float(capital)
    nav = np.empty(T)
    cash_eod = np.empty(T)
    o_rows, f_rows = [], []
    kf = A["factor"] if N else None

    for t in range(T):
        if t in pos and N:
            prev_px, prev_adj = A["prev_px"][t], A["prev_adj"][t]
            ao, px = A["adj_open"][t], A["px_open"][t]
            held = sh > 0
            # stock dividends / splits since the last trade day: scale the share count to
            # today's shares; the value of a fractional remainder stays in ``u`` and is
            # paid out as cash by the sweep at the open below
            sh = np.where(held, sh * k_last / kf[t], sh)
            if scheme != "frac":
                sh = np.floor(sh + 1e-6)
            k_last = kf[t].copy()

            # plan: the account value at the previous close (dividends receivable included)
            val_prev = np.where(np.isfinite(prev_adj), u * prev_adj, 0.0)
            if plan_at == "open":
                ref_px = px
                ref_val = np.where(np.isfinite(ao), u * ao, val_prev)
            else:
                ref_px = prev_px
                ref_val = val_prev
            nav_plan = cash + float(ref_val.sum())

            # at the open: what the adjusted value holds beyond shares x price is the cash
            # dividend of every ex-date since the last sweep (an ex-date today included)
            live = held & np.isfinite(ao) & np.isfinite(px)
            cash += float(np.sum(np.where(live, u * ao - sh * px, 0.0)))
            u = np.where(live, sh * px / ao, u)

            w = np.zeros(N)
            for s, x in pos[t].items():
                w[cols.index(s)] = x
            with np.errstate(invalid="ignore", divide="ignore"):
                tgt = w * nav_plan / ref_px
            tgt = np.where(np.isfinite(tgt), tgt, sh)
            if scheme != "frac":
                tgt = np.floor(tgt + 1e-9)
            d = tgt - sh
            d[np.abs(d) < 1e-9] = 0.0
            blocked = ((d > 0) & A["no_buy"][t]) | ((d < 0) & A["no_sell"][t])
            b_sell = blocked & (d < 0)
            b_sell_val = float(np.nansum(np.where(b_sell, -d * prev_px, 0.0)))
            d[blocked] = 0.0
            sell = d < 0
            trade = d != 0
            d = np.where(trade, d, 0.0)
            pxs = np.where(trade, px, 0.0)
            fee, tax, notional, odd = order_fees(d, pxs, sell, scheme, costs)
            fee, tax, notional, odd = (np.where(trade, x, 0.0) for x in (fee, tax, notional, odd))
            buy_amt = float(notional[~sell].sum())
            sell_amt = float(notional[sell].sum())
            cash_before = cash
            cash += sell_amt - buy_amt - float(fee.sum()) - float(tax.sum())
            u = u + np.where(trade, d * px / ao, 0.0)
            sh = sh + d
            mid = np.abs(d) * pxs
            o_rows.append({
                "date": idx[t], "nav_plan": nav_plan, "cash_before": cash_before,
                "buy": buy_amt, "sell": sell_amt, "fee": float(fee.sum()), "tax": float(tax.sum()),
                "slip": float(np.abs(notional - mid).sum()), "cash_after": cash,
                "n_orders": int(trade.sum()), "n_odd": int(((odd > 0) & trade).sum()),
                "odd_value": float((odd * pxs).sum()), "blocked": int(blocked.sum()),
                "blocked_sell": int(b_sell.sum()), "blocked_sell_value": b_sell_val,
                "scale": float(sum(pos[t].values())),
            })
            if record:
                for j in np.nonzero(trade | blocked)[0]:
                    f_rows.append({"date": idx[t], "stock": cols[j], "shares": d[j],
                                   "odd": odd[j], "price": px[j], "fee": fee[j], "tax": tax[j],
                                   "notional": notional[j], "blocked": bool(blocked[j]),
                                   "held_after": sh[j]})
            sh[np.abs(sh) < 1e-9] = 0.0
            u[sh == 0] = 0.0
        if N:
            v = u * A["adj_close_ff"][t]
            mv = float(np.nansum(v))
        else:
            mv = 0.0
        nav[t] = cash + mv
        cash_eod[t] = cash
    nav_s = pd.Series(nav, index=idx)
    ret = nav_s.pct_change()
    ret.iloc[0] = nav[0] / capital - 1
    return Sim(nav=nav_s, ret=ret, cash=pd.Series(cash_eod, index=idx),
               orders=pd.DataFrame(o_rows),
               fills=pd.DataFrame(f_rows) if record else None,
               shares={cols[j]: float(sh[j]) for j in np.nonzero(sh)[0]} if N else {})
