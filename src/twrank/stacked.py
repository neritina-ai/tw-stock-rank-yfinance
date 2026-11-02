"""Stacked equity curve: total equity drawn as the sum of its holdings.

Each holding is a coloured band whose height is its market value; bands are
stacked, and the gap up to the equity line is cash.  A name that survives a
rebalance keeps its band, so a stock that is picked week after week grows into
one long continuous area.

Stacking order is the whole problem: a band only looks continuous when the
total below it does not jump at a rebalance.  The approach:

* **Runs.**  A run is a stretch of consecutive trade days in which a stock is
  held.  Each run occupies one layer for its whole length.
* **Layers.**  Two assignments are tried.  ``handover`` walks forward in time
  and gives each vacated layer to a newcomer (longest newcomer first, lowest
  free layer first); with equal weights a newcomer is about as tall as the
  stock it replaced, so the layers above barely move.  ``longest`` places the
  longest runs first, each in the lowest layer free for its whole run - the
  "longest at the bottom" rule taken literally.
* **Order.**  Layers start sorted by how often they change hands (stable
  layers at the bottom) and are then reordered by an insertion local search
  that minimises the total jump of continuing bands at every rebalance,
  measured as a fraction of equity.  The assignment with the smaller jump wins.
* **Colour.**  One colour per stock for its whole history; stocks whose bands
  touch (adjacent layers, or one handing its layer to the next) get different
  colours where possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PALETTE = [
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#edc948", "#76b7b2",
    "#ff9da7", "#9c755f", "#a0cbe8", "#ffbe7d", "#8cd17d", "#d37295", "#b6992d",
    "#499894", "#fabfd2", "#86bcb6", "#d4a6c8", "#f1ce63", "#bab0ac",
]
CASH = "#e6e5e0"
INK, INK2, GRID = "#111111", "#55544f", "#d9d8d2"


@dataclass(frozen=True)
class Run:
    stock: str
    a: int  # first trade index held
    b: int  # last trade index held (inclusive)

    @property
    def length(self) -> int:
        return self.b - self.a + 1


def build_runs(holdings: list[list[str]]) -> list[Run]:
    runs, open_ = [], {}
    for m, now in enumerate(holdings):
        now = set(now)
        for s in [s for s in open_ if s not in now]:
            runs.append(Run(s, open_.pop(s), m - 1))
        for s in now:
            open_.setdefault(s, m)
    runs += [Run(s, a, len(holdings) - 1) for s, a in open_.items()]
    return sorted(runs, key=lambda r: (r.a, r.stock))


def assign_handover(runs: list[Run], n: int) -> np.ndarray:
    layer = np.full(len(runs), -1)
    starts: dict[int, list[int]] = {}
    for k, r in enumerate(runs):
        starts.setdefault(r.a, []).append(k)
    occ: list[int | None] = []
    for m in range(n):
        for j, k in enumerate(occ):
            if k is not None and runs[k].b < m:
                occ[j] = None
        for k in sorted(starts.get(m, []), key=lambda k: (-runs[k].length, runs[k].stock)):
            free = [j for j, o in enumerate(occ) if o is None]
            j = free[0] if free else len(occ)
            if not free:
                occ.append(None)
            occ[j] = k
            layer[k] = j
    return layer


def assign_longest(runs: list[Run], n: int) -> np.ndarray:
    layer = np.full(len(runs), -1)
    busy: list[np.ndarray] = []
    for k in sorted(range(len(runs)), key=lambda k: (-runs[k].length, runs[k].a, runs[k].stock)):
        r = runs[k]
        j = next((j for j, b in enumerate(busy) if not b[r.a : r.b + 1].any()), None)
        if j is None:
            j = len(busy)
            busy.append(np.zeros(n, dtype=bool))
        busy[j][r.a : r.b + 1] = True
        layer[k] = j
    return layer


@dataclass
class Layout:
    runs: list[Run]
    layer: np.ndarray
    order: np.ndarray  # layer ids bottom -> top
    values: np.ndarray  # (layers, days) market value
    occupant: np.ndarray  # (layers, days) run index or -1
    cost: float
    method: str


def _values(runs, layer, value: pd.DataFrame, trade_pos: np.ndarray):
    T = len(value)
    L = int(layer.max()) + 1 if len(layer) else 0
    vals = np.zeros((L, T))
    occ = np.full((L, T), -1)
    for k, r in enumerate(runs):
        t0 = trade_pos[r.a]
        t1 = trade_pos[r.b + 1] if r.b + 1 < len(trade_pos) else T
        if r.stock in value.columns:
            vals[layer[k], t0:t1] = value[r.stock].to_numpy()[t0:t1]
        occ[layer[k], t0:t1] = k
    return vals, occ


def _jump_inputs(vals, occ, equity, trade_pos):
    bars = trade_pos[1:]
    bars = bars[bars > 0]
    delta = (vals[:, bars] - vals[:, bars - 1]).T
    cont = ((occ[:, bars] == occ[:, bars - 1]) & (occ[:, bars] >= 0)).T
    return delta, cont, equity[bars]


def jump_cost(delta, cont, equity, order) -> float:
    d = delta[:, order]
    below = np.cumsum(d, axis=1) - d
    return float((np.abs(below) * cont[:, order]).sum(axis=1).dot(1.0 / equity))


def optimise(delta, cont, equity, start, passes: int = 12):
    order = list(start)
    best = jump_cost(delta, cont, equity, np.array(order))
    for _ in range(passes):
        improved = False
        for p in range(len(order)):
            item = order[p]
            rest = order[:p] + order[p + 1 :]
            costs = [jump_cost(delta, cont, equity, np.array(rest[:q] + [item] + rest[q:]))
                     for q in range(len(order))]
            q = int(np.argmin(costs))
            if costs[q] < best - 1e-12:
                best, order, improved = costs[q], rest[:q] + [item] + rest[q:], True
        if not improved:
            break
    return np.array(order), best


def layout(holdings, value: pd.DataFrame, equity: np.ndarray, trade_pos) -> Layout:
    runs = build_runs(holdings)
    best = None
    for method, fn in (("handover", assign_handover), ("longest", assign_longest)):
        layer = fn(runs, len(holdings))
        vals, occ = _values(runs, layer, value, trade_pos)
        delta, cont, eq = _jump_inputs(vals, occ, equity, trade_pos)
        changes = (np.diff(occ, axis=1) != 0).sum(axis=1)
        order, cost = optimise(delta, cont, eq, np.argsort(changes, kind="stable"))
        cand = Layout(runs, layer, order, vals, occ, cost, method)
        if best is None or cand.cost < best.cost:
            best = cand
    return best


def assign_colours(lay: Layout) -> dict[str, str]:
    stacked = lay.occupant[lay.order]
    stock_of = np.array([r.stock for r in lay.runs] + [""])
    weight: dict[tuple[str, str], float] = {}

    def add(a, b, w):
        if a and b and a != b:
            key = (a, b) if a < b else (b, a)
            weight[key] = weight.get(key, 0.0) + w

    for j in range(len(stacked) - 1):
        pairs = pd.DataFrame({"a": stock_of[stacked[j]], "b": stock_of[stacked[j + 1]]})
        pairs = pairs[(pairs.a != "") & (pairs.b != "")]
        for (a, b), n in pairs.value_counts().items():
            add(a, b, float(n))
    for row in stacked:
        ch = np.flatnonzero((row[1:] != row[:-1]) & (row[1:] >= 0) & (row[:-1] >= 0))
        for t in ch:
            add(stock_of[row[t]], stock_of[row[t + 1]], 250.0)
    length: dict[str, int] = {}
    for r in lay.runs:
        length[r.stock] = length.get(r.stock, 0) + r.length
    nb: dict[str, dict[str, float]] = {}
    for (a, b), w in weight.items():
        nb.setdefault(a, {})[b] = w
        nb.setdefault(b, {})[a] = w
    colour: dict[str, str] = {}
    used = dict.fromkeys(PALETTE, 0)
    for s in sorted(length, key=lambda s: (-length[s], s)):
        clash = dict.fromkeys(PALETTE, 0.0)
        for o, w in nb.get(s, {}).items():
            if o in colour:
                clash[colour[o]] += w
        c = min(PALETTE, key=lambda c: (clash[c], used[c], PALETTE.index(c)))
        colour[s] = c
        used[c] += 1
    return colour


def _mpl():
    import logging

    import matplotlib

    matplotlib.use("Agg")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft JhengHei", "Noto Sans CJK TC", "Noto Sans TC", "PingFang TC",
                 "WenQuanYi Zen Hei"):
        if name in have:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams.update({
        "font.family": "sans-serif", "axes.unicode_minus": False,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.dpi": 130,
    })
    return plt


def _ink_on(c: str) -> str:
    r, g, b = (int(c[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else "white"


def _segments(row):
    out, t, n = [], 0, len(row)
    while t < n:
        k, u = row[t], t + 1
        while u < n and row[u] == k:
            u += 1
        if k >= 0:
            out.append((int(k), t, u))
        t = u
    return out


def draw(exp_id: str, path: Path | None = None, names: dict | None = None, log=print) -> Layout:
    from . import experiments as X
    from .metrics import bench_returns

    names = names or {}
    summ = X.load_summary(exp_id)
    curve = X.load_curve(exp_id)
    pos = X.load_positions(exp_id)
    trades = X.load_trades(exp_id)
    dates = curve.index
    equity = (1 + curve["ret"]).cumprod().to_numpy()
    w = pos.pivot_table(index="date", columns="stock", values="weight", aggfunc="sum")
    w = w.reindex(index=dates).fillna(0.0)
    value = w.mul(equity, axis=0)
    trade_pos = dates.get_indexer(pd.DatetimeIndex(trades["date"]))
    keep = trade_pos >= 0
    holdings = [h for h, k in zip(trades["held"], keep, strict=True) if k]
    trade_pos = trade_pos[keep]
    lay = layout(holdings, value, equity, trade_pos)
    colours = assign_colours(lay)

    plt = _mpl()
    from .data import load_bench

    bench = (1 + bench_returns(load_bench(), dates)).cumprod().to_numpy()
    x = dates.to_numpy()
    x_ext = np.append(x, x[-1] + np.timedelta64(1, "D"))
    stacked = lay.values[lay.order]
    tops = np.cumsum(stacked, axis=0)
    bots = tops - stacked
    occ = lay.occupant[lay.order]
    shares = stacked / equity
    s_tops = np.cumsum(shares, axis=0)
    s_bots = s_tops - shares

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(20, 11), sharex=True,
                                  gridspec_kw={"height_ratios": [1.35, 1]})
    fig.subplots_adjust(left=0.045, right=0.99, top=0.885, bottom=0.05, hspace=0.06)
    for j in range(len(stacked)):
        for k, t0, t1 in _segments(occ[j]):
            c = colours[lay.runs[k].stock]
            sl = slice(t0, t1 + 1)
            for axis, lo, hi in ((ax, bots, tops), (ax2, s_bots, s_tops)):
                axis.fill_between(x_ext[sl], np.append(lo[j, t0:t1], lo[j, t1 - 1]),
                                  np.append(hi[j, t0:t1], hi[j, t1 - 1]), step="post",
                                  facecolor=c, edgecolor="white", linewidth=0.15)
    top = tops[-1] if len(tops) else np.zeros_like(equity)
    ax.fill_between(x_ext, np.append(top, top[-1]), np.append(equity, equity[-1]), step="post",
                    facecolor=CASH, edgecolor="none")
    st = s_tops[-1] if len(s_tops) else np.zeros_like(equity)
    ax2.fill_between(x_ext, np.append(st, st[-1]), 1.0, step="post", facecolor=CASH,
                     edgecolor="none")
    ax.plot(x, equity, color=INK, linewidth=0.9, label="總權益（扣成本）")
    ax.plot(x, bench, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)), label="0050 含息（同起點）")

    fig.canvas.draw()
    for axis, lo, hi in ((ax, bots, tops), (ax2, s_bots, s_tops)):
        bb = axis.get_window_extent()
        y0, y1 = axis.get_ylim()
        ppu, ppb = bb.height / (y1 - y0), bb.width / len(x)
        for j in range(len(stacked)):
            for k, t0, t1 in _segments(occ[j]):
                sid = lay.runs[k].stock
                label = names.get(sid, sid)
                h = (hi[j, t0:t1] - lo[j, t0:t1]) * ppu
                need = 7.5 * (len(label) + 0.5)
                if (t1 - t0) * ppb < need + 6 or h.max() < 9:
                    continue
                ok = np.flatnonzero(h >= 9)
                t = int(ok[len(ok) // 2]) + t0
                half = int(need / ppb / 2)
                if t - half < t0 or t + half >= t1 or h[t - half - t0 : t + half - t0].min() < 7:
                    continue
                axis.text(x[t], (lo[j, t] + hi[j, t]) / 2, label, fontsize=7, ha="center",
                          va="center", color=_ink_on(colours[sid]), clip_on=True)

    ax.set_xlim(x[0], x_ext[-1])
    ax.set_ylim(0, max(equity.max(), bench.max()) * 1.03)
    ax.set_ylabel("權益（起始 = 1）")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax2.set_ylabel("占權益比例（淺灰 = 現金）")
    for axis in (ax, ax2):
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    m = summ["full"]["strategy"]
    longest = max(lay.runs, key=lambda r: r.length)
    n_names = len({r.stock for r in lay.runs})
    fig.suptitle(f"疊加型 Equity Curve｜{summ['label']}　{dates[0]:%Y-%m}–{dates[-1]:%Y-%m}",
                 x=0.045, ha="left", fontsize=15, y=0.975)
    fig.text(
        0.045, 0.93,
        f"每個色塊是一檔持股的市值，連續持有時色塊相連；換股時新進的股票接手被汰除者的那一層，"
        f"換手最少的層在最下面。上圖為金額，下圖為占比。{summ['note']}\n"
        f"年化 {m['cagr']:+.1%}　Sharpe {m['sharpe']:.2f}　最大回檔 {m['mdd']:.1%}　"
        f"{len(holdings)} 次換股、{n_names} 檔曾持有；連續持有最久："
        f"{names.get(longest.stock, longest.stock)}（{longest.stock}）{longest.length} 次",
        fontsize=10, color=INK2, va="top",
    )
    path = path or (X.EXP_DIR / exp_id / "stacked.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    log(f"{exp_id}: {len(lay.order)} layers ({lay.method}), jump {lay.cost:.2f} -> {path}")
    return lay
