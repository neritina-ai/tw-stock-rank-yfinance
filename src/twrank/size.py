"""How many names to hold by hand: weekly 12 against weekly 20 (``results/size``).

The rebalance sweep put weekly 12 to 20 names on one Sharpe plateau and the
first recommendation took 20, its highest point.  An account traded by hand
pays for every name every week, so this module asks whether the smaller end
of the plateau costs anything:

``run``
    the weekly buffer x smoothing x EMA-check sweep of
    ``rebalance.settings_specs`` repeated at 10, 12 and 15 names (20 names is
    already in ``settings_n20``), saved as ``settings_weekly_small``.
``build``
    ``results/size/SUMMARY.md`` and two figures: full-period, stage and
    yearly numbers, a block bootstrap of the Sharpe and return gap, the gap
    under every other setting of the sweep, the orders an account sends each
    week and the integer-share friction (both from ``lots``, as the deploy
    study measures them).
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from . import deploy as D
from . import experiments as X
from . import lots, metrics
from . import rebalance as R
from .config import RESULTS_DIR
from .data import load_bench
from .stacked import GRID, INK, INK2, _mpl

OUT = RESULTS_DIR / "size"
SMALL = [10, 12, 15]
PAIR = ("w12", "w20")
SWEEP = "settings_weekly_small"
BLOCK = 63  # trading days per bootstrap block (about a quarter)
REPS = 2000
CAPITALS = [1e5, 3e5, 1e6]
COLOURS = {"w12": "#2f6db5", "w20": "#e07b39", "0050": INK2}


def sweep_specs() -> list:
    out = []
    for n in SMALL:
        base = X.BASE.with_(top_n=n)
        for b, s, c in itertools.product(R.BANDS, R.SMOOTHS, ["trade", "daily"]):
            out.append(base.with_(id=R._key(base), band_mult=b, smooth_weeks=s, regime_check=c))
    return out


def run(log=print) -> None:
    R.run_many(sweep_specs(), SWEEP, log=log)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def bootstrap(a: pd.Series, b: pd.Series, block: int = BLOCK, reps: int = REPS,
              seed: int = 0) -> dict:
    """Circular block bootstrap of the annual Sharpe and log-return gap a - b."""
    A, B = a.to_numpy(), b.to_numpy()
    T = len(A)
    rng = np.random.default_rng(seed)
    ds, dr = np.empty(reps), np.empty(reps)
    for k in range(reps):
        starts = rng.integers(0, T, size=T // block + 1)
        ii = ((starts[:, None] + np.arange(block)[None, :]).ravel()[:T]) % T
        x, y = A[ii], B[ii]
        ds[k] = (x.mean() / x.std() - y.mean() / y.std()) * np.sqrt(252)
        dr[k] = (np.log1p(x).mean() - np.log1p(y).mean()) * 252
    q = lambda v: [float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))]  # noqa: E731
    return {"sharpe": float(np.mean(ds)), "sharpe_90": q(ds), "p_sharpe": float(np.mean(ds > 0)),
            "logret": float(np.mean(dr)), "logret_90": q(dr), "p_logret": float(np.mean(dr > 0))}


def workload(exp_id: str) -> dict:
    """Name changes per week and holding spells, from the recorded trades."""
    tr = X.load_trades(exp_id)
    adds, drops = tr["added"].map(len), tr["dropped"].map(len)
    spans, since, prev = [], {}, set()
    for i, held in enumerate(tr["held"]):
        h = set(held)
        for s in h - prev:
            since[s] = i
        for s in prev - h:
            spans.append(i - since.pop(s))
        prev = h
    d = pd.DatetimeIndex(tr["date"])
    years = (d[-1] - d[0]).days / 365.25
    return {"weeks": len(tr), "adds": float(adds.mean()), "drops": float(drops.mean()),
            "change_weeks": float((adds > 0).mean()), "trips_per_year": len(spans) / years,
            "hold_median": float(np.median(spans)), "hold_mean": float(np.mean(spans))}


def accounts(exp_id: str, slip: float) -> list[dict]:
    """Integer-share accounts at plan's defaults (scheme B, 6 折): orders, friction, cash dip.

    Orders and friction are measured from the odd-lot era on; the settlement dip (a sell
    blocked at limit-down or by a halt while the buys go through) over the whole period,
    because it comes from rare events.
    """
    rows = []
    for cap in CAPITALS:
        ret_f, _ = D.run_chain(exp_id, cap, "frac", lots.Costs(discount=0.6, min_fee_lot=0.0,
                                                                 min_fee_odd=0.0))
        ret, o = D.run_chain(exp_id, cap, "B", lots.Costs(discount=0.6, slip_odd=slip),
                             keep_orders=True)
        short = (-o["cash_after"]).clip(lower=0) / o["nav_plan"]
        blocked = int((o["blocked_sell"] > 0).sum())
        o = o[o["date"].dt.year >= D.ODD_ERA]
        years = (o["date"].iloc[-1] - pd.Timestamp(f"{D.ODD_ERA}-01-01")).days / 365.25
        s, f = D._stats(ret, D.ODD_ERA), D._stats(ret_f, D.ODD_ERA)
        rows.append({"exp": exp_id, "capital": cap, "cagr": s["cagr"], "gap": f["cagr"] - s["cagr"],
                     "order_days": len(o) / years, "stocks_per_order_day": float(o["n_orders"].mean()),
                     "stocks_per_year": float(o["n_orders"].sum()) / years,
                     "shortfall_99": float(short.quantile(0.99)), "shortfall_max": float(short.max()),
                     "blocked_days": blocked})
    return rows


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def equity_figure(path) -> None:
    plt = _mpl()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1]})
    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.07, hspace=0.07)
    navs = {}
    for e in PAIR:
        c = X.load_curve(e)
        navs[e] = (1 + c["ret"]).cumprod()
    bench = (1 + metrics.bench_returns(load_bench(), navs[PAIR[0]].index)).cumprod()
    lines = ((PAIR[1], navs[PAIR[1]], "每週 20 檔", 1.3, "-"),
             (PAIR[0], navs[PAIR[0]], "每週 12 檔（上線）", 1.8, "-"),
             ("0050", bench, "0050 含息", 1.1, (0, (4, 3))))
    ends = sorted((nav.iloc[-1], key) for key, nav, *_ in lines)
    shift, last = {}, None
    for v, key in ends:  # end labels at least ~11 points apart on the log axis
        shift[key] = 0.0 if last is None or np.log10(v / last[0]) > 0.05 else shift[last[1]] + 11
        last = (v, key)
    for key, nav, lab, lw, ls in lines:
        ax.plot(nav.index, nav, color=COLOURS[key], lw=lw, ls=ls, label=lab)
        ax.annotate(f"{nav.iloc[-1]:.1f}×", (nav.index[-1], nav.iloc[-1]), xytext=(6, shift[key]),
                    textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.set_yscale("log")
    ax.set_ylabel("權益（起始 = 1，對數）")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    rel = navs[PAIR[0]] / navs[PAIR[1]]
    ax2.axhline(1.0, color=INK2, lw=0.8)
    ax2.plot(rel.index, rel, color=COLOURS[PAIR[0]], lw=1.2)
    ax2.fill_between(rel.index, 1.0, rel, where=rel >= 1, color=COLOURS[PAIR[0]], alpha=0.15, lw=0)
    ax2.fill_between(rel.index, 1.0, rel, where=rel < 1, color=COLOURS[PAIR[1]], alpha=0.15, lw=0)
    ax2.set_ylabel("12 檔 ÷ 20 檔")
    for a in (ax, ax2):
        a.grid(color=GRID, lw=0.6)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.suptitle("每週 12 檔對 20 檔：扣成本後的權益（下：兩者的比值，高於 1 是 12 檔領先）",
                 x=0.08, ha="left", fontsize=12)
    fig.savefig(path)
    plt.close(fig)


def settings_figure(d: pd.DataFrame, path) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.14)
    sizes = sorted(d["top_n"].unique())
    rng = np.random.default_rng(1)
    for k, n in enumerate(sizes):
        v = d.loc[d["top_n"] == n, "sharpe"].to_numpy()
        x = k + rng.uniform(-0.18, 0.18, len(v))
        ax.scatter(x, v, s=12, color=COLOURS["w12"] if n == 12 else INK2, alpha=0.55, lw=0)
        ax.plot([k - 0.28, k + 0.28], [np.median(v)] * 2, color=INK, lw=2)
    ax.set_xticks(range(len(sizes)), [f"{n} 檔" for n in sizes])
    ax.set_ylabel("Sharpe")
    ax.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("每週換股：50 種續抱 × 平滑 × 檢查時點的 Sharpe（橫線是中位數）", fontsize=11, loc="left")
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _p(v, d=1, sign=True):
    return "—" if v is None or not np.isfinite(v) else (f"{v * 100:+.{d}f}%" if sign else f"{v * 100:.{d}f}%")


def build(log=print) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    grid, rets = R.load("grid")
    small, _ = R.load(SWEEP)
    n20, _ = R.load("settings_n20")
    sett = pd.concat([small, n20[n20["schedule"] == "weekly"]], ignore_index=True)
    a, b = (X.load_curve(e)["ret"] for e in PAIR)
    idx = a.index.intersection(b.index)
    a, b = a[idx], b[idx]
    S = {e: X.load_summary(e) for e in PAIR}
    boot = bootstrap(a, b)
    wk = {e: workload(e) for e in PAIR}
    slip = D.empirical_slip()
    acc = pd.DataFrame([r for e in PAIR for r in accounts(e, slip)])
    acc.to_csv(OUT / "accounts.csv", index=False)
    equity_figure(OUT / "equity.png")
    settings_figure(sett[sett["top_n"].isin([10, 12, 15, 20])], OUT / "settings.png")

    L = ["# 要持有幾檔：每週 12 檔對 20 檔", "",
         "由 `twrank size` 與 `twrank size-report` 產生。2006-01 ～ 2026-10、扣成本、同一套樣本外分數；"
         "沒列出的設定都和基本款相同（續抱 4 × 檔數、4 週平滑、跌破 120 日 EMA 降半倉）。", "",
         "換股頻率的掃描裡，每週 12 ～ 20 檔是 Sharpe 差不多的高原，第一版上線選了最高點的 20 檔。"
         "人工下單時每一檔每週都要處理，這裡比較高原較少的一端（12 檔）會不會吃虧。", ""]

    # --- full period across sizes
    L += ["## 全期（每週換股，各種檔數）", "",
          "| 檔數 | 年化報酬 | Sharpe | 最大回檔 | 年化波動 | 年週轉 | 年成本 |", "|---|---|---|---|---|---|---|"]
    for n in (8, 10, 12, 15, 20, 25, 30):
        g = grid[grid["key"] == R._key(X.BASE.with_(top_n=n))].iloc[0]
        bold = "**" if n in (12, 20) else ""
        L.append(f"| {bold}{n}{bold} | {_p(g.cagr)} | {g.sharpe:.2f} | {_p(g.mdd, sign=False)} | "
                 f"{_p(g.vol, sign=False)} | {g.turnover:.1f}× | {_p(g.cost, 2, False)} |")
    L += ["", "![每週 12 檔對 20 檔](equity.png)", ""]

    # --- the gap is noise
    corr = float(a.corr(b))
    te = float((a - b).std() * np.sqrt(252))
    L += ["## 差距在雜訊範圍內", "",
          f"* 兩者的日報酬相關 {corr:.3f}，報酬差的年化波動（追蹤誤差）{_p(te, sign=False)}："
          "同一套分數、前 12 名本來就在前 20 名裡，差別只在多持有的 8 檔",
          f"* 區塊拔靴法（{BLOCK} 個交易日一塊、{REPS:,} 次）：12 檔減 20 檔的 Sharpe 差 "
          f"{boot['sharpe']:+.2f}（90% 區間 {boot['sharpe_90'][0]:+.2f} ～ {boot['sharpe_90'][1]:+.2f}），"
          f"12 檔比較高的機率 {boot['p_sharpe']:.0%}；對數報酬差每年 {_p(boot['logret'])}"
          f"（90% 區間 {_p(boot['logret_90'][0])} ～ {_p(boot['logret_90'][1])}），12 檔比較高的機率 {boot['p_logret']:.0%}", ""]

    # --- stages and years
    L += ["**十段 Walk-Forward**", "", "| 段 | 期間 | 12 檔 年化 | 20 檔 年化 | 12 檔 Sharpe | 20 檔 Sharpe | 12 檔 回檔 | 20 檔 回檔 |",
          "|---|---|---|---|---|---|---|---|"]
    w_c = w_s = 0
    for k, (x, y) in enumerate(zip(S[PAIR[0]]["stages"], S[PAIR[1]]["stages"], strict=True)):
        sx, sy = x["strategy"], y["strategy"]
        w_c += sx["cagr"] > sy["cagr"]
        w_s += sx["sharpe"] > sy["sharpe"]
        L.append(f"| {k + 1} | {str(sx['start'])[:7]} ～ {str(sx['end'])[:7]} | {_p(sx['cagr'])} | {_p(sy['cagr'])} | "
                 f"{sx['sharpe']:.2f} | {sy['sharpe']:.2f} | {_p(sx['mdd'], sign=False)} | {_p(sy['mdd'], sign=False)} |")
    ya = metrics.yearly(a)
    yb = metrics.yearly(b)
    diff = (ya - yb).dropna()
    L += ["", f"12 檔有 {w_c} 段年化報酬比較高、{w_s} 段 Sharpe 比較高。", "",
          f"**逐年**：{len(diff)} 年裡 12 檔有 {int((diff > 0).sum())} 年贏。差距常常到好幾個百分點，"
          f"兩個方向都有（最好 {_p(diff.max())}，{diff.idxmax()} 年；最差 {_p(diff.min())}，{diff.idxmin()} 年）。", "",
          "| 年 | 12 檔 | 20 檔 | 差距（百分點） |", "|---|---|---|---|"]
    for y in diff.index:
        L.append(f"| {y} | {_p(ya[y])} | {_p(yb[y])} | {diff[y] * 100:+.1f} |")
    L.append("")

    # --- other settings
    k = ["band_mult", "smooth_weeks", "regime_check"]
    m = sett[sett["top_n"] == 12].merge(sett[sett["top_n"] == 20], on=k, suffixes=("12", "20"))
    med = sett.groupby("top_n")[["sharpe", "cagr", "mdd"]].median()
    L += ["## 換成其他續抱、平滑設定也一樣", "",
          "續抱倍數 2／3／4／6／8 × 分數平滑 1／2／4／8／13 週 × 降半倉只在換股日或每天檢查，共 50 種，"
          "每一種都拿 12 檔和 20 檔配對比較：", "",
          f"* Sharpe：12 檔贏 {int((m.sharpe12 > m.sharpe20).sum())} / {len(m)} 種，差距中位數 {(m.sharpe12 - m.sharpe20).median():+.3f}",
          f"* 年化報酬：12 檔贏 {int((m.cagr12 > m.cagr20).sum())} / {len(m)} 種，差距中位數 {(m.cagr12 - m.cagr20).median() * 100:+.1f} 個百分點",
          f"* 最大回檔：12 檔比較淺的有 {int((m.mdd12 > m.mdd20).sum())} / {len(m)} 種，差距中位數 {(m.mdd12 - m.mdd20).median() * 100:+.1f} 個百分點", "",
          "| 檔數 | Sharpe 中位數 | 年化報酬中位數 | 最大回檔中位數 |", "|---|---|---|---|"]
    for n, r in med.iterrows():
        L.append(f"| {n} | {r.sharpe:.2f} | {_p(r.cagr)} | {_p(r.mdd, sign=False)} |")
    L += ["", "![設定掃描](settings.png)", ""]

    # --- workload and friction
    a0, a1 = (acc[acc["exp"] == e].set_index("capital") for e in PAIR)
    L += ["## 下單的工作量", "",
          "從換股紀錄算（2006 ～ 2026）：", "",
          "|  | 12 檔 | 20 檔 |", "|---|---|---|",
          f"| 每週新買進 | {wk[PAIR[0]]['adds']:.1f} 檔 | {wk[PAIR[1]]['adds']:.1f} 檔 |",
          f"| 每週賣出 | {wk[PAIR[0]]['drops']:.1f} 檔 | {wk[PAIR[1]]['drops']:.1f} 檔 |",
          f"| 有換名單的週 | {wk[PAIR[0]]['change_weeks']:.0%} | {wk[PAIR[1]]['change_weeks']:.0%} |",
          f"| 每年買賣一輪 | {wk[PAIR[0]]['trips_per_year']:.0f} 次 | {wk[PAIR[1]]['trips_per_year']:.0f} 次 |",
          f"| 持有週數（中位數／平均） | {wk[PAIR[0]]['hold_median']:.0f}／{wk[PAIR[0]]['hold_mean']:.1f} | "
          f"{wk[PAIR[1]]['hold_median']:.0f}／{wk[PAIR[1]]['hold_mean']:.1f} |", "",
          "但名單一變，其他持股也要調回平均權重，所以真正要下單的檔數多很多。"
          f"整數股帳戶（`twrank plan` 的預設：方案 B、手續費 6 折），{D.ODD_ERA} 年以後：", "",
          "| 資金 | 12 檔：每年下單日 | 每次下單幾檔 | 每年下單檔次 | 20 檔：每年下單日 | 每次下單幾檔 | 每年下單檔次 |",
          "|---|---|---|---|---|---|---|"]
    for cap in CAPITALS:
        x, y = a0.loc[cap], a1.loc[cap]
        L.append(f"| {cap / 1e4:.0f} 萬 | {x.order_days:.0f} | {x.stocks_per_order_day:.1f} | {x.stocks_per_year:.0f} | "
                 f"{y.order_days:.0f} | {y.stocks_per_order_day:.1f} | {y.stocks_per_year:.0f} |")
    L += ["", f"12 檔每年少下 {1 - a0.stocks_per_year.mean() / a1.stocks_per_year.mean():.0%} 的單。", "",
          f"**整數股的摩擦**：落後無摩擦回測的百分點／年（{D.ODD_ERA} 年以後，零股滑價 {slip:.2%}）。"
          "每檔的金額比較大，整數股的零頭與最低手續費占比變小，小資金的摩擦反而比較少：", "",
          "| 資金 | 12 檔 | 20 檔 |", "|---|---|---|"]
    for cap in CAPITALS:
        L.append(f"| {cap / 1e4:.0f} 萬 | {round(a0.loc[cap].gap * 100, 1) + 0:.1f} | {round(a1.loc[cap].gap * 100, 1) + 0:.1f} |")
    x, y = a0.loc[CAPITALS[-1]], a1.loc[CAPITALS[-1]]
    L += ["", "**交割**：代價在這裡。開盤跌停或停牌賣不掉、新股照買時，T+2 會短少那檔的市值；"
          f"12 檔一檔約淨值的 {1 / 12:.0%}，20 檔是 {1 / 20:.0%}。"
          f"{CAPITALS[-1] / 1e4:.0f} 萬、2006 年起（短少占淨值，下單日的 99% 分位／最大）："
          f"12 檔 {_p(x.shortfall_99, sign=False)}／{_p(x.shortfall_max, sign=False)}（{x.blocked_days} 個下單日有賣不掉的股票），"
          f"20 檔 {_p(y.shortfall_99, sign=False)}／{_p(y.shortfall_max, sign=False)}（{y.blocked_days} 個）。"
          "最大的一次是兩檔同時賣不掉。12 檔的交割戶要多留淨值的 9% 左右（約一檔），20 檔是 5%。", ""]

    sx, sy = S[PAIR[0]]["full"]["strategy"], S[PAIR[1]]["full"]["strategy"]
    L += ["## 結論：上線改用每週 12 檔", "",
          f"* 報酬與風險分不出高下：Sharpe {sx['sharpe']:.2f} 對 {sy['sharpe']:.2f}、年化 {_p(sx['cagr'])} 對 {_p(sy['cagr'])}、"
          f"最大回檔 {_p(sx['mdd'], sign=False)} 對 {_p(sy['mdd'], sign=False)}；拔靴法、十段、逐年、50 種設定都看不出哪一邊比較好",
          f"* 下單少四成左右，小資金的整數股摩擦也比較小",
          "* 再少就開始掉：每週 10 檔以下 Sharpe 往下（8 檔 1.41、7 檔 1.35），個股風險也更集中，12 檔是高原較少的一端",
          "* 要承擔的是：逐年和 20 檔可能差好幾個百分點（兩個方向都有）；一檔停牌或跌停賣不掉時卡住的錢比較多，交割戶多留淨值的 9%", ""]
    (OUT / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
    out = {"bootstrap": boot, "workload": wk, "accounts": acc.to_dict("records"),
           "paired": {"n": len(m), "sharpe_wins": int((m.sharpe12 > m.sharpe20).sum()),
                      "cagr_wins": int((m.cagr12 > m.cagr20).sum()),
                      "mdd_shallower": int((m.mdd12 > m.mdd20).sum())}}
    (OUT / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float),
                                      encoding="utf-8")
    log(f"wrote {OUT / 'SUMMARY.md'}")
    return out
