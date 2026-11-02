"""Figures, tests and ``results/rebalance/SUMMARY.md`` from the rebalance sweeps.

Reads only what :mod:`twrank.rebalance` saved; never re-runs a backtest.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import experiments as X
from . import metrics
from .rebalance import DAYS, OUT, SIZES, load
from .stacked import GRID, INK, INK2, _mpl

PLATEAU = 0.9  # a size is "usable" while its Sharpe stays within 90% of the plateau


def _label(row) -> str:
    return "每週" if row["schedule"] == "weekly" else f"每月 {int(row['day'])} 日"


def matrix(grid: pd.DataFrame, value: str = "sharpe") -> pd.DataFrame:
    g = grid.copy()
    g["row"] = np.where(g["schedule"] == "weekly", 0, g["day"])
    return g.pivot_table(index="row", columns="top_n", values=value).sort_index()


def smooth_days(m: pd.DataFrame, half: int = 2) -> pd.DataFrame:
    """Circular moving average over neighbouring days of the month (weekly row untouched)."""
    monthly = m.loc[DAYS]
    arr = monthly.to_numpy()
    k = np.arange(-half, half + 1)
    sm = np.mean([np.roll(arr, s, axis=0) for s in k], axis=0)
    out = m.copy()
    out.loc[DAYS] = sm
    return out


def kendall_w(rets: pd.DataFrame, keys: list[str]) -> tuple[float, float]:
    """Agreement of the days' ranking across calendar years (Kendall's W, chi-square p)."""
    from scipy.stats import chi2

    yearly = (1 + rets[keys]).groupby(rets.index.year).prod() - 1
    yearly = yearly.loc[[y for y in yearly.index if 2006 <= y <= 2025]]
    ranks = yearly.rank(axis=1)
    m, n = ranks.shape  # m judges (years), n objects (days)
    s = ((ranks.sum(axis=0) - m * (n + 1) / 2) ** 2).sum()
    w = 12 * s / (m**2 * (n**3 - n))
    stat = m * (n - 1) * w
    return float(w), float(chi2.sf(stat, n - 1))


def heatmap(m: pd.DataFrame, path, title: str, fmt="{:.2f}", mark=None, cmap="viridis") -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8.6, 9.5))
    fig.subplots_adjust(left=0.12, right=0.92, top=0.92, bottom=0.07)
    data = m.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(m.columns)), [str(c) for c in m.columns])
    ylabels = ["每週" if r == 0 else f"{r} 日" for r in m.index]
    ax.set_yticks(range(len(m.index)), ylabels)
    ax.set_xlabel("持股檔數")
    lo, hi = np.nanmin(data), np.nanmax(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                    color="white" if (v - lo) / (hi - lo + 1e-12) < 0.55 else INK)
    if mark is not None:
        i, j = mark
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#e8483b", lw=2.2))
    ax.axhline(0.5, color="white", lw=2)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    ax.set_title(title, loc="left", fontsize=12)
    fig.savefig(path)
    plt.close(fig)


def size_profile(grid: pd.DataFrame, path) -> pd.DataFrame:
    """Sharpe / CAGR / MDD against book size: weekly vs monthly (median and 10-90% over days)."""
    plt = _mpl()
    rows = []
    for n in SIZES:
        w = grid[(grid.schedule == "weekly") & (grid.top_n == n)].iloc[0]
        mo = grid[(grid.schedule == "monthly") & (grid.top_n == n)]
        rows.append({"top_n": n, "w_sharpe": w.sharpe, "w_cagr": w.cagr, "w_mdd": w.mdd,
                     "m_sharpe": mo.sharpe.median(), "m_sharpe_lo": mo.sharpe.quantile(0.1),
                     "m_sharpe_hi": mo.sharpe.quantile(0.9), "m_cagr": mo.cagr.median(),
                     "m_mdd": mo.mdd.median(), "m_mdd_lo": mo.mdd.quantile(0.1),
                     "m_mdd_hi": mo.mdd.quantile(0.9), "w_turnover": w.turnover,
                     "m_turnover": mo.turnover.median(), "w_cost": w.cost, "m_cost": mo.cost.median()})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.85, bottom=0.13, wspace=0.25)
    for ax, key, lab, pctfmt in ((axes[0], "sharpe", "Sharpe", False), (axes[1], "cagr", "年化報酬", True),
                                 (axes[2], "mdd", "最大回檔", True)):
        ax.plot(df.top_n, df[f"w_{key}"], color="#2f6db5", marker="o", ms=4, lw=1.8, label="每週")
        ax.plot(df.top_n, df[f"m_{key}"], color="#e07b39", marker="o", ms=4, lw=1.8,
                label="每月（28 個換股日的中位數）")
        if f"m_{key}_lo" in df:
            ax.fill_between(df.top_n, df[f"m_{key}_lo"], df[f"m_{key}_hi"], color="#e07b39",
                            alpha=0.18, lw=0, label="每月 10%～90%")
        ax.set_title(lab, loc="left", fontsize=11)
        ax.set_xlabel("持股檔數")
        ax.set_xticks(SIZES)
        ax.grid(color=GRID, lw=0.6)
        if pctfmt:
            ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("持股檔數與表現（2006-01 ～ 2026-10，扣成本）", x=0.05, ha="left", fontsize=13)
    fig.savefig(path)
    plt.close(fig)
    return df


def day_profile(grid: pd.DataFrame, n: int, path) -> None:
    plt = _mpl()
    sub = grid[(grid.schedule == "monthly")].pivot_table(index="day", columns="top_n", values="sharpe")
    wk = grid[grid.schedule == "weekly"].set_index("top_n")["sharpe"]
    fig, ax = plt.subplots(figsize=(12, 4.4))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.13)
    for c in sub.columns:
        ax.plot(sub.index, sub[c], color="#c9c7bf", lw=0.8)
    ax.plot(sub.index, sub[n], color="#e07b39", lw=2.2, marker="o", ms=4, label=f"每月換股・{n} 檔")
    sm = np.mean([np.roll(sub[n].to_numpy(), s) for s in (-2, -1, 0, 1, 2)], axis=0)
    ax.plot(sub.index, sm, color=INK, lw=1.2, ls=(0, (4, 2)), label="同上，前後 2 日平均")
    ax.axhline(wk[n], color="#2f6db5", lw=1.6, label=f"每週換股・{n} 檔")
    ax.axvspan(0.5, 10.5, color="#2f6db5", alpha=0.05)
    ax.text(1, ax.get_ylim()[1], "營收公布期（10 日以前）", va="top", fontsize=9, color=INK2)
    ax.set_xticks(DAYS)
    ax.set_xlabel("每月換股日（遇休市順延）")
    ax.set_ylabel("Sharpe")
    ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower right")
    ax.set_title("每月幾日換股：灰線是其他檔數", loc="left", fontsize=12)
    fig.savefig(path)
    plt.close(fig)


def stage_compare(keys: dict[str, str], rets: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Same walk-forward stages side by side for a few configurations."""
    rows = []
    g = grid.set_index("key")
    for label, key in keys.items():
        r = rets[key].astype("float64")
        for i in range(len(X.STAGES)):
            a, b = X.stage_bounds(i)
            s = metrics.summary(r[(r.index >= a) & (r.index <= b)])
            rows.append({"config": label, "stage": i + 1, "cagr": s["cagr"], "sharpe": s["sharpe"],
                         "mdd": s["mdd"], "turnover": g.loc[key, f"s{i + 1}_turnover"]})
        s = metrics.summary(r)
        rows.append({"config": label, "stage": "全期", "cagr": s["cagr"], "sharpe": s["sharpe"],
                     "mdd": s["mdd"], "turnover": g.loc[key, "turnover"], "cost": g.loc[key, "cost"]})
    return pd.DataFrame(rows)


def _pct(v, d=1, sign=True):
    if v is None or pd.isna(v):
        return "—"
    return f"{v * 100:+.{d}f}%" if sign else f"{v * 100:.{d}f}%"


def _wf_label(p: str) -> str:
    n = p.split("_n")[1]
    return f"每週 {n} 檔" if p.startswith("W") else f"{int(p[1:3])} 日 {n} 檔"


def build(log=print) -> dict:
    from .rebalance import REC_DAY, REC_SIZE, _key, wf_choice

    grid, rets = load("grid")
    rets = rets.astype("float64")
    settings, _ = load("settings_n20")
    sh = matrix(grid, "sharpe")
    sm = smooth_days(sh)

    prof = size_profile(grid, OUT / "sizes.png")
    plat_m, plat_w = prof["m_sharpe"].max(), prof["w_sharpe"].max()
    ok_m = int(prof.loc[prof["m_sharpe"] >= PLATEAU * plat_m, "top_n"].min())
    ok_w = int(prof.loc[prof["w_sharpe"] >= PLATEAU * plat_w, "top_n"].min())
    day = pick_day(sm, [REC_SIZE])
    if day != REC_DAY:
        raise RuntimeError(f"the sweep now prefers day {day}; update rebalance.REC_DAY")
    mark = (list(sh.index).index(day), list(sh.columns).index(REC_SIZE))
    heatmap(sh, OUT / "heatmap.png", "Sharpe：換股日 × 持股檔數（2006-01～2026-10，扣成本）", mark=mark)
    heatmap(sm, OUT / "heatmap_smoothed.png", "Sharpe，每月換股日取前後 2 日平均（紅框 = 建議）",
            mark=mark)
    heatmap(matrix(grid, "mdd"), OUT / "heatmap_mdd.png", "最大回檔：換股日 × 持股檔數",
            fmt="{:.0%}", mark=mark)
    day_profile(grid, REC_SIZE, OUT / "days.png")
    mcols = [c for c in rets.columns if c.startswith("M")]
    w20, p20 = kendall_w(rets, [c for c in mcols if f"_n{REC_SIZE}_" in c])
    w30, p30 = kendall_w(rets, [c for c in mcols if "_n30_" in c])
    evl = early_vs_late(grid, [15, 20, 25])

    keys = {
        "每週 30 檔（基本款）": _key(X.BASE),
        f"每週 {REC_SIZE} 檔": _key(X.BASE.with_(top_n=REC_SIZE)),
        f"每月 {day} 日 {REC_SIZE} 檔": _key(X.BASE.with_(schedule="monthly", day=day, top_n=REC_SIZE)),
        f"每月 {day} 日 30 檔": _key(X.BASE.with_(schedule="monthly", day=day)),
    }
    comp = stage_compare(keys, rets, grid)
    comp.to_csv(OUT / "stages.csv", index=False)
    spread = []
    for i in range(len(X.STAGES)):
        a, b = X.stage_bounds(i)
        r = rets.loc[(rets.index >= a) & (rets.index <= b), [c for c in mcols if f"_n{REC_SIZE}_" in c]]
        s = r.mean() / r.std() * np.sqrt(252)
        spread.append({"stage": i + 1, "min": s.min(), "median": s.median(), "max": s.max()})
    spread = pd.DataFrame(spread)
    wf = wf_choice("grid", log=lambda m: None)

    L = ["# 每週換股還是每月換股：結果", "",
         "由 `twrank rebalance-report` 產生。需求見 [genesis/FIAT-rebalance.md](../../genesis/FIAT-rebalance.md)。"
         "所有數字都是 2006-01 ～ 2026-10、扣成本、同一套樣本外分數；沒列出的設定都和基本款相同。", ""]
    L += ["## 換股日 × 持股檔數", "", "![Sharpe 熱力圖（平滑）](heatmap_smoothed.png)", "",
          "未平滑的熱力圖：[heatmap.png](heatmap.png)；最大回檔：[heatmap_mdd.png](heatmap_mdd.png)。", "",
          f"* 前後 2 日平滑後，{REC_SIZE} 檔時最好的換股日是 **{day} 日**，9～15 日都差不多",
          f"* 但哪一天好**逐年不一致**：28 個換股日的逐年報酬排名，Kendall's W = {w20:.3f}"
          f"（p = {p20:.2f}，{REC_SIZE} 檔）、{w30:.3f}（p = {p30:.2f}，30 檔），和隨機排名分不出來",
          f"* 營收公布期（每月 10 日以前）換股和之後換股沒有差別：平均 Sharpe {evl['early_mean']:.2f} 對 "
          f"{evl['late_mean']:.2f}（Mann-Whitney p = {evl['p']:.2f}，15～25 檔）", "",
          "![換股日](days.png)", ""]
    L += ["## 持股檔數", "", "![檔數](sizes.png)", ""]
    L.append(_md_table(prof, {
        "top_n": ("檔數", "{:.0f}"), "w_sharpe": ("每週 Sharpe", "{:.2f}"),
        "m_sharpe": ("每月 Sharpe 中位數", "{:.2f}"), "m_sharpe_lo": ("每月 10%", "{:.2f}"),
        "m_sharpe_hi": ("每月 90%", "{:.2f}"), "w_mdd": ("每週回檔", lambda v: _pct(v, 0, False)),
        "m_mdd": ("每月回檔中位數", lambda v: _pct(v, 0, False)),
        "w_cost": ("每週年成本", lambda v: _pct(v, 1, False)),
        "m_cost": ("每月年成本", lambda v: _pct(v, 1, False))}))
    L += ["", f"以「Sharpe 不低於高原的 {PLATEAU:.0%}」當作還能用：每週 {ok_w} 檔以上、每月（中位數）{ok_m} 檔以上。", ""]

    L += [f"## 跟著換股頻率一起看的設定（{REC_SIZE} 檔）", ""]
    m = settings[settings.schedule == "monthly"]
    w = settings[settings.schedule == "weekly"]
    pm = m[m.regime_check == "trade"].pivot_table(index="band_mult", columns="smooth_weeks",
                                                  values="sharpe", aggfunc="median")
    pw = w[w.regime_check == "trade"].pivot_table(index="band_mult", columns="smooth_weeks", values="sharpe")
    for title, piv in (("每月（28 個換股日的中位數）", pm), ("每週", pw)):
        L += [f"Sharpe，{title}。列：續抱倍數（還在前 N × 倍數名內就續抱）；欄：分數平滑（週）", "",
              "| 續抱＼平滑 | " + " | ".join(f"{c:g}" for c in piv.columns) + " |",
              "|---|" + "---|" * len(piv.columns)]
        L += [f"| {b:g} | " + " | ".join(f"{v:.2f}" for v in r) + " |" for b, r in piv.iterrows()]
        L.append("")
    chk = settings.groupby(["schedule", "regime_check"])[["sharpe", "cagr", "mdd", "turnover"]].median()
    names = {"trade": "只在換股日", "weekly": "每週第一個交易日", "daily": "每天"}
    L += ["降半倉的檢查時點（所有續抱 × 平滑組合的中位數）：", "",
          "| 換股 | 檢查時點 | Sharpe | 年化報酬 | 最大回檔 | 年週轉 |", "|---|---|---|---|---|---|"]
    for (sched, c), r in chk.iterrows():
        L.append(f"| {'每週' if sched == 'weekly' else '每月'} | {names[c]} | {r.sharpe:.2f} | "
                 f"{_pct(r.cagr)} | {_pct(r.mdd, 0, False)} | {r.turnover:.1f}× |")
    L.append("")

    L += ["## 同樣的 Walk-Forward 段落並排", ""]
    for metric, lab, f in (("cagr", "年化報酬", _pct), ("sharpe", "Sharpe", lambda v: f"{v:.2f}"),
                           ("mdd", "最大回檔", lambda v: _pct(v, 1, False)),
                           ("turnover", "年週轉（買＋賣）", lambda v: f"{v:.1f}×")):
        piv = comp.pivot_table(index="stage", columns="config", values=metric, aggfunc="first",
                               sort=False)[list(keys)]
        L += [f"**{lab}**", "", "| 段 | " + " | ".join(piv.columns) + " |", "|---|" + "---|" * len(piv.columns)]
        L += [f"| {stg} | " + " | ".join(f(v) for v in r) + " |" for stg, r in piv.iterrows()]
        L.append("")
    cost = comp[comp.stage == "全期"].set_index("config")["cost"]
    L += ["全期年化交易成本：" + "、".join(f"{k} {_pct(v, 2, False)}" for k, v in cost.items()), "",
          f"每月 {REC_SIZE} 檔時，28 個換股日在各段的 Sharpe 範圍（換股日的運氣有多大）：", "",
          "| 段 | 最低 | 中位數 | 最高 |", "|---|---|---|---|"]
    L += [f"| {int(r.stage)} | {r['min']:.2f} | {r['median']:.2f} | {r['max']:.2f} |" for _, r in spread.iterrows()]
    L += ["", "## 誠實版：每段只用訓練期挑換股方式與檔數", "",
          "每段用 2004 年起到該段開始的資料，在每月（28 個換股日 × 10 種檔數）或每週（10 種檔數）裡挑 Sharpe 最高的，"
          "原封不動用在測試期：", "", "| 段 | 每月挑到 | 測試期 Sharpe | 每週挑到 | 測試期 Sharpe |",
          "|---|---|---|---|---|"]
    for stg in sorted(wf.stage.unique()):
        mm = wf[(wf.stage == stg) & (wf.family == "monthly")].iloc[0]
        ww = wf[(wf.stage == stg) & (wf.family == "weekly")].iloc[0]
        L.append(f"| {stg} | {_wf_label(mm.pick)} | {mm.test_sharpe:.2f} | {_wf_label(ww.pick)} | "
                 f"{ww.test_sharpe:.2f} |")
    mean_m = float(wf[wf.family == "monthly"].test_sharpe.mean())
    mean_w = float(wf[wf.family == "weekly"].test_sharpe.mean())
    L += [f"| 平均 | | **{mean_m:.2f}** | | **{mean_w:.2f}** |", ""]
    (OUT / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")

    out = {"day": day, "size": REC_SIZE, "min_usable": {"weekly": ok_w, "monthly": ok_m},
           "plateau": {"weekly": float(plat_w), "monthly": float(plat_m)},
           "kendall": {"w20": w20, "p20": p20, "w30": w30, "p30": p30}, "early_vs_late": evl,
           "wf_mean": {"monthly": mean_m, "weekly": mean_w},
           "size_profile": prof.round(4).to_dict("records")}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"rebalance report -> {OUT / 'SUMMARY.md'}")
    return out


def _md_table(df: pd.DataFrame, fmts: dict) -> str:
    cols = list(fmts)
    lines = ["| " + " | ".join(fmts[c][0] for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            f = fmts[c][1]
            cells.append(f(v) if callable(f) else (f.format(v) if pd.notna(v) else "—"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pick_day(sm: pd.DataFrame, sizes: list[int]) -> int:
    """Best day on the day-smoothed Sharpe map, averaged over the plateau sizes."""
    return int(sm.loc[DAYS, sizes].mean(axis=1).idxmax())


def early_vs_late(grid: pd.DataFrame, sizes: list[int]) -> dict:
    m = grid[(grid.schedule == "monthly") & grid.top_n.isin(sizes)]
    early = m[m.day <= 10].groupby("day").sharpe.mean()
    late = m[m.day > 10].groupby("day").sharpe.mean()
    from scipy.stats import mannwhitneyu

    p = mannwhitneyu(early, late, alternative="two-sided").pvalue
    return {"early_mean": float(early.mean()), "late_mean": float(late.mean()), "p": float(p)}
