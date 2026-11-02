"""Markdown tables and figures for the README (``results/report.md``, ``results/*.png``)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import experiments as X
from .config import RESULTS_DIR

#: Six walk-forward stages of the earlier LightGBM ranking study (same portfolio
#: rules: top 30, buffer 4 x 30, 4-week smoothing, half exposure below the
#: 120-day EMA).  Before 2019 its score also used fundamental and chip
#: features; from 2019 on it was the same price/volume 21-day + 63-day ensemble.
ORIGINAL = pd.DataFrame(
    {
        "stage": [5, 6, 7, 8, 9, 10],
        "period": ["2014-15", "2016-17", "2018-19", "2020-21", "2022-23", "2024-26/04"],
        "sharpe": [1.301, 2.539, 1.300, 2.098, 1.792, 1.462],
        "cagr": [0.1216, 0.2592, 0.1045, 0.3186, 0.1747, 0.2111],
        "mdd": [-0.1061, -0.0366, -0.0826, -0.1541, -0.0878, -0.1220],
        "turnover": [8.1, 7.2, 9.2, 9.5, 9.0, 7.0],
    }
)


def _p(v, d=1, sign=True):
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v * 100:+.{d}f}%" if sign else f"{v * 100:.{d}f}%"


def _n(v, d=2):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def full_table(ids: list[str]) -> str:
    lines = ["| 實驗 | 年化報酬 | Sharpe | 最大回檔 | 年化波動 | 年週轉 | 年成本 | 平均持股 | 平均曝險 |",
             "|---|---|---|---|---|---|---|---|---|"]
    bench = None
    for i in ids:
        s = X.load_summary(i)
        m = s["full"]["strategy"]
        lines.append(f"| {s['label']} | {_p(m['cagr'])} | {_n(m['sharpe'])} | {_p(m['mdd'], sign=False)} | "
                     f"{_p(m['vol'], sign=False)} | {_n(m['turnover'], 1)}× | {_p(m['cost'], 2, False)} | "
                     f"{_n(m['positions'], 1)} | {_p(m['exposure'], 0, False)} |")
        bench = s["full"]
    for k, lab in (("0050", "0050 含息"), ("TR", "加權報酬指數（含息）"), ("TAIEX", "加權指數（不含息）")):
        m = bench[k]
        lines.append(f"| {lab} | {_p(m['cagr'])} | {_n(m['sharpe'])} | {_p(m['mdd'], sign=False)} | "
                     f"{_p(m['vol'], sign=False)} | | | | |")
    s0 = X.load_summary(ids[0])["full"]["strategy"]
    return "\n".join(lines) + f"\n\n期間 {s0['start']} ～ {s0['end']}（{s0['years']:.1f} 年）。"


def stage_table(exp_id: str) -> str:
    s = X.load_summary(exp_id)
    lines = ["| 段 | 期間 | 年化報酬 | Sharpe | 最大回檔 | 年週轉 | 持股 | 0050 年化 | 0050 Sharpe | 0050 回檔 | 加權報酬指數年化 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for st in s["stages"]:
        a, b, t = st["strategy"], st["0050"], st["TR"]
        lines.append(f"| {st['stage']} | {a['start'][:7]} ～ {a['end'][:7]} | {_p(a['cagr'])} | {_n(a['sharpe'])} | "
                     f"{_p(a['mdd'], sign=False)} | {_n(a['turnover'], 1)}× | {_n(a.get('positions'), 1)} | "
                     f"{_p(b['cagr'])} | {_n(b['sharpe'])} | {_p(b['mdd'], sign=False)} | {_p(t['cagr'])} |")
    sh = [st["strategy"]["sharpe"] for st in s["stages"]]
    bs = [st["0050"]["sharpe"] for st in s["stages"]]
    lines.append(f"| 平均 | | | **{np.mean(sh):.2f}** | | | | | {np.mean(bs):.2f} | | |")
    return "\n".join(lines)


def stage_sharpe_matrix(ids: list[str]) -> str:
    sums = [X.load_summary(i) for i in ids]
    head = "| 段 | 期間 | " + " | ".join(s["label"] for s in sums) + " | 0050 |"
    lines = [head, "|---|---|" + "---|" * (len(ids) + 1)]
    for k, st in enumerate(sums[0]["stages"]):
        a = st["strategy"]
        row = [str(st["stage"]), f"{a['start'][:7]} ～ {a['end'][:7]}"]
        row += [_n(s["stages"][k]["strategy"]["sharpe"]) for s in sums]
        row.append(_n(st["0050"]["sharpe"]))
        lines.append("| " + " | ".join(row) + " |")
    avg = ["平均", ""] + [f"{np.mean([x['strategy']['sharpe'] for x in s['stages']]):.2f}" for s in sums]
    avg.append(f"{np.mean([x['0050']['sharpe'] for x in sums[0]['stages']]):.2f}")
    lines.append("| " + " | ".join(avg) + " |")
    return "\n".join(lines)


def yearly_table(ids: list[str]) -> str:
    sums = [X.load_summary(i) for i in ids]
    head = "| 年 | " + " | ".join(s["label"] for s in sums) + " | 0050 含息 | 加權報酬指數 | 加權指數 |"
    lines = [head, "|---|" + "---|" * (len(ids) + 3)]
    years = sorted(sums[0]["yearly"], key=int)
    for y in years:
        row = [str(y)] + [_p(s["yearly"][y]["strategy"]) for s in sums]
        row += [_p(sums[0]["yearly"][y][k]) for k in ("0050", "TR", "TAIEX")]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def original_table(exp_id: str = "base") -> str:
    s = X.load_summary(exp_id)
    by = {st["stage"]: st for st in s["stages"]}
    lines = ["| 段 | 期間 | 本 repo Sharpe | 先前研究 Sharpe | 本 repo 年化 | 先前研究年化 | 本 repo 回檔 | 先前研究回檔 | 本 repo 週轉 | 先前研究週轉 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in ORIGINAL.itertuples():
        a = by[r.stage]["strategy"]
        lines.append(f"| {r.stage} | {r.period} | {_n(a['sharpe'])} | {r.sharpe:.2f} | {_p(a['cagr'])} | {_p(r.cagr)} | "
                     f"{_p(a['mdd'], sign=False)} | {_p(r.mdd, sign=False)} | {_n(a['turnover'], 1)}× | {r.turnover:.1f}× |")
    ours = np.mean([by[k]["strategy"]["sharpe"] for k in ORIGINAL["stage"]])
    lines.append(f"| 平均 | | **{ours:.2f}** | **{ORIGINAL['sharpe'].mean():.2f}** | | | | | | |")
    return "\n".join(lines)


def ic_table() -> str:
    ic = pd.read_csv(X.RESULTS_DIR / "models" / "ic.csv", parse_dates=["cutoff"])
    ic["year"] = ic["cutoff"].dt.year
    g = ic.groupby((ic["year"] - 2004) // 4)
    lines = ["| 期間 | 21 日模型 IC | 63 日模型 IC | 重訓次數 |", "|---|---|---|---|"]
    for _, x in g:
        lines.append(f"| {x['year'].min()} ～ {x['year'].max()} | {x['ic21'].mean():.3f} | {x['ic63'].mean():.3f} | {len(x)} |")
    lines.append(f"| 全部 | **{ic['ic21'].mean():.3f}** | **{ic['ic63'].mean():.3f}** | {len(ic)} |")
    pos21, pos63 = (ic["ic21"] > 0).mean(), (ic["ic63"] > 0).mean()
    return "\n".join(lines) + f"\n\nIC 為正的重訓段：21 日 {pos21:.0%}、63 日 {pos63:.0%}。"


def survivorship() -> tuple[str, dict]:
    """Rough size of the survivorship bias from the TWSE delisting list."""
    from .data import load_panel
    from .universe import load_delisted, load_universe

    dl = load_delisted()
    uni = load_universe()
    twse = set(uni.loc[uni["market"] == "TWSE", "stock_id"])
    close = load_panel("close")
    alive = close[[c for c in close.columns if c in twse]].notna().groupby(close.index.year).any().sum(axis=1)
    per_year = dl.groupby(dl["date"].dt.year).size()
    rows = []
    for y in range(2006, int(close.index[-1].year) + 1):
        n_dl = int(per_year.get(y, 0))
        n_alive = int(alive.get(y, 0))
        rate = n_dl / (n_alive + n_dl) if n_alive else np.nan
        rows.append({"year": y, "delisted": n_dl, "listed_in_data": n_alive, "rate": rate})
    df = pd.DataFrame(rows)
    mean_rate = float(df["rate"].mean())
    lines = [f"證交所終止上市清單（四碼普通股）2006 年起共 {int(df['delisted'].sum())} 家，"
             f"平均每年約占當年上市家數的 {mean_rate:.1%}。"]
    out = {"per_year": df.round(4).to_dict("records"), "mean_rate": mean_rate}
    return "\n".join(lines), out


def equity_png(ids: list[str], path) -> None:
    from .stacked import GRID, INK2, _mpl

    plt = _mpl()
    from .data import load_bench, load_index
    from .metrics import bench_returns

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 6.6), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1]})
    fig.subplots_adjust(left=0.07, right=0.985, top=0.93, bottom=0.07, hspace=0.06)
    colours = ["#2f6db5", "#e07b39", "#7a5cc7", "#3a9e6a", "#c4466b", "#8a8a3a"]
    idx = None
    for k, i in enumerate(ids):
        c = X.load_curve(i)
        s = X.load_summary(i)
        nav = (1 + c["ret"]).cumprod()
        idx = c.index
        ax.plot(nav.index, nav, color=colours[k % len(colours)], lw=1.8 if k == 0 else 1.1,
                label=s["label"])
        dd = nav / nav.cummax() - 1
        ax2.plot(dd.index, dd, color=colours[k % len(colours)], lw=1.3 if k == 0 else 0.8)
    for lab, px, style in (("0050 含息", load_bench(), (0, (4, 3))),
                           ("加權指數（不含息）", load_index()["Close"], (0, (1, 2)))):
        nav = (1 + bench_returns(px, idx)).cumprod()
        ax.plot(nav.index, nav, color=INK2, lw=1.1, ls=style, label=lab)
        dd = nav / nav.cummax() - 1
        ax2.plot(dd.index, dd, color=INK2, lw=0.9, ls=style)
    ax.set_yscale("log")
    ax.set_ylabel("權益（起始 = 1，對數）")
    ax2.set_ylabel("回檔")
    ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    for a in (ax, ax2):
        a.grid(color=GRID, lw=0.6)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("扣成本後的權益曲線與回檔（2006 起，每段兩年 Walk-Forward 串接）", x=0.07, ha="left",
                 fontsize=13)
    fig.savefig(path)
    plt.close(fig)


def build(ids: list[str] | None = None, log=print) -> None:
    ids = ids or [e.id for e in X.viewer_experiments()]
    parts = ["# 結果表（由 `twrank report` 產生）", ""]
    parts += ["## 全期", "", full_table(ids), ""]
    parts += ["## 基本款各段", "", stage_table("base"), ""]
    parts += ["## 各實驗各段 Sharpe", "", stage_sharpe_matrix(ids), ""]
    parts += ["## 逐年報酬", "", yearly_table(ids), ""]
    parts += ["## 和先前研究六段對照", "", original_table("base"), ""]
    sel = RESULTS_DIR / "exp" / "wfv" / "summary.json"
    if sel.exists():
        s = json.loads(sel.read_text(encoding="utf-8"))
        rows = s.get("selection", [])
        parts += ["## 每段用訓練期挑參數", "", "| 段 | 測試期 | 續抱倍數 | 平滑（週） | 檔數 | 訓練期 Sharpe | 測試期 Sharpe |",
                  "|---|---|---|---|---|---|---|"]
        for r, st in zip(rows, s["stages"], strict=False):
            parts.append(f"| {r['stage']} | {r['test']} | {r['band_mult']:.0f} | {r['smooth_weeks']:.0f} | "
                         f"{r['top_n']} | {r['train_sharpe']:.2f} | {st['strategy']['sharpe']:.2f} |")
        parts.append("")
    parts += ["## 模型 IC（每次重訓後到下次重訓之間，週資料的橫斷面 rank IC）", "", ic_table(), ""]
    text, surv = survivorship()
    parts += ["## 倖存者偏誤", "", text, ""]
    (RESULTS_DIR / "survivorship.json").write_text(json.dumps(surv, indent=1), encoding="utf-8")
    (RESULTS_DIR / "report.md").write_text("\n".join(parts), encoding="utf-8")
    equity_png(ids, RESULTS_DIR / "equity.png")
    log(f"wrote {RESULTS_DIR / 'report.md'} and equity.png")
