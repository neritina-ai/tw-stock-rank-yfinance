"""``results/deploy/SUMMARY.md`` and figures from what :mod:`twrank.deploy` saved."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .deploy import CAPITALS, OUT, SLIPS
from .stacked import GRID, INK, INK2, _mpl

GAP_LIMIT = 0.01  # "close enough to the frictionless backtest": less than 1 percentage point
LABEL = {"w20": "每週 20 檔", "m11n20": "每月 11 日 20 檔"}
SCHEME = {"A": "方案 A（全部零股）", "B": "方案 B（整張＋零股）", "frac": "無摩擦"}
DISC = {1.0: "不打折", 0.6: "6 折", 0.28: "2.8 折"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots 1-4


def _wan(c: float) -> str:
    return f"{c / 1e4:,.0f} 萬"


def _pct(v, d=1, sign=False) -> str:
    if v is None or not np.isfinite(v):
        return "-"
    return f"{v * 100:+.{d}f}%" if sign else f"{v * 100:.{d}f}%"


def _pp(v) -> str:
    if not np.isfinite(v):
        return "-"
    return "0.0" if abs(v) < 0.0005 else f"{v * 100:.1f}"


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def threshold(g: pd.DataFrame, col: str) -> float:
    """Smallest capital from which every larger capital stays under the gap limit."""
    g = g.sort_values("capital")
    ok = g[col].to_numpy() < GAP_LIMIT
    for i in range(len(ok)):
        if ok[i:].all():
            return float(g["capital"].iloc[i])
    return float("nan")


def thresholds(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    g = grid[grid["scheme"] != "frac"]
    for (e, sch, d, sl), sub in g.groupby(["exp", "scheme", "discount", "slip"]):
        rows.append({"exp": e, "scheme": sch, "discount": d, "slip": sl,
                     "min_capital": threshold(sub, "gap"),
                     "min_capital_2021": threshold(sub, "gap_2021"),
                     "grid": "fine" if len(sub) > len(CAPITALS) else "main"})
    return pd.DataFrame(rows)


def gap_figure(grid: pd.DataFrame, emp: float, path) -> None:
    plt = _mpl()
    order = [emp] + [s for s in SLIPS if abs(s - emp) > 1e-9]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.86, top=0.88, bottom=0.08, hspace=0.25, wspace=0.08)
    g = grid[(grid["discount"] == 0.6) & (grid["scheme"] != "frac")]
    ymax = 0.0
    for i, e in enumerate(["w20", "m11n20"]):
        for j, sch in enumerate(["A", "B"]):
            ax = axes[i, j]
            for k, sl in enumerate(order):
                sub = g[(g["exp"] == e) & (g["scheme"] == sch) & (np.isclose(g["slip"], sl))]
                sub = sub.sort_values("capital")
                if sub.empty:
                    continue
                y = sub["gap_2021"] * 100
                ymax = max(ymax, float(y.max()))
                lab = f"實測 {sl:.2%}" if abs(sl - emp) < 1e-9 else f"{sl:.1%}"
                ax.plot(sub["capital"], y, color=SERIES[k], lw=2, marker="o", ms=4.5,
                        label=f"零股滑價 {lab}")
                if j == 0:  # flat, well separated lines: label them at the right end
                    ax.annotate(lab, (sub["capital"].iloc[-1], y.iloc[-1]), xytext=(6, 0),
                                textcoords="offset points", va="center", fontsize=9, color=INK2)
            ax.axhline(GAP_LIMIT * 100, color=INK, lw=1, ls=(0, (4, 3)))
            ax.set_xscale("log")
            ticks = [1e5, 3e5, 1e6, 3e6, 1e7, 2e7]
            ax.set_xticks(ticks)
            ax.set_xticklabels([_wan(t) for t in ticks])
            ax.grid(color=GRID, lw=0.6)
            ax.set_title(f"{LABEL[e]}・{SCHEME[sch]}", loc="left", fontsize=11)
            if j == 0:
                ax.set_ylabel("落後無摩擦回測（百分點／年）")
            if i == 1:
                ax.set_xlabel("資金（每年從現金開始）")
    for ax in axes.flat:
        ax.set_ylim(0, ymax * 1.05)
    axes[0, 1].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("資金規模與摩擦：整數股、最低手續費、零股滑價（2021-01 ～ 2026-10，手續費 6 折）",
                 x=0.07, ha="left", fontsize=13)
    fig.text(0.07, 0.915, "虛線：落後 1 個百分點。零股滑價是單邊，買進加、賣出減", ha="left",
             fontsize=10, color=INK2)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def settlement_table(st: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    s = st[~st["window_start"]].copy()
    rows = []
    worst = {}
    for (e, sch, c), g in s.groupby(["exp", "scheme", "capital"]):
        up = g[(g["prev_scale"] == 0.5) & (g["scale"] == 1.0)]
        dn = g[(g["prev_scale"] == 1.0) & (g["scale"] == 0.5)]
        normal = g[~g.index.isin(up.index) & ~g.index.isin(dn.index)]
        sf = g["shortfall_pct"]
        i = sf.idxmax()
        clean = g[g["blocked_sell"] == 0]["shortfall_pct"]
        rows.append({
            "exp": e, "scheme": sch, "capital": c,
            "pay_median": normal["pay_pct"].median(), "pay_p95": normal["pay_pct"].quantile(0.95),
            "pay_max": normal["pay_pct"].max(),
            "up_pay_max": up["pay_pct"].max() if len(up) else np.nan,
            "up_cash_min": up["cash_before_pct"].min() if len(up) else np.nan,
            "dn_pay_min": dn["pay_pct"].min() if len(dn) else np.nan,
            "short_days": int((sf > 1e-12).sum()), "days": int(len(g)),
            "short_p99": sf.quantile(0.99), "short_max": sf.max(),
            "short_max_clean": clean.max(), "short_p99_clean": clean.quantile(0.99),
            "blocked_days": int((g["blocked_sell"] > 0).sum()),
            "short_max_date": str(pd.Timestamp(g.loc[i, "date"]).date()),
        })
        worst[(e, sch, c)] = g.loc[i].to_dict()
    return pd.DataFrame(rows), worst


def build(log=print) -> dict:
    grid = pd.read_csv(OUT / "grid.csv")
    odd = json.loads((OUT / "oddlot.json").read_text(encoding="utf-8"))
    emp = float(odd["slip_estimate"])
    st = pd.read_csv(OUT / "settlement.csv", parse_dates=["date"])
    th = thresholds(grid)
    th.to_csv(OUT / "thresholds.csv", index=False)
    gap_figure(grid, emp, OUT / "gap.png")
    stl, worst = settlement_table(st)
    stl.to_csv(OUT / "settlement_summary.csv", index=False)
    md = render(grid, odd, emp, th, stl, worst)
    (OUT / "SUMMARY.md").write_text(md, encoding="utf-8")
    log(f"wrote {OUT / 'SUMMARY.md'}")
    return {"grid": grid, "odd": odd, "thresholds": th, "settlement": stl}


def _ref(grid, e, d):
    r = grid[(grid["exp"] == e) & (grid["scheme"] == "frac") & (grid["discount"] == d)].iloc[0]
    return r


def render(grid, odd, emp, th, stl, worst) -> str:
    out = ["# 上線前：資金、零股、成本與交割資金", "",
           "需求見 [genesis/FIAT-deploy.md](../../genesis/FIAT-deploy.md)。由 `twrank odd-lot`、"
           "`twrank deploy`、`twrank deploy-report` 產生。", ""]
    out += ["## 怎麼量", "",
            "* **每年從現金開始**：每年第一個換股日，帳戶以指定的資金從現金開始，買進策略當時的持股，"
            "一直做到隔年第一個換股日的前一天；各年的報酬再串起來算年化、Sharpe、最大回檔。"
            "這樣每個資金規模都維持在那個規模附近（一年內大約 0.6 ～ 1.6 倍），不會因為複利長大而看不出摩擦",
            "* **持股是整數股**：用前一天收盤價把目標權重換成股數（無條件捨去），隔天開盤成交，買不滿的錢留現金。"
            "方案 A 全部用零股（每筆 1 ～ 999 股）；方案 B 每筆委託裡 1,000 股的倍數用整張，餘數用零股",
            "* **成本**：手續費 0.1425% × 折扣，每筆最低整股 20 元、零股 1 元（假設，見 README），"
            "賣出證交稅 0.3%，都按每筆委託無條件捨去到元",
            "* **成交**：整張以開盤價成交；零股以開盤價加減滑價成交（買進加、賣出減）。開盤漲停買不到、開盤跌停賣不掉",
            "* **無摩擦回測**：同樣的每年重來，但可以買小數股、手續費按比例（同樣的折扣）、沒有最低手續費、沒有滑價。"
            "差距 = 無摩擦的年化報酬 − 整數股的年化報酬",
            "* 股價用實際成交的價格：Yahoo 把配股記成分割，舊的「原始」股價被往前調低（例如鴻海 2005 年初顯示 41 元，"
            "實際約 145 元），這裡用 Yahoo 的分割紀錄還原，整張的門檻才不會被低估",
            "* 盤中零股 2020-10-26 才開始，**下面主要看 2021 年以後**（2021-01 ～ 2026-10），全期（2006 ～ 2026）列在旁邊參考", ""]

    # --- odd-lot market
    f, v = odd["first"], odd["vwap"]
    out += ["## 盤中零股的成交價與量能", "",
            f"策略在 2020-10-26 ～ 2026-10 實際換股的日子，每一筆買進（新增）與賣出（剔除），"
            f"對上證交所與櫃買中心當天的盤中零股日行情與整股開盤價，共 {odd['with_quotes']:,} 筆"
            f"（另有 {odd['no_odd_trade']:,} 筆當天零股沒有成交）。"
            "「成本」是對策略不利的方向：買進時零股價高於整股開盤價、賣出時低於開盤價為正。", ""]
    rows = []
    for k, lab in (("buy", "買進"), ("sell", "賣出"), ("all", "合計")):
        rows.append({"": lab, "筆數": f"{f[k]['n']:,}",
                     "第一筆成交價：平均": _pct(f[k]["trimmed_mean"], 2, True),
                     "中位數": _pct(f[k]["median"], 2, True),
                     "10%～90% 分位": f"{_pct(f[k]['p10'], 2, True)} ～ {_pct(f[k]['p90'], 2, True)}",
                     "全日均價：平均": _pct(v[k]["trimmed_mean"], 2, True),
                     "中位數 ": _pct(v[k]["median"], 2, True)})
    out += [_md(pd.DataFrame(rows)), "",
            "（平均去掉兩端各 1% 的極端值。第一筆成交價是當天第一次撮合的價格，盤中零股 9:10 才第一次撮合。）", ""]
    rows = [{"撮合間隔": k, "筆數": f"{x['n']:,}", "第一筆成本：平均": _pct(x["trimmed_mean"], 2, True),
             "中位數": _pct(x["median"], 2, True)} for k, x in odd["by_regime"].items()]
    out += [_md(pd.DataFrame(rows)), "",
            f"**回測用的零股滑價取 {emp:.2%}**：現行每 5 秒撮合以來，第一筆成交價對整股開盤價的平均成本"
            "（買賣合計）。另外用單邊 0.1%、0.3%、0.5% 做敏感度。"
            f"零股收盤時的買賣價差中位數 {_pct(odd['spread']['median'], 2)}。", ""]
    cap = odd["capacity"]
    rows = []
    for c in CAPITALS:
        a, b = cap[f"A|{c:g}"], cap[f"B|{c:g}"]
        rows.append({"資金": _wan(c), "每檔約": f"{c / 20:,.0f}",
                     "A：占當天零股成交金額（中位數）": _pct(a["median_share"], 1),
                     "A：超過 25% 的比例": _pct(a["over_25pct"], 0),
                     "A：超過全天成交的比例": _pct(a["over_100pct"], 0),
                     "B：中位數": _pct(b["median_share"], 1),
                     "B：超過 25%": _pct(b["over_25pct"], 0)})
    out += ["量能：一檔新買進的單（資金 ÷ 20）占當天這檔股票盤中零股成交金額的比例。", "",
            _md(pd.DataFrame(rows)), ""]

    # --- main table
    for e in ("w20", "m11n20"):
        g = grid[(grid["exp"] == e)]
        out += [f"## {LABEL[e]}：資金 × 方案 × 手續費折扣", "",
                f"零股滑價 {emp:.2%}（實測）。年化報酬、Sharpe、最大回檔是 2021-01 ～ 2026-10；"
                "差距是落後無摩擦回測的百分點（每年），括號內是全期 2006 ～ 2026。", ""]
        rows = []
        for d in (1.0, 0.6, 0.28):
            r0 = _ref(grid, e, d)
            rows.append({"折扣": DISC[d], "資金": "無摩擦", "方案": "",
                         "年化": _pct(r0["cagr_2021"], 1), "Sharpe": f"{r0['sharpe_2021']:.2f}",
                         "最大回檔": _pct(r0["mdd_2021"], 1), "差距": "",
                         "全期年化": _pct(r0["cagr"], 1)})
            for c in CAPITALS:
                for sch in ("A", "B"):
                    r = g[(g["scheme"] == sch) & (g["discount"] == d) & (g["capital"] == c)
                          & np.isclose(g["slip"], emp)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    rows.append({"折扣": DISC[d], "資金": _wan(c), "方案": sch,
                                 "年化": _pct(r["cagr_2021"], 1), "Sharpe": f"{r['sharpe_2021']:.2f}",
                                 "最大回檔": _pct(r["mdd_2021"], 1),
                                 "差距": f"{_pp(r['gap_2021'])}（{_pp(r['gap'])}）",
                                 "全期年化": _pct(r["cagr"], 1)})
        out += [_md(pd.DataFrame(rows)), ""]

    # --- where the gap comes from
    dp = OUT / "decompose.csv"
    if dp.exists():
        dc = pd.read_csv(dp)
        ds = float(dc["slip"].iloc[0]) if "slip" in dc else emp
        note = "" if abs(ds - emp) < 1e-9 else f"（實測約 {emp:.2%}，這裡用 {ds:.1%} 示範滑價的份量）"
        out += ["## 差距從哪裡來（手續費 6 折）", "",
                "依序加上去：只有整數股（捨去的零頭留現金，手續費按比例、沒有最低）→ 加上最低手續費 → "
                f"加上零股滑價 {ds:.1%}{note}。百分點／年，2021 以後（括號內全期）。"
                "整數股這一步常是負的：手續費與證交稅按每筆無條件捨去到元，小單省下的零頭比捨去股數留下的現金拖累還多。", ""]
        rows = []
        for e in ("w20", "m11n20"):
            for sch in ("A", "B"):
                for c in CAPITALS:
                    a1 = dc[(dc["exp"] == e) & (dc["scheme"] == sch) & (dc["capital"] == c)]
                    if a1.empty:
                        continue
                    x = a1.set_index("period")
                    def cell(k, x=x):
                        return f"{_pp(x.loc['2021-', k])}（{_pp(x.loc['2006-', k])}）"
                    rows.append({"實驗": LABEL[e], "方案": sch, "資金": _wan(c),
                                 "整數股": cell("whole_shares"), "最低手續費": cell("min_fee"),
                                 f"零股滑價 {ds:.1%}": cell("slippage"), "合計": cell("total")})
        out += [_md(pd.DataFrame(rows)), ""]

    # --- slippage sensitivity
    out += ["## 零股滑價的敏感度（手續費 6 折）", "",
            "差距（百分點／年，2021 以後；括號內全期）。", ""]
    slips = sorted(grid["slip"].unique())
    rows = []
    for e in ("w20", "m11n20"):
        for sch in ("A", "B"):
            for c in CAPITALS:
                row = {"實驗": LABEL[e], "方案": sch, "資金": _wan(c)}
                for sl in slips:
                    r = grid[(grid["exp"] == e) & (grid["scheme"] == sch) & (grid["discount"] == 0.6)
                             & (grid["capital"] == c) & np.isclose(grid["slip"], sl)]
                    lab = f"{sl:.2%}" + ("（實測）" if abs(sl - emp) < 1e-9 else "")
                    row[lab] = (f"{_pp(r.iloc[0]['gap_2021'])}（{_pp(r.iloc[0]['gap'])}）"
                                if len(r) else "-")
                rows.append(row)
    out += [_md(pd.DataFrame(rows)), ""]

    # --- thresholds
    out += ["## 差距小於 1 個百分點的最低資金", "",
            "從這個資金以上（掃描過的每一個資金）差距都小於 1 個百分點。6 折用細的資金格點"
            "（10 萬 ～ 2,000 萬共 14 點），其他折扣用 6 個主要資金。「-」表示掃描範圍內都達不到。", ""]
    rows = []
    for e in ("w20", "m11n20"):
        for sch in ("A", "B"):
            for d in (1.0, 0.6, 0.28):
                row = {"實驗": LABEL[e], "方案": sch, "折扣": DISC[d]}
                for sl in slips:
                    r = th[(th["exp"] == e) & (th["scheme"] == sch) & (th["discount"] == d)
                           & np.isclose(th["slip"], sl)]
                    lab = f"{sl:.2%}" + ("（實測）" if abs(sl - emp) < 1e-9 else "")
                    if r.empty:
                        row[lab] = "-"
                        continue
                    a, b = r.iloc[0]["min_capital_2021"], r.iloc[0]["min_capital"]
                    row[lab] = f"{_wan(a) if np.isfinite(a) else '-'}（{_wan(b) if np.isfinite(b) else '-'}）"
                rows.append(row)
    out += [_md(pd.DataFrame(rows)), "", "（2021 以後；括號內全期。）", "",
            "![資金規模與摩擦](gap.png)", ""]

    # --- settlement
    out += ["## 交割資金", "",
            f"手續費 6 折、零股滑價 {emp:.2%}。每個換股日的「淨付款」= 買進金額 + 手續費 − 賣出金額 + 手續費 + 證交稅，"
            "T+2 從交割戶扣（負數是收款）；以前一天收盤時的帳戶淨值為分母。每年第一天從現金建倉的那一次不算。"
            "「不足」是付完以後現金變成負數的部分，也就是帳上要另外準備的交割款。", ""]
    rows = []
    for _, r in stl.iterrows():
        rows.append({"實驗": LABEL[r["exp"]], "方案": r["scheme"], "資金": _wan(r["capital"]),
                     "一般換股日淨付款：中位數": _pct(r["pay_median"], 2, True),
                     "95% 分位": _pct(r["pay_p95"], 2, True), "最大": _pct(r["pay_max"], 2, True),
                     "回補滿倉：淨付款最大": _pct(r["up_pay_max"], 1, True),
                     "回補前現金最少": _pct(r["up_cash_min"], 1),
                     "降半倉：淨收款": _pct(-r["dn_pay_min"], 1) if np.isfinite(r["dn_pay_min"]) else "-",
                     "不足的天數": f"{r['short_days']}/{r['days']}",
                     "不足：99% 分位": _pct(r["short_p99"], 2), "不足：最大": _pct(r["short_max"], 2),
                     "最大那天": r["short_max_date"],
                     "沒有賣不掉時：最大": _pct(r["short_max_clean"], 2),
                     "有賣不掉的天數": r["blocked_days"]})
    out += [_md(pd.DataFrame(rows)), ""]
    return "\n".join(out) + "\n"
