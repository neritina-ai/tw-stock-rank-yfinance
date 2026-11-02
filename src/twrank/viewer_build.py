"""Export experiments as static data for the single-page viewer (``docs/viewer``).

GitHub Pages only serves files, so everything is pre-computed:

``data/meta.json``
    trading calendar (from one lookback before the first trade), stock names,
    the experiment list and the data snapshot digest.
``data/exp/<id>.json``
    one experiment: metrics, daily equity of strategy and 0050, and every
    trade day's ranking list (top 50 by score plus holdings outside it and the
    names dropped that day) with score percentile, weight and holding history.
``data/px/<stock>.bin``
    one stock, fetched on demand: zlib-compressed little-endian arrays of
    ``n`` days each -- int16 close in log ticks of 0.1% (-32768 = no bar),
    int16 open/high/low offsets from the close in the same ticks, then uint8
    score percentiles in steps of 0.5% (255 = not eligible) for the ensemble,
    the 21-day and the 63-day model, every trading day.
"""

from __future__ import annotations

import json
import shutil
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import experiments as X
from .config import DOCS_DIR
from .data import load_bench, load_panel, read_lock
from .features import LOOKBACK, calendar
from .metrics import bench_returns
from .portfolio import score_panel
from .universe import name_map

TEMPLATE = Path(__file__).parent / "templates" / "viewer.html"
OUT = DOCS_DIR / "viewer"
TOP = 50
TICK = 0.001  # log-price tick
MISSING = -32768


def _dump(obj, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")
    return len(text.encode())


def _r(x, nd=4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _metrics(summ: dict) -> dict:
    def pick(m):
        keys = ("cagr", "sharpe", "mdd", "vol", "total", "calmar", "turnover", "cost",
                "positions", "exposure", "years")
        return {k: _r(m.get(k)) for k in keys if k in m}

    return {
        "full": {k: pick(v) for k, v in summ["full"].items()},
        "stages": [{"stage": s["stage"], "start": s["strategy"]["start"],
                    "end": s["strategy"]["end"],
                    **{k: pick(v) for k, v in s.items() if isinstance(v, dict)}}
                   for s in summ["stages"]],
        "yearly": {str(y): {k: _r(v) for k, v in row.items()} for y, row in summ["yearly"].items()},
    }


def export_experiment(spec, days: pd.DatetimeIndex, out: Path) -> tuple[set, int]:
    summ = X.load_summary(spec.id)
    curve = X.load_curve(spec.id)
    trades = X.load_trades(spec.id)
    pos = X.load_positions(spec.id)
    score = score_panel(spec.score)
    pct = score.rank(axis=1, pct=True)
    w = pos.pivot_table(index="date", columns="stock", values="weight", aggfunc="sum")

    day_ix = {d: i for i, d in enumerate(days)}
    held_since: dict[str, int] = {}
    dropped_at: dict[str, int] = {}
    rows_out = []
    union: set[str] = set()
    smin, smax = 1.0, 0.0
    prev: set[str] = set()
    for i, tr in enumerate(trades.itertuples(index=False)):
        d = pd.Timestamp(tr.date)
        held = set(tr.held)
        for s in held - prev:
            held_since[s] = i
            dropped_at.pop(s, None)
        for s in prev - held:
            dropped_at[s] = i
            held_since.pop(s, None)
        prev = held
        p = pct.loc[d].dropna().sort_values(ascending=False, kind="stable")
        listed = list(p.index[:TOP])
        extra = [s for s in tr.held if s not in set(listed)]
        extra += [s for s in tr.dropped if s not in set(listed) and s not in set(extra)]
        wrow = w.loc[d] if d in w.index else pd.Series(dtype=float)
        rows = []
        for s in listed + extra:
            v = p.get(s, np.nan)
            ww = wrow.get(s, 0.0)
            rows.append([s, int(round(v * 1000)) if np.isfinite(v) else -1,
                         int(round(ww * 10000)) if np.isfinite(ww) else 0,
                         held_since.get(s, -1), dropped_at.get(s, -1)])
        union.update(listed)
        union.update(extra)
        rows_out.append({"d": day_ix[d], "x": tr.scale, "n": int(len(p)), "r": rows,
                         "top": len(listed)})
    for s in union:
        if s in pct.columns:
            v = pct.loc[pd.DatetimeIndex(trades["date"]), s].dropna()
            if len(v):
                smin, smax = min(smin, float(v.min())), max(smax, float(v.max()))

    nav = (1 + curve["ret"]).cumprod()
    bench = (1 + bench_returns(load_bench(), curve.index)).cumprod()
    obj = {
        "id": spec.id, "label": spec.label, "note": spec.note, "score": spec.score,
        "spec": spec.params(), "metrics": _metrics(summ), "data": summ.get("data"),
        "extra": {k: summ[k] for k in ("selection",) if k in summ},
        "curve": {"d0": day_ix[curve.index[0]],
                  "nav": [round(float(v), 5) for v in nav],
                  "bench": [round(float(v), 5) for v in bench],
                  "expo": [round(float(v), 3) for v in curve["exposure"]]},
        "trades": rows_out, "smin": round(smin * 100, 2), "smax": round(smax * 100, 2),
    }
    size = _dump(obj, out / "data" / "exp" / f"{spec.id}.json")
    return union, size


def export_prices(stocks: set[str], days: pd.DatetimeIndex, out: Path) -> int:
    o, h, l, c = (load_panel(k).reindex(days) for k in ("open", "high", "low", "close"))
    pcts = {k: score_panel(k).rank(axis=1, pct=True).reindex(days) for k in ("ens", "s21", "s63")}
    pdir = out / "data" / "px"
    if pdir.exists():
        shutil.rmtree(pdir)
    pdir.mkdir(parents=True)
    total = 0
    for s in sorted(stocks):
        if s not in c.columns:
            continue
        lc = np.log(c[s].to_numpy(dtype="float64"))
        ok = np.isfinite(lc)
        ct = np.where(ok, np.round(lc / TICK), MISSING).astype(np.int16)
        parts = [ct]
        for m in (o, h, l):
            lm = np.log(m[s].to_numpy(dtype="float64"))
            off = np.where(ok & np.isfinite(lm), np.round((lm - lc) / TICK), 0)
            parts.append(np.clip(off, -32000, 32000).astype(np.int16))
        blob = b"".join(p.astype("<i2").tobytes() for p in parts)
        for k in ("ens", "s21", "s63"):
            v = pcts[k][s].to_numpy(dtype="float64") if s in pcts[k].columns else np.full(len(days), np.nan)
            q = np.where(np.isfinite(v), np.clip(np.round(v * 200), 0, 200), 255).astype(np.uint8)
            blob += q.tobytes()
        z = zlib.compress(blob, 9)
        (pdir / f"{s}.bin").write_bytes(z)
        total += len(z)
    return total


def build(log=print) -> None:
    specs = X.viewer_experiments()
    cal = calendar()
    first = min(pd.Timestamp(X.load_trades(e.id)["date"].iloc[0]) for e in specs)
    i0 = max(0, cal.searchsorted(first) - LOOKBACK - 10)
    days = cal[i0:]
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(TEMPLATE, OUT / "index.html")
    union: set[str] = set()
    for e in specs:
        u, size = export_experiment(e, days, OUT)
        union |= u
        log(f"exp {e.id}: {size / 1e6:.2f} MB, {len(u)} listed stocks")
    names = name_map()
    lock = read_lock()
    d = np.diff(days.values.astype("datetime64[D]").astype(np.int64))
    meta = {
        "d0": str(days[0].date()), "dd": d.tolist(), "lookback": LOOKBACK,
        "names": {s: names.get(s, s) for s in sorted(union)},
        "exps": [{"id": e.id, "label": e.label, "note": e.note, "tags": list(e.tags)}
                 for e in specs],
        "data": {k: lock.get(k) for k in ("digest", "dates", "symbols")},
        "built": pd.Timestamp.now().strftime("%Y-%m-%d"),
    }
    _dump(meta, OUT / "data" / "meta.json")
    total = export_prices(union, days, OUT)
    log(f"px: {len(union)} stocks, {total / 1e6:.1f} MB -> {OUT}")
