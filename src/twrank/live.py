"""Live signal and paper trading (``genesis/FIAT-deploy.md``): no broker, no daemon.

Three commands, run by hand:

``update``
    Download the recent bars of every stock (plus dividends and splits) into
    ``data/live/prices/``, lay them over the research cache, and rebuild the
    panels, the feature table and the scores under ``data/live/derived/``.
    The research snapshot (``data/prices``, ``data/derived``, ``data.lock``) is
    never touched.  The feature table gets one extra row: the next session,
    which has no bar yet; its features are those of the latest close.
``plan``
    For a trade date, use the data through the previous session to build the
    strategy's next book, size it in whole shares for the paper account and
    write the order list (buy / sell, whole lots / odd lots).
``paper``
    After that date's close (and an ``update``), fill the plan at the real
    open (whole lots) and at the first intraday odd-lot print of the day (odd
    lots, from the exchanges' free daily summary), and book it.

A paper account lives in its own directory (default ``paper/default/``,
git-ignored)::

    ledger.json   settings, cash, shares, last order day and exposure
    fills.csv     every simulated fill (or why it did not fill)
    nav.csv       daily cash, market value and account value
    plans/<date>.json, plans/<date>.csv   what ``plan`` said

Models.  The scores come from the same rolling refits as the research run:
the model of a date is the one fit at the latest cutoff on or before it
(every 26 weekly trade days from 2004), on every weekly row whose label had
closed before the cutoff.  Live mode refits only the cutoffs it needs and
keeps them in ``data/live/models/``.
"""

from __future__ import annotations

import io
import json
import math
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .config import INDEX_TICKER, LIVE_DIR, LIVE_PRICES_DIR, ROOT

MODELS_DIR = LIVE_DIR / "models"
PAPER_DIR = ROOT / "paper"
OVERLAP_DAYS = 35  # recent bars overlap the research cache by this many calendar days
SCORE_SESSIONS = 400  # sessions scored live (enough for the hold-score smoothing)
LIVE_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def research_end() -> pd.Timestamp:
    from .data import read_lock

    dates = read_lock().get("dates", "")
    if ".." in dates:
        return pd.Timestamp(dates.split("..")[1])
    raise RuntimeError("data.lock has no date range; run `twrank fetch` first")


def live_path(ticker: str) -> Path:
    return LIVE_PRICES_DIR / f"{ticker.replace('^', '_')}.parquet"


def _clean_live(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=LIVE_COLS)
    df = df[[c for c in LIVE_COLS if c in df.columns]].copy()
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.astype("float64")


def fetch_recent(start=None, batch_size: int = 50, pause: float = 2.0, passes: int = 3,
                 log=print) -> dict:
    """Download recent bars for the index and every stock (throttled, retried)."""
    import yfinance as yf

    from .universe import load_universe

    LIVE_PRICES_DIR.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(start) if start is not None else research_end() - pd.Timedelta(
        days=OVERLAP_DAYS)
    tickers = [INDEX_TICKER, *load_universe()["ticker"].tolist()]
    t0 = time.time()
    got: dict[str, pd.Timestamp] = {}
    todo = list(tickers)
    for p in range(passes):
        if not todo:
            break
        bs = max(5, batch_size // (2 ** p))
        log(f"recent bars from {start.date()}: pass {p + 1}, {len(todo)} symbols, batch {bs}")
        failed = []
        for i in range(0, len(todo), bs):
            batch = todo[i: i + bs]
            raw = None
            for attempt in range(3):
                buf = io.StringIO()
                try:
                    with redirect_stdout(buf), redirect_stderr(buf):
                        raw = yf.download(batch, start=f"{start:%Y-%m-%d}", auto_adjust=False,
                                          actions=True, group_by="ticker", progress=False,
                                          threads=True, timeout=60)
                    break
                except Exception:  # network or rate limit: back off
                    time.sleep(20 * (attempt + 1))
            for t in batch:
                df = None
                if raw is not None and not raw.empty and isinstance(raw.columns, pd.MultiIndex) \
                        and t in set(raw.columns.get_level_values(0)):
                    df = _clean_live(raw[t])
                if df is not None and len(df):
                    df.to_parquet(live_path(t))
                    got[t] = df.index[-1]
                else:
                    failed.append(t)
            time.sleep(pause * (2 ** p))
        todo = failed
    last = pd.Series(got)
    log(f"recent bars: {len(got)}/{len(tickers)} symbols in {time.time() - t0:.0f}s; "
        f"latest bar {last.max().date() if len(last) else '-'} "
        f"({int((last == last.max()).sum()) if len(last) else 0} symbols on it); "
        f"{len(todo)} without data")
    latest = str(last.max().date()) if len(last) else None
    return {"ok": len(got), "missing": todo, "latest": latest}


def merged(ticker: str) -> pd.DataFrame | None:
    """Research cache with the recent bars on top, rescaled across the seam.

    Yahoo restates history after every dividend (``Adj Close``) and split
    (``Close``); the ratio on the first overlapping day carries the old bars
    onto the new basis.
    """
    from .data import CACHED, price_path

    r = pd.read_parquet(price_path(ticker)) if price_path(ticker).exists() else None
    lp = live_path(ticker)
    o = pd.read_parquet(lp)[CACHED] if lp.exists() else None
    if o is None or o.empty:
        return r
    if r is None or r.empty:
        return o
    head = r[r.index < o.index[0]].copy()
    common = r.index.intersection(o.index)
    if len(common):
        d0 = common[0]
        rc = o.at[d0, "Close"] / r.at[d0, "Close"]
        ra = o.at[d0, "Adj Close"] / r.at[d0, "Adj Close"]
        if np.isfinite(rc) and abs(rc - 1) > 1e-6:
            head[["Open", "High", "Low", "Close"]] *= rc
            head["Volume"] /= rc
        if np.isfinite(ra) and abs(ra - 1) > 1e-9:
            head["Adj Close"] *= ra
    return pd.concat([head, o]).sort_index()


def actions(ticker: str) -> pd.DataFrame:
    """Recent dividends (per share) and split ratios from the live download."""
    lp = live_path(ticker)
    if not lp.exists():
        return pd.DataFrame(columns=["Dividends", "Stock Splits"])
    df = pd.read_parquet(lp)
    out = df[[c for c in ("Dividends", "Stock Splits") if c in df.columns]].fillna(0.0)
    return out[(out != 0).any(axis=1)]


def next_weekday(d: pd.Timestamp) -> pd.Timestamp:
    d = d + pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


def meta_path() -> Path:
    return config.LIVE_DERIVED_DIR / "meta.json"


def read_meta() -> dict:
    if not meta_path().exists():
        raise FileNotFoundError("no live data yet: run `twrank update` first")
    return json.loads(meta_path().read_text(encoding="utf-8"))


def update(fetch: bool = True, log=print) -> dict:
    from . import data
    from .features import build_table, calendar

    config.use_live()
    info = fetch_recent(log=log) if fetch else {}
    data.build_panels(log=log, loader=merged)
    data.load_panel.cache_clear()
    days = calendar()
    nxt = next_weekday(days[-1])
    build_table(log=log, days=days.append(pd.DatetimeIndex([nxt])))
    score_live(log=log)
    meta = {"last_session": str(days[-1].date()), "next_session": str(nxt.date()),
            "built": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "missing": info.get("missing", [])}
    meta_path().write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"live data through {meta['last_session']}; the next session scored is "
        f"{meta['next_session']}")
    return meta


_RESEARCH_TABLE: dict = {}


def _research_table():
    """The research snapshot's feature table (models of research-era cutoffs train on it)."""
    if "t" not in _RESEARCH_TABLE:
        from .features import FEATURES

        p = config.DERIVED_DIR / "table.parquet"
        t = pd.read_parquet(p) if p.exists() else None
        _RESEARCH_TABLE["t"] = t
        _RESEARCH_TABLE["X"] = t[FEATURES].to_numpy(dtype="float32") if t is not None else None
    return _RESEARCH_TABLE["t"], _RESEARCH_TABLE["X"]


def _model(table: pd.DataFrame, X: np.ndarray, cutoff: pd.Timestamp, h: int, log=print):
    """The model fit at ``cutoff``, from disk or fit now.

    A cutoff inside the research snapshot is fit on the research feature table,
    so it is bit-for-bit the model behind the backtest (Yahoo revises recent
    bars a little, and boosted trees amplify even that).  Later cutoffs are fit
    on the live table.
    """
    import lightgbm as lgb

    from .train import _fit

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    p = MODELS_DIR / f"h{h}_{cutoff:%Y%m%d}.txt"
    if p.exists():
        return lgb.Booster(model_file=str(p))
    src = "live"
    rt, rX = _research_table()
    if rt is not None and cutoff <= pd.Timestamp(rt["date"].max()):
        table, X, src = rt, rX, "research snapshot"
    y = table[f"y{h}"].to_numpy()
    end = table[f"end{h}"].to_numpy()
    tr = table["weekly"].to_numpy() & (end < np.datetime64(cutoff)) & np.isfinite(y)
    t0 = time.time()
    m = _fit(X[tr], y[tr], "cpu")
    m.save_model(str(p))
    log(f"model h{h} cutoff {cutoff.date()} ({src}): {int(tr.sum()):,} rows, "
        f"{time.time() - t0:.0f}s")
    return m


def score_live(sessions: int = SCORE_SESSIONS, log=print) -> pd.DataFrame:
    from .features import FEATURES, HORIZONS, load_table
    from .train import cutoffs, scores_path

    table = load_table()
    dates = np.sort(table["date"].unique())
    first = dates[max(0, len(dates) - sessions)]
    cuts = cutoffs(table)
    bounds = [*cuts, pd.Timestamp("2100-01-01")]
    X = table[FEATURES].to_numpy(dtype="float32")
    dcol = table["date"].to_numpy()
    want = dcol >= first
    out = table.loc[want, ["date", "stock"]].copy()
    for h in HORIZONS:
        pred = np.full(len(table), np.nan, dtype="float32")
        for k, c in enumerate(cuts):
            seg = want & (dcol >= np.datetime64(c)) & (dcol < np.datetime64(bounds[k + 1]))
            if seg.any():
                pred[seg] = _model(table, X, c, h, log).predict(X[seg])
        out[f"s{h}"] = pred[want]
    out = out.dropna(subset=[f"s{h}" for h in HORIZONS])
    out.to_parquet(scores_path(), index=False)
    log(f"live scores: {out['date'].nunique()} sessions, {pd.Timestamp(out['date'].min()).date()}"
        f"..{pd.Timestamp(out['date'].max()).date()}")
    return out


# ---------------------------------------------------------------------------
# prices and tick sizes
# ---------------------------------------------------------------------------


def tick(p: float) -> float:
    """Tick size of Taiwanese stocks (TWSE and TPEx)."""
    if p < 10:
        return 0.01
    if p < 50:
        return 0.05
    if p < 100:
        return 0.1
    if p < 500:
        return 0.5
    if p < 1000:
        return 1.0
    return 5.0


def limit_prices(ref: float, pct: float = 0.10) -> tuple[float, float]:
    """Limit-up and limit-down prices for a reference price (rounded inside the band)."""
    up = ref * (1 + pct)
    up = math.floor(up / tick(up) + 1e-9) * tick(up)
    dn = ref * (1 - pct)
    dn = math.ceil(dn / tick(dn) - 1e-9) * tick(dn)
    return round(up, 2), round(dn, 2)


def _split_factor_live(stocks: list[str], days: pd.DatetimeIndex) -> pd.DataFrame:
    """raw x factor = price in the shares of that day, from the live splits."""
    from .universe import load_universe

    tick_of = dict(zip(load_universe()["stock_id"], load_universe()["ticker"], strict=False))
    out = pd.DataFrame(1.0, index=days, columns=stocks)
    for s in stocks:
        a = actions(tick_of.get(s, s))
        sp = a.get("Stock Splits", pd.Series(dtype=float))
        for d, r in sp[sp > 0].items():
            out.loc[out.index < d, s] *= r
    return out


# ---------------------------------------------------------------------------
# the paper account
# ---------------------------------------------------------------------------


@dataclass
class Ledger:
    path: Path
    settings: dict
    cash: float
    positions: dict = field(default_factory=dict)  # stock -> shares
    last_order: str | None = None
    last_scale: float | None = None
    last_mark: str | None = None
    recorded: list = field(default_factory=list)  # trade dates already booked

    @staticmethod
    def resolve(name_or_path: str | None) -> Path:
        if not name_or_path:
            return PAPER_DIR / "default"
        p = Path(name_or_path)
        return p if (p.is_absolute() or len(p.parts) > 1) else PAPER_DIR / name_or_path

    @classmethod
    def load(cls, path: Path) -> Ledger:
        j = json.loads((path / "ledger.json").read_text(encoding="utf-8"))
        j.pop("path", None)
        return cls(path=path, **j)

    @classmethod
    def create(cls, path: Path, settings: dict) -> Ledger:
        if (path / "ledger.json").exists():
            raise FileExistsError(f"{path} already has a ledger")
        path.mkdir(parents=True, exist_ok=True)
        lg = cls(path=path, settings=settings, cash=float(settings["capital"]))
        lg.save()
        return lg

    def save(self) -> None:
        d = asdict(self)
        d.pop("path")
        d["positions"] = {k: int(v) for k, v in sorted(self.positions.items()) if v > 0}
        (self.path / "ledger.json").write_text(json.dumps(d, ensure_ascii=False, indent=1),
                                               encoding="utf-8")

    def held(self) -> list[str]:
        return [s for s, n in self.positions.items() if n > 0]

    def costs(self):
        from .lots import Costs

        s = self.settings
        return Costs(discount=s["discount"], min_fee_lot=s["min_fee_lot"],
                     min_fee_odd=s["min_fee_odd"], slip_odd=s["slip_odd"])

    def append_csv(self, name: str, rows: list[dict]) -> None:
        if not rows:
            return
        p = self.path / name
        new = not p.exists()
        # a byte-order mark only at the top, so spreadsheet programs read the Chinese names
        pd.DataFrame(rows).to_csv(p, mode="a", header=new, index=False,
                                  encoding="utf-8-sig" if new else "utf-8")


def _fee_tax(shares: int, price: float, board: str, sell: bool, c) -> tuple[float, float]:
    rate = 0.001425 * c.discount
    minimum = c.min_fee_lot if board == "lot" else c.min_fee_odd
    v = shares * price
    fee = max(minimum, math.floor(v * rate)) if shares else 0.0
    tax = math.floor(v * 0.003) if sell else 0.0
    return float(fee), float(tax)


def order_lines(stock: str, delta: int, scheme: str) -> list[tuple[str, int]]:
    """Split a share change into (board, shares) orders: whole lots and odd lots <= 999."""
    n = abs(int(delta))
    lots = (n // 1000) * 1000 if scheme == "B" else 0
    out = [("lot", lots)] if lots else []
    odd = n - lots
    while odd > 0:
        k = min(odd, 999)
        out.append(("odd", k))
        odd -= k
    return out


def _closes(stocks: list[str], upto: pd.Timestamp) -> pd.Series:
    from .data import load_panel

    rc = load_panel("raw_close")
    sub = rc.loc[:upto, [s for s in stocks if s in rc.columns]].ffill()
    return sub.iloc[-1] if len(sub) else pd.Series(dtype=float)


def is_trade_day(spec, last: pd.Timestamp, date: pd.Timestamp) -> bool:
    """Would ``date`` (the first session after ``last``) be a scheduled trade day?"""
    if spec.schedule == "weekly":
        return tuple(last.isocalendar())[:2] != tuple(date.isocalendar())[:2]
    if spec.schedule == "monthly":
        target = pd.Timestamp(date.year, date.month, spec.day)
        return last < target <= date
    raise ValueError(spec.schedule)


def next_trade_day(spec, last: pd.Timestamp) -> pd.Timestamp:
    """The next scheduled trade day after ``last`` if every weekday were a session."""
    d = next_weekday(last)
    prev = last
    while not is_trade_day(spec, prev, d):
        prev, d = d, next_weekday(d)
    return d


def plan(date=None, capital: float | None = None, ledger: str | None = None, exp: str = "w12",
         scheme: str = "B", discount: float = 0.6, min_fee_lot: float = 20.0,
         min_fee_odd: float = 1.0, slip_odd: float = 0.002, reserve: float = 0.0,
         force: bool = False, log=print) -> dict:
    from . import experiments as X
    from .data import load_index
    from .features import calendar
    from .portfolio import REFRESH_DAYS, hold_panel, score_panel, select
    from .universe import name_map

    config.use_live()
    meta = read_meta()
    last = pd.Timestamp(meta["last_session"])
    virtual = pd.Timestamp(meta["next_session"])
    date = pd.Timestamp(date) if date is not None else virtual
    if date <= last:
        raise ValueError(f"{date.date()} is not after the latest data ({last.date()}); "
                         "plan the next session, or run `twrank update` later for newer data")
    path = Ledger.resolve(ledger)
    if (path / "ledger.json").exists():
        lg = Ledger.load(path)
        if capital is not None and abs(capital - lg.settings["capital"]) > 0.5:
            log(f"note: {path} already exists (capital {lg.settings['capital']:,.0f}); "
                "--capital is ignored")
    else:
        if capital is None:
            raise ValueError(f"no ledger at {path}: give --capital to open one")
        lg = Ledger.create(path, {"capital": float(capital), "exp": exp, "scheme": scheme,
                                  "discount": discount, "min_fee_lot": min_fee_lot,
                                  "min_fee_odd": min_fee_odd, "slip_odd": slip_odd,
                                  "reserve": reserve,
                                  "created": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")})
        log(f"opened paper account {path} with {capital:,.0f}")
    st = lg.settings
    spec = X.get(st["exp"])
    unbooked = [p.stem for p in (path / "plans").glob("*.json")
                if p.stem not in lg.recorded and pd.Timestamp(p.stem) < date] \
        if (path / "plans").exists() else []
    if unbooked:
        log(f"warning: plans {unbooked} were never booked with `twrank paper`; the account "
            "still shows the holdings from before them")

    trade_day = is_trade_day(spec, last, date)
    score = score_panel(spec.score).loc[virtual]
    hold = hold_panel(spec.score, spec.smooth_weeks).loc[virtual]
    days = calendar()
    px = load_index()["Close"].reindex(days).ffill()
    risk_on = bool(px.iloc[-1] > px.ewm(span=120).mean().iloc[-1])
    scale = 0.5 if (spec.regime == "half" and not risk_on) else 1.0
    held = lg.held()
    new_held = select(score, hold, held, spec) or []
    stale = lg.last_order is None or (date - pd.Timestamp(lg.last_order)).days >= REFRESH_DAYS
    changed = set(new_held) != set(held)
    scale_changed = lg.last_scale is None or abs(scale - lg.last_scale) > 1e-9
    emit = (trade_day or force) and (changed or scale_changed or stale)

    names = name_map()
    stocks = sorted(set(held) | set(new_held))
    close = _closes(stocks, last)
    mv = sum(n * float(close.get(s, np.nan)) for s, n in lg.positions.items() if n > 0)
    nav = lg.cash + mv
    rank = score.dropna().rank(ascending=False, method="first")
    target = {}
    if emit and new_held:
        per = scale / len(new_held) * nav * (1 - st.get("reserve", 0.0))
        target = {s: math.floor(per / float(close[s])) for s in new_held
                  if s in close and np.isfinite(close[s])}
    costs = lg.costs()
    lines, book_rows = [], []
    for s in stocks:
        cur = int(lg.positions.get(s, 0))
        tgt = target.get(s, cur) if emit else cur
        if emit and s not in new_held:
            tgt = 0
        p = round(float(close.get(s, np.nan)), 2)
        up, dn = limit_prices(p) if np.isfinite(p) else (np.nan, np.nan)
        rk = rank.get(s)
        book_rows.append({"stock": s, "name": names.get(s, ""),
                          "rank": int(rk) if rk is not None else None,
                          "score": round(float(score.get(s, np.nan)), 4),
                          "hold_score": round(float(hold.get(s, np.nan)), 4),
                          "status": "keep" if (s in held and s in new_held) else
                          ("add" if s in new_held else "drop"),
                          "shares_now": cur, "shares_target": tgt, "close": p,
                          "value_target": round(tgt * p)})
        d = tgt - cur
        if d == 0:
            continue
        sell = d < 0
        for board, n in order_lines(s, d, st["scheme"]):
            fee, tax = _fee_tax(n, p, board, sell, costs)
            lines.append({"date": str(date.date()), "stock": s, "name": names.get(s, ""),
                          "side": "sell" if sell else "buy",
                          "board": "整股" if board == "lot" else "零股",
                          "shares": n, "lots": n // 1000 if board == "lot" else 0,
                          "ref_close": p, "limit_up": up, "limit_down": dn,
                          "est_value": round(n * p), "est_fee": fee, "est_tax": tax})
    buys = sum(x["est_value"] + x["est_fee"] for x in lines if x["side"] == "buy")
    sells = sum(x["est_value"] - x["est_fee"] - x["est_tax"] for x in lines if x["side"] == "sell")
    out = {
        "date": str(date.date()), "data_through": str(last.date()), "experiment": spec.id,
        "trade_day": trade_day, "orders_issued": bool(emit),
        "why": ("not a scheduled trade day" if not (trade_day or force) else
                "names changed" if changed else "exposure changed" if scale_changed else
                f"{REFRESH_DAYS}-day re-weighting" if stale else "nothing to do"),
        "next_trade_day": str(next_trade_day(spec, last).date()),
        "risk_on": risk_on, "scale": scale, "nav_at_close": round(nav), "cash": round(lg.cash),
        "added": [s for s in new_held if s not in held] if emit else [],
        "dropped": [s for s in held if s not in new_held] if emit else [],
        "est_buy": round(buys), "est_sell": round(sells),
        "est_settlement": round(buys - sells),  # >0: the account pays this at T+2
        "book": book_rows, "orders": lines,
    }
    pdir = path / "plans"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / f"{date.date()}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    pd.DataFrame(lines, columns=["date", "stock", "name", "side", "board", "shares", "lots",
                                 "ref_close", "limit_up", "limit_down", "est_value", "est_fee",
                                 "est_tax"]).to_csv(pdir / f"{date.date()}.csv", index=False,
                                                    encoding="utf-8-sig")
    return out


ODD_MARKETS = {"twse_odd": "證交所", "tpex_odd": "櫃買"}


def _odd_first(date: pd.Timestamp) -> tuple[pd.Series, set, list[str]]:
    """First intraday odd-lot print on ``date``, the stocks covered, the markets missing.

    Each market's file lists every stock, traded or not.  A market whose file
    cannot be fetched or comes back empty (asked before it is published) covers
    nothing; the empty answer is not cached, so a later run asks again.
    """
    from . import oddlot

    first, covered, missing = [], set(), []
    for src in ODD_MARKETS:
        try:
            oddlot.fetch([date], sources=(src,), passes=2, log=lambda m: None, record=False)
            df = pd.read_parquet(oddlot._path(src, date))
        except Exception:  # no network, or not published yet
            missing.append(src)
            continue
        if df.empty:
            oddlot._path(src, date).unlink(missing_ok=True)
            missing.append(src)
            continue
        covered |= set(df["stock"])
        s = df.set_index("stock")["first"]
        first.append(s[s > 0])
    return (pd.concat(first) if first else pd.Series(dtype=float)), covered, missing


class _Marker:
    """Closes, split factors and corporate actions for valuing a paper account."""

    def __init__(self, stocks: list[str]):
        from .data import load_panel
        from .features import calendar
        from .universe import load_universe

        self.days = calendar()
        uni = load_universe()
        tick_of = dict(zip(uni["stock_id"], uni["ticker"], strict=False))
        self.rc = load_panel("raw_close").reindex(self.days).ffill()
        self.k = _split_factor_live(stocks, self.days)
        self.acts = {s: actions(tick_of.get(s, s)) for s in stocks}

    def price(self, d, s) -> float:
        k = float(self.k.at[d, s]) if s in self.k.columns else 1.0
        return float(self.rc.at[d, s]) * k

    def apply_actions(self, lg: Ledger, d) -> list[str]:
        """Ex-dividend cash and stock dividends / splits of day ``d`` on the current shares."""
        note = []
        for s in list(lg.held()):
            a = self.acts.get(s)
            if a is None or d not in a.index:
                continue
            k = float(self.k.at[d, s]) if s in self.k.columns else 1.0
            div = float(a.at[d, "Dividends"]) * k if "Dividends" in a else 0.0
            ratio = float(a.at[d, "Stock Splits"]) if "Stock Splits" in a else 0.0
            if div > 0:
                lg.cash += lg.positions[s] * div
                note.append(f"{s} 除息 {div:g}")
            if ratio > 0 and abs(ratio - 1) > 1e-9:
                new = lg.positions[s] * ratio
                lg.cash += (new - math.floor(new)) * self.price(d, s)
                lg.positions[s] = math.floor(new)
                note.append(f"{s} 配股/分割 x{ratio:g}")
        return note


def mark(lg: Ledger, upto: pd.Timestamp, skip_actions_on=None, log=print) -> list[dict]:
    """Value the account at every session after ``last_mark`` through ``upto``.

    Cash dividends are credited on the ex-date and stock dividends / splits
    change the share count (a fractional share is paid out at that close).
    """
    m = _Marker(lg.held())
    days = m.days
    start = pd.Timestamp(lg.last_mark) if lg.last_mark else None
    sess = days[(days <= upto) & ((days > start) if start is not None else (days >= upto))]
    if not len(sess):
        return []
    rows = []
    for d in sess:
        skip = skip_actions_on is not None and d == skip_actions_on
        note = [] if skip else m.apply_actions(lg, d)
        mv = sum(n * m.price(d, s) for s, n in lg.positions.items() if n > 0)
        rows.append({"date": str(d.date()), "cash": round(lg.cash, 2), "market_value": round(mv),
                     "nav": round(lg.cash + mv), "positions": len(lg.held()),
                     "note": "; ".join(note)})
    lg.last_mark = str(sess[-1].date())
    lg.append_csv("nav.csv", rows)
    return rows


def paper(date, ledger: str | None = None, odd: str = "exchange", log=print) -> dict:
    """Book the plan of ``date`` at the real open (and odd-lot prints), then mark to market."""
    from .data import load_panel
    from .features import calendar

    config.use_live()
    date = pd.Timestamp(date)
    path = Ledger.resolve(ledger)
    lg = Ledger.load(path)
    days = calendar()
    if date not in days:
        raise ValueError(f"no bar for {date.date()} yet (data through {days[-1].date()}); "
                         "run `twrank update` after the close")
    if str(date.date()) in lg.recorded:
        raise ValueError(f"{date.date()} is already booked in {path}")
    if lg.last_mark and pd.Timestamp(lg.last_mark) >= date:
        raise ValueError(f"{path} is already valued through {lg.last_mark}; book trade days "
                         "in order")
    pf = path / "plans" / f"{date.date()}.json"
    if not pf.exists():
        raise FileNotFoundError(f"no plan for {date.date()} in {path / 'plans'}")
    pl = json.loads(pf.read_text(encoding="utf-8"))
    first, covered = pd.Series(dtype=float), set()
    if odd == "exchange" and any(o["board"] == "零股" for o in pl["orders"]):
        first, covered, missing = _odd_first(date)
        if missing:  # booking cannot be redone, so wait for the data rather than guess
            names = "、".join(ODD_MARKETS[m] for m in missing)
            raise ValueError(f"{names} {date.date()} 的盤中零股日資料還沒公布或抓不到：晚一點再跑，"
                             "或用 --odd slip 改用開盤價加減滑價記帳")
    prev = days[days.get_loc(date) - 1]
    mark(lg, prev, log=log)
    # an ex-date on the trade day belongs to the shares held at the open
    _Marker(lg.held()).apply_actions(lg, date)

    ro = load_panel("raw_open").reindex(days)
    rc = load_panel("raw_close").reindex(days).ffill()
    c = lg.costs()
    fills = []
    for o in sorted(pl["orders"], key=lambda x: x["side"] != "sell"):  # sells first
        s, n, sell = o["stock"], int(o["shares"]), o["side"] == "sell"
        board = "lot" if o["board"] == "整股" else "odd"
        p_open = float(ro.at[date, s]) if s in ro.columns else np.nan
        ref = float(rc.at[prev, s]) if s in rc.columns else np.nan
        up, dn = limit_prices(ref) if np.isfinite(ref) else (np.nan, np.nan)
        status, price = "filled", p_open
        if not np.isfinite(p_open):
            status = "no trade today"
        elif (not sell and p_open >= up - 1e-9) or (sell and p_open <= dn + 1e-9):
            status = "limit at the open"
        elif board == "odd":
            if s in first.index:
                price = float(first[s])
            elif s in covered:
                status = "no odd-lot trade today"
            else:
                price = p_open * (1 - c.slip_odd if sell else 1 + c.slip_odd)
        fee = tax = amount = 0.0
        if status == "filled":
            fee, tax = _fee_tax(n, price, board, sell, c)
            amount = n * price
            if sell:
                n_eff = min(n, int(lg.positions.get(s, 0)))
                if n_eff < n:
                    status = f"only {n_eff} held"
                    fee, tax = _fee_tax(n_eff, price, board, sell, c)
                    amount = n_eff * price
                n = n_eff
                lg.positions[s] = int(lg.positions.get(s, 0)) - n
                lg.cash += amount - fee - tax
            else:
                lg.positions[s] = int(lg.positions.get(s, 0)) + n
                lg.cash -= amount + fee
        fills.append({"date": str(date.date()), "stock": s, "name": o.get("name", ""),
                      "side": o["side"], "board": o["board"], "shares": n if status ==
                      "filled" or status.startswith("only") else 0, "ordered": int(o["shares"]),
                      "price": round(price, 2) if np.isfinite(price) else None,
                      "open": p_open, "amount": round(amount), "fee": fee, "tax": tax,
                      "status": status})
    lg.positions = {s: n for s, n in lg.positions.items() if n > 0}
    lg.append_csv("fills.csv", fills)
    if pl.get("orders_issued"):
        lg.last_order = str(date.date())
        lg.last_scale = float(pl["scale"])
    if str(date.date()) not in lg.recorded:
        lg.recorded.append(str(date.date()))
    rows = mark(lg, days[-1], skip_actions_on=date, log=log)
    lg.save()
    buy = sum(f["amount"] + f["fee"] for f in fills if f["side"] == "buy")
    sell = sum(f["amount"] - f["fee"] - f["tax"] for f in fills if f["side"] == "sell")
    i = days.get_loc(date)
    settles_on = (str(days[i + 2].date()) if i + 2 < len(days) else
                  f"{date + pd.offsets.BDay(2):%Y-%m-%d}，遇休市順延")
    return {"date": str(date.date()), "fills": fills, "settlement": round(buy - sell),
            "settles_on": settles_on,
            "cash": round(lg.cash), "nav": rows[-1]["nav"] if rows else None,
            "unfilled": [f for f in fills if f["status"] != "filled"]}


def mark_only(ledger: str | None = None, log=print) -> list[dict]:
    from .features import calendar

    config.use_live()
    lg = Ledger.load(Ledger.resolve(ledger))
    rows = mark(lg, calendar()[-1], log=log)
    lg.save()
    return rows
