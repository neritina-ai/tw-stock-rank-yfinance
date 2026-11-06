"""yfinance download, on-disk cache, wide panels and the data lock.

Layout::

    data/universe.csv              ordinary shares from the TWSE ISIN service
    data/prices/<TICKER>.parquet   raw OHLCV + Adj Close per symbol
    data/fetch_status.csv          per-ticker download status (resumable)
    data/derived/*.parquet         wide date x stock matrices built from the cache

Downloads are throttled and resumable: a ticker that already has a cache file
is skipped, and tickers that failed are recorded and retried in later passes
instead of restarting the whole batch.

Adjusted prices: Yahoo's ``Close`` is split-adjusted but not dividend-adjusted;
``Adj Close`` also removes dividends.  Every bar is scaled by
``Adj Close / Close`` so candles keep their shape without ex-dividend gaps.
Turnover is measured with the raw close times shares traded.
"""

from __future__ import annotations

import hashlib
import io
import time
from collections.abc import Iterable
from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config
from .config import (
    DATA_LOCK,
    DATA_MANIFEST,
    DATA_START,
    INDEX_TICKER,
    PRICES_DIR,
    ensure_dirs,
)
from .universe import load_universe

CACHED = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
STATUS_CSV = PRICES_DIR.parent / "fetch_status.csv"


def price_path(ticker: str):
    return PRICES_DIR / f"{ticker.replace('^', '_')}.parquet"


def _clean(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=CACHED)
    df = df[[c for c in CACHED if c in df.columns]].copy()
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.astype("float64")


def _download(tickers: list[str], start: str, end: str | None) -> tuple[dict, str]:
    """One yfinance call; returns ({ticker: frame}, captured log text)."""
    import yfinance as yf

    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=False,
            actions=False,
            group_by="ticker",
            progress=False,
            threads=True,
            timeout=60,
        )
    out: dict[str, pd.DataFrame] = {}
    if raw is not None and not raw.empty:
        if isinstance(raw.columns, pd.MultiIndex):
            lv0 = set(raw.columns.get_level_values(0))
            for t in tickers:
                if t in lv0:
                    out[t] = _clean(raw[t])
        elif len(tickers) == 1:
            out[tickers[0]] = _clean(raw)
    return out, buf.getvalue()


def _load_status() -> pd.DataFrame:
    if STATUS_CSV.exists():
        return pd.read_csv(STATUS_CSV, dtype={"ticker": str}).set_index("ticker")
    return pd.DataFrame(columns=["status", "rows", "first", "last", "attempts"]).rename_axis(
        "ticker"
    )


def fetch_prices(
    tickers: Iterable[str],
    start: str = DATA_START,
    end: str | None = None,
    batch_size: int = 20,
    pause: float = 2.0,
    max_attempts: int = 4,
    log=print,
) -> pd.DataFrame:
    """Download every ticker that has no cache file yet; return the status table.

    Tickers that come back empty are retried in later passes (smaller batches,
    longer pauses) up to ``max_attempts`` times, then marked ``missing``.
    """
    ensure_dirs()
    status = _load_status()
    tickers = list(dict.fromkeys(tickers))
    for t in tickers:
        if price_path(t).exists() and status.get("status", pd.Series(dtype=str)).get(t) != "ok":
            df = pd.read_parquet(price_path(t))
            status.loc[t, ["status", "rows"]] = ["ok", len(df)]
            status.loc[t, "attempts"] = status.get("attempts", pd.Series(dtype=float)).get(t, 1)

    def save():
        status.sort_index().to_csv(STATUS_CSV)

    for pass_no in range(max_attempts):
        todo = [
            t
            for t in tickers
            if not price_path(t).exists()
            and status.get("status", pd.Series(dtype=str)).get(t) != "missing"
        ]
        if not todo:
            break
        bs = max(1, batch_size // (2**pass_no))
        wait = pause * (2**pass_no)
        log(f"pass {pass_no + 1}: {len(todo)} tickers, batch {bs}, pause {wait:.0f}s")
        for i in range(0, len(todo), bs):
            batch = todo[i : i + bs]
            for attempt in range(4):
                try:
                    got, text = _download(batch, start, end)
                    break
                except Exception as exc:  # network or rate limit; back off and retry
                    got, text = {}, repr(exc)
                    time.sleep(30 * (attempt + 1))
            if "Rate" in text and "limit" in text.lower():
                log("rate limited; sleeping 120s")
                time.sleep(120)
            for t in batch:
                df = got.get(t)
                prev = status.get("attempts", pd.Series(dtype=float)).get(t, 0)
                prev = 0 if pd.isna(prev) else int(prev)
                if df is not None and len(df):
                    df.to_parquet(price_path(t))
                    status.loc[t, ["status", "rows", "first", "last", "attempts"]] = [
                        "ok",
                        len(df),
                        str(df.index[0].date()),
                        str(df.index[-1].date()),
                        prev + 1,
                    ]
                else:
                    final = prev + 1 >= max_attempts
                    status.loc[t, ["status", "rows", "attempts"]] = [
                        "missing" if final else "failed",
                        0,
                        prev + 1,
                    ]
            done = min(i + bs, len(todo))
            if (i // bs) % 10 == 0 or done == len(todo):
                n_ok = int((status["status"] == "ok").sum())
                log(f"  {done}/{len(todo)} this pass, {n_ok} cached")
                save()
            time.sleep(wait)
        save()
    save()
    return status


def fetch_all(end: str | None = None, log=print) -> pd.DataFrame:
    uni = load_universe()
    tickers = [INDEX_TICKER, *uni["ticker"].tolist()]
    return fetch_prices(tickers, end=end, log=log)


# ---------------------------------------------------------------------------
# Stock splits (Yahoo books Taiwanese stock dividends as splits)
# ---------------------------------------------------------------------------

SPLITS_CSV = PRICES_DIR.parent / "splits.csv"


def fetch_splits(tickers: Iterable[str], batch_size: int = 20, pause: float = 2.0,
                 log=print) -> pd.DataFrame:
    """Split ratios per ticker (resumable; tickers already listed are skipped).

    Yahoo's raw ``Close`` is back-adjusted for splits, and it records stock
    dividends (配股) and capital reductions as splits, so an old raw price can be
    a fraction of what actually traded (Hon Hai shows 41 in January 2005 when it
    traded near 130).  Lot sizes need the real price: ``raw x product of the
    ratios of every later split``.  Tickers without splits get a row with an
    empty date so that a rerun does not ask for them again.
    """
    import yfinance as yf

    if SPLITS_CSV.exists():
        done = pd.read_csv(SPLITS_CSV, dtype={"ticker": str})
    else:
        done = pd.DataFrame(columns=["ticker", "date", "ratio"])
    todo = [t for t in dict.fromkeys(tickers) if t not in set(done["ticker"])]
    log(f"splits: {len(todo)} tickers to ask")
    for i in range(0, len(todo), batch_size):
        batch = todo[i: i + batch_size]
        rows = []
        for attempt in range(4):
            try:
                buf = io.StringIO()
                with redirect_stdout(buf), redirect_stderr(buf):
                    raw = yf.download(batch, start=DATA_START, auto_adjust=False, actions=True,
                                      group_by="ticker", progress=False, threads=True, timeout=60)
                break
            except Exception:  # network or rate limit; back off and retry
                raw = None
                time.sleep(30 * (attempt + 1))
        if raw is None or raw.empty:
            log(f"  batch {i // batch_size + 1}: empty answer, will retry on the next run")
            time.sleep(60)
            continue
        for t in batch:
            if not isinstance(raw.columns, pd.MultiIndex) or t not in set(
                    raw.columns.get_level_values(0)):
                continue
            sp = raw[t].get("Stock Splits")
            if sp is None:
                continue
            sp = sp[sp.fillna(0) > 0]
            if sp.empty:
                rows.append({"ticker": t, "date": "", "ratio": np.nan})
            for d, r in sp.items():
                rows.append({"ticker": t, "date": str(pd.Timestamp(d).date()), "ratio": float(r)})
        if rows:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_csv(SPLITS_CSV, index=False)
        if (i // batch_size) % 10 == 0:
            log(f"  {min(i + batch_size, len(todo))}/{len(todo)}")
        time.sleep(pause)
    return done


def split_factor(days: pd.DatetimeIndex) -> pd.DataFrame:
    """(date x stock) multiplier turning Yahoo's raw price into the price that traded.

    Stocks without split data get 1.  The factor on day ``d`` is the product of
    the ratios of all splits dated after ``d``.
    """
    if not SPLITS_CSV.exists():
        return pd.DataFrame(index=days)
    sp = pd.read_csv(SPLITS_CSV, dtype={"ticker": str}).dropna(subset=["ratio"])
    sp = sp[sp["date"].astype(str).str.len() > 0]
    out = {}
    for t, g in sp.groupby("ticker"):
        sid = t.split(".")[0]
        ev = pd.Series(g["ratio"].to_numpy(), index=pd.to_datetime(g["date"])).groupby(
            level=0).prod().sort_index()
        # factor(d) = prod of ratios with date > d
        tail = ev[::-1].cumprod()[::-1]  # tail[k] = prod of ratios from event k on
        pos = ev.index.searchsorted(days, side="right")
        vals = np.append(tail.to_numpy(), 1.0)[pos]
        out[sid] = vals
    return pd.DataFrame(out, index=days)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

SPIKE = 0.15  # log move that marks a bar as a candidate spike
SPIKE_REVERT = 0.10  # the neighbours must agree with each other within this log distance


def clean_bars(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Drop bars that are not real trades.

    * Zero-volume bars: Yahoo pads holidays and no-trade days with quotes.
    * Spikes: a close that jumps away from *both* neighbours by more than 15%
      in the same direction while the neighbours agree within 10%.  These are
      foreign prices mixed into a series (e.g. two unrelated stocks both
      jumping 50% for a single day on 2005-12-22 and 2005-12-26).  Taiwan's
      daily limit is 7% (10% since 2015-06), so a real one-day round trip of
      that size cannot happen.

    Returns the cleaned frame and the number of zero-volume and spike bars.
    """
    if df.empty:
        return df, 0, 0
    zero = df["Volume"].fillna(0) <= 0
    out = df[~zero]
    n_spike = 0
    for _ in range(3):
        lc = np.log(out["Adj Close"].to_numpy())
        if len(lc) < 3:
            break
        up = lc[1:-1] - lc[:-2]
        dn = lc[1:-1] - lc[2:]
        nb = np.abs(lc[2:] - lc[:-2])
        spike = (np.abs(up) > SPIKE) & (np.abs(dn) > SPIKE) & (np.sign(up) == np.sign(dn)) & (
            nb < SPIKE_REVERT)
        if not spike.any():
            break
        mask = np.zeros(len(out), dtype=bool)
        mask[1:-1] = spike
        n_spike += int(mask.sum())
        out = out[~mask]
    return out, int(zero.sum()), n_spike


LIMIT_CHANGE = pd.Timestamp("2015-06-01")  # daily price limit 7% -> 10%
JUMP_TOL = 0.015


def fix_jumps(df: pd.DataFrame, sessions: pd.DatetimeIndex) -> tuple[pd.DataFrame, int]:
    """Clip moves no real session can produce, treating the excess as an unrecorded action.

    Between two bars ``k`` sessions apart (k <= 5) the adjusted close cannot
    move more than the daily limit compounded ``k`` times.  A larger move is
    a corporate action Yahoo did not adjust for (capital reduction, rights
    issue, a stock dividend booked on the wrong day) or a bad print; its
    excess is removed by rescaling all earlier bars, and the move itself is
    set to the limit in the same direction.  The first five bars of a series
    (listing days have no limit) and gaps longer than five sessions are left
    alone.  Returns the adjusted frame and the number of clipped moves.
    """
    if len(df) < 7:
        return df, 0
    lc = np.log(df["Adj Close"].to_numpy())
    pos = sessions.searchsorted(df.index)
    gap = np.diff(pos).clip(min=1)
    d = np.diff(lc)
    lim = np.where(df.index[1:] < LIMIT_CHANGE, 0.07, 0.10)
    up_max = np.log1p(lim + JUMP_TOL) * gap
    dn_max = np.log1p(-(lim + JUMP_TOL)) * gap
    bad = ((d > up_max) | (d < dn_max)) & (gap <= 5)
    bad[:5] = False
    if not bad.any():
        return df, 0
    new_d = np.where(bad, np.log1p(np.sign(d) * lim) * gap, d)
    excess = d - new_d
    shift = np.concatenate([np.cumsum(excess[::-1])[::-1], [0.0]])
    out = df.copy()
    out["Adj Close"] = out["Adj Close"] * np.exp(shift)
    return out, int(bad.sum())


# ---------------------------------------------------------------------------
# Wide panels
# ---------------------------------------------------------------------------

PANEL_FIELDS = ["open", "high", "low", "close", "raw_close", "raw_open", "volume", "turnover"]
def _panel_path(name: str):
    return config.derived_dir() / f"px_{name}.parquet"


def _cached(ticker: str) -> pd.DataFrame | None:
    p = price_path(ticker)
    return pd.read_parquet(p) if p.exists() else None


def build_panels(log=print, loader=_cached) -> None:
    """Assemble wide matrices (dates x stock_id) from the per-symbol cache.

    ``loader(ticker)`` returns the raw bars of one symbol (default: the research
    cache; live mode passes the cache merged with recent bars).
    """
    ensure_dirs()
    out_dir = config.derived_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    uni = load_universe()
    fields: dict[str, dict[str, pd.Series]] = {k: {} for k in PANEL_FIELDS}
    cleaning = []
    if config.is_live():
        idx_raw = loader(INDEX_TICKER)
        idx_raw.to_parquet(out_dir / "index.parquet")
        load_index.cache_clear()
    sessions = load_index().index
    for tkr, sid in zip(uni["ticker"], uni["stock_id"], strict=True):
        raw = loader(tkr)
        if raw is None or raw.empty:
            continue
        df, n_zero, n_spike = clean_bars(raw)
        df, n_jump = fix_jumps(df, sessions)
        cleaning.append({"stock": sid, "bars": len(df), "zero_volume": n_zero, "spikes": n_spike,
                         "jumps": n_jump})
        if df.empty:
            continue
        f = (df["Adj Close"] / df["Close"]).where(lambda x: np.isfinite(x) & (x > 0), 1.0)
        fields["open"][sid] = df["Open"] * f
        fields["high"][sid] = df["High"] * f
        fields["low"][sid] = df["Low"] * f
        fields["close"][sid] = df["Adj Close"]
        fields["raw_close"][sid] = df["Close"]
        fields["raw_open"][sid] = df["Open"]
        fields["volume"][sid] = df["Volume"]
        fields["turnover"][sid] = df["Close"] * df["Volume"]
    idx = load_index()
    # Trading calendar: days on which the index or any stock printed a bar.
    close = pd.DataFrame(fields["close"]).sort_index()
    days = close.index.union(idx.index).sort_values()
    days = days[days >= pd.Timestamp(DATA_START)]
    for k, d in fields.items():
        m = pd.DataFrame(d).reindex(index=days).sort_index(axis=1).astype("float32")
        # Bars with zero volume are quotes without trades; treat as no trade.
        m.to_parquet(_panel_path(k))
    cl = pd.DataFrame(cleaning)
    cl.to_csv(out_dir / "cleaning.csv", index=False)
    log(f"panels: {len(days)} days x {len(fields['close'])} stocks; dropped "
        f"{int(cl['zero_volume'].sum()):,} zero-volume and {int(cl['spikes'].sum()):,} spike bars, "
        f"clipped {int(cl['jumps'].sum()):,} beyond-limit moves")
    load_panel.cache_clear()


@lru_cache(maxsize=None)
def load_panel(name: str) -> pd.DataFrame:
    return pd.read_parquet(_panel_path(name))


def trading_days() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(load_panel("close").index)


@lru_cache(maxsize=None)
def load_index() -> pd.DataFrame:
    """^TWII bars (price index, no dividends); in live mode the merged live copy."""
    if config.is_live():
        return pd.read_parquet(config.derived_dir() / "index.parquet")
    return pd.read_parquet(price_path(INDEX_TICKER))


def load_bench() -> pd.Series:
    """0050 total return (TWSE data; TAIEX total return index before 2010), see twse.py."""
    from .twse import benchmark_0050

    return benchmark_0050()


def load_tr() -> pd.Series:
    """TAIEX total return index (dividends included), from TWSE."""
    from .twse import tr_index

    return tr_index()


def load_ohlc(ticker: str) -> pd.DataFrame | None:
    """Adjusted OHLC for one ticker, or None when not cached."""
    p = price_path(ticker)
    if not p.exists():
        return None
    df, _, _ = clean_bars(pd.read_parquet(p))
    df, _ = fix_jumps(df, load_index().index)
    f = (df["Adj Close"] / df["Close"]).where(lambda x: np.isfinite(x) & (x > 0), 1.0)
    out = df[["Open", "High", "Low", "Close"]].mul(f, axis=0)
    out["Volume"] = df["Volume"]
    return out


# ---------------------------------------------------------------------------
# Data lock: pin the snapshot behind every result
# ---------------------------------------------------------------------------


def _frame_digest(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(df.index.asi8, dtype="int64").tobytes())
    for c in CACHED:
        if c in df.columns:
            h.update(np.ascontiguousarray(df[c].to_numpy(dtype="float64")).tobytes())
    return h.hexdigest()[:16]


def write_lock(log=print) -> dict:
    """Hash every cached file into data.manifest.tsv and summarise in data.lock."""
    rows = []
    for p in sorted(PRICES_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        rows.append(
            {
                "ticker": p.stem.replace("_", "^", 1) if p.stem.startswith("_") else p.stem,
                "rows": len(df),
                "first": str(df.index[0].date()) if len(df) else "",
                "last": str(df.index[-1].date()) if len(df) else "",
                "sha": _frame_digest(df),
            }
        )
    man = pd.DataFrame(rows)
    man.to_csv(DATA_MANIFEST, sep="\t", index=False)
    combined = hashlib.sha256("\n".join(man["ticker"] + ":" + man["sha"]).encode()).hexdigest()
    lock = {
        "source": "Yahoo Finance via yfinance (auto_adjust=False)",
        "symbols": int(len(man)),
        "dates": f"{man['first'].replace('', np.nan).min()}..{man['last'].max()}",
        "digest": combined[:16],
        "built": pd.Timestamp.now().strftime("%Y-%m-%d"),
    }
    lines = ["# Data snapshot behind the committed results (see README: data)"]
    lines += [f"{k}={v}" for k, v in lock.items()]
    DATA_LOCK.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"data.lock: {lock}")
    return lock


def read_lock() -> dict:
    if not DATA_LOCK.exists():
        return {}
    out = {}
    for line in DATA_LOCK.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out
