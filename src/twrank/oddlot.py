"""Intraday odd-lot quotes and official whole-lot opens (TWSE and TPEx public data).

Intraday odd-lot trading (盤中零股) started on 2020-10-26.  Both exchanges
publish a free daily summary per stock:

* TWSE ``afterTrading/TWTC7U`` (盤中零股交易行情單): shares, value, trades,
  first / last / high / low trade price, last bid and ask with sizes.
* TPEx ``afterTrading/oddQuote`` (盤中零股每日收盤行情): the same fields.

The whole-lot side comes from the official daily quotes, not from Yahoo
(Yahoo's raw prices are back-adjusted for later stock dividends):

* TWSE ``afterTrading/MI_INDEX`` (type ALLBUT0999): open, high, low, close.
* TPEx ``afterTrading/dailyQuotes``: the same.

Only the dates that matter are fetched (the trade days of the experiments
studied in ``deploy.py``).  Every (source, date) is parsed into a small
parquet file under ``data/odd/<source>/``; an interrupted run resumes where it
stopped, failures are listed in ``data/odd/failed.txt`` and retried in later
passes.  Requests are spaced out (TWSE blocks clients that ask too fast).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from . import config

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
TWSE = "https://www.twse.com.tw/rwd/zh/afterTrading"
TPEX = "https://www.tpex.org.tw/www/zh-tw/afterTrading"
ODD_START = pd.Timestamp("2020-10-26")  # first intraday odd-lot session
PAUSE = 3.0

ODD_COLS = ["stock", "volume", "value", "trades", "first", "last", "high", "low",
            "bid", "bid_vol", "ask", "ask_vol"]
LOT_COLS = ["stock", "open", "high", "low", "close", "volume", "value"]


def odd_dir():
    return config.DATA_DIR / "odd"


def _num(x) -> float:
    s = str(x).replace(",", "").strip()
    if s in ("", "--", "---", "X"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _tables(j: dict) -> list[tuple[list, list]]:
    out = []
    if "fields" in j and "data" in j:
        out.append((j["fields"], j["data"]))
    for t in j.get("tables") or []:
        if t.get("fields") and t.get("data") is not None:
            out.append((t["fields"], t["data"]))
    return out


def _pick(j: dict, must: tuple[str, ...]) -> tuple[list, list] | None:
    for f, d in _tables(j):
        names = [str(x).strip() for x in f]
        if all(m in names for m in must):
            return names, d
    return None


def _frame(names: list, rows: list, mapping: dict[str, str], cols: list[str]) -> pd.DataFrame:
    idx = {k: names.index(v) for k, v in mapping.items()}
    recs = []
    for r in rows:
        code = str(r[idx["stock"]]).strip()
        if not (len(code) == 4 and code.isdigit()) or code.startswith("0"):
            continue  # ordinary shares: four digits, not 0xxx (ETFs); skips warrants too
        rec = {"stock": code}
        for k, i in idx.items():
            if k != "stock":
                rec[k] = _num(r[i])
        recs.append(rec)
    df = pd.DataFrame(recs, columns=cols)
    for c in cols[1:]:
        df[c] = df[c].astype("float64")
    return df


SOURCES = {
    "twse_odd": {
        "url": TWSE + "/TWTC7U?date={ymd}&selectType=ALL&response=json",
        "must": ("證券代號", "當日第一次成交價"),
        "map": {"stock": "證券代號", "volume": "成交股數", "value": "成交金額",
                "trades": "成交筆數",
                "first": "當日第一次成交價", "last": "當日最後一次成交價", "high": "當日最高價",
                "low": "當日最低價", "bid": "最後揭示買價", "bid_vol": "最後揭示買量",
                "ask": "最後揭示賣價", "ask_vol": "最後揭示賣量"},
        "cols": ODD_COLS,
    },
    "tpex_odd": {
        "url": TPEX + "/oddQuote?date={slash}&id=&response=json",
        "must": ("代號", "首筆成交價"),
        "map": {"stock": "代號", "volume": "成交股數", "value": "成交金額(元)",
                "trades": "成交筆數",
                "first": "首筆成交價", "last": "最後成交價", "high": "最高", "low": "最低",
                "bid": "最後買價", "bid_vol": "最後買量(股)", "ask": "最後賣價",
                "ask_vol": "最後賣量(股)"},
        "cols": ODD_COLS,
    },
    "twse_lot": {
        "url": TWSE + "/MI_INDEX?date={ymd}&type=ALLBUT0999&response=json",
        "must": ("證券代號", "開盤價"),
        "map": {"stock": "證券代號", "open": "開盤價", "high": "最高價", "low": "最低價",
                "close": "收盤價", "volume": "成交股數", "value": "成交金額"},
        "cols": LOT_COLS,
    },
    "tpex_lot": {
        "url": TPEX + "/dailyQuotes?date={slash}&id=&response=json",
        "must": ("代號", "開盤"),
        "map": {"stock": "代號", "open": "開盤", "high": "最高", "low": "最低", "close": "收盤",
                "volume": "成交股數", "value": "成交金額(元)"},
        "cols": LOT_COLS,
    },
}


def _path(src: str, d: pd.Timestamp):
    return odd_dir() / src / f"{d:%Y%m%d}.parquet"


def _fetch_one(src: str, d: pd.Timestamp) -> pd.DataFrame:
    spec = SOURCES[src]
    url = spec["url"].format(ymd=f"{d:%Y%m%d}", slash=f"{d:%Y}%2F{d:%m}%2F{d:%d}")
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    j = r.json()
    hit = _pick(j, spec["must"])
    if hit is None:
        stat = str(j.get("stat", ""))
        # a trading day always has rows; an explicit "no data" answer is kept as empty
        if "沒有符合" in stat or "很抱歉" in stat:
            return pd.DataFrame(columns=spec["cols"])
        raise RuntimeError(f"{src} {d.date()}: unexpected answer {stat[:60]!r}")
    names, rows = hit
    return _frame(names, rows, spec["map"], spec["cols"])


def fetch(dates, sources=tuple(SOURCES), passes: int = 3, log=print,
          record: bool = True) -> list[tuple[str, str]]:
    """Download every (source, date) not cached yet; returns what still failed.

    ``record`` lists the failures in ``data/odd/failed.txt``.
    """
    dates = sorted({pd.Timestamp(d).normalize() for d in dates if pd.Timestamp(d) >= ODD_START})
    failed: list[tuple[str, str]] = []
    for p in range(passes):
        todo = [(s, d) for d in dates for s in sources if not _path(s, d).exists()]
        if not todo:
            break
        log(f"odd-lot pass {p + 1}: {len(todo)} files")
        failed = []
        for k, (s, d) in enumerate(todo):
            try:
                df = _fetch_one(s, d)
                _path(s, d).parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(_path(s, d), index=False)
            except Exception as exc:  # network error, throttling, odd answer: retry later
                failed.append((s, str(d.date())))
                log(f"  {s} {d.date()} failed: {str(exc)[:100]}")
                time.sleep(30 * (p + 1))
            if (k + 1) % 50 == 0 or k + 1 == len(todo):
                log(f"  {k + 1}/{len(todo)}")
            time.sleep(PAUSE)
    if record:
        f = odd_dir() / "failed.txt"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("".join(f"{s}\t{d}\n" for s, d in failed), encoding="utf-8")
    return failed


def load(kind: str, dates=None) -> pd.DataFrame:
    """Long frame (date, stock, ...) of ``kind`` = "odd" or "lot", both markets."""
    parts = []
    for src in (f"twse_{kind}", f"tpex_{kind}"):
        for p in sorted((odd_dir() / src).glob("*.parquet")):
            d = pd.Timestamp(p.stem)
            if dates is not None and d not in dates:
                continue
            df = pd.read_parquet(p)
            if len(df):
                df.insert(0, "date", d)
                df["market"] = "TWSE" if src.startswith("twse") else "TPEx"
                parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(["date", "stock"], keep="first")


def quotes(dates=None) -> pd.DataFrame:
    """Odd-lot and whole-lot quotes side by side, one row per (date, stock).

    Where the exchange's whole-lot file is missing (TPEx's large daily file
    sometimes cannot be fetched), the open and close come from Yahoo's raw
    prices undone for later splits (``lot_source`` says which).
    """
    odd = load("odd", dates)
    lot = load("lot", dates)
    if odd.empty:
        return pd.DataFrame()
    if lot.empty:
        lot = pd.DataFrame(columns=["date", "stock", "market", *LOT_COLS[1:]])
    m = odd.merge(lot.drop(columns="market"), on=["date", "stock"], how="left",
                  suffixes=("_odd", "_lot"))
    m["lot_source"] = np.where(m["open"].notna(), "exchange", "yahoo")
    miss = m["open"].isna()
    if miss.any():
        from .lots import prices

        P = prices()
        po = (P.raw_open * P.factor).stack()
        pc = (P.raw_close_ff * P.factor).stack()
        key = pd.MultiIndex.from_arrays([m.loc[miss, "date"], m.loc[miss, "stock"]])
        m.loc[miss, "open"] = po.reindex(key).to_numpy()
        m.loc[miss, "close"] = pc.reindex(key).to_numpy()
    for c in ("first", "last", "high_odd", "low_odd", "bid", "ask", "open", "close"):
        m.loc[m[c] <= 0, c] = np.nan
    return m
