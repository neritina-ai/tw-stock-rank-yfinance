"""Benchmarks from TWSE public data.

Yahoo's ``0050.TW`` cannot serve as the benchmark: its history starts in 2009,
the 2025 four-for-one split is only back-adjusted to 2014-01-02 (a fake -75%
day), and early 2009 has moves no ETF can make under the 7% daily limit.
Instead:

* **0050 total return**: TWSE daily closes (``STOCK_DAY``, available from
  2010-01-04) with every cash distribution from the ex-rights/ex-dividend
  results (``TWT49U``): on an ex-date the return is measured from the official
  reference price.  The 2025-06-18 four-for-one split is detected from the
  price ratio (an ETF cannot fall 40% in a day).
* **TAIEX total return index** (發行量加權股價報酬指數, ``MFI94U``, from 2003):
  stands in for 0050 before 2010, and is shown on its own as the dividend-
  inclusive market benchmark.

Requests are throttled and every month is cached under ``data/twse/`` so an
interrupted download resumes where it stopped.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import requests

from .config import DATA_DIR

BASE = "https://www.twse.com.tw/rwd/zh"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
CACHE = DATA_DIR / "twse"
PAUSE = 3.0
ETF_FROM = "2010-01"  # STOCK_DAY refuses dates before 2010-01-04
TR_FROM = "2003-01"


def _roc(s: str) -> pd.Timestamp:
    s = s.strip().replace("年", "/").replace("月", "/").replace("日", "")
    y, m, d = (int(x) for x in s.split("/"))
    return pd.Timestamp(y + 1911, m, d)


def _num(s: str) -> float:
    s = str(s).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _get(url: str, path, log=print) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    for wait in (10, 30, 60, 120, None):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            j = r.json()
            if j.get("stat") == "OK" or "沒有符合" in str(j.get("stat")):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
                time.sleep(PAUSE)
                return j
            log(f"TWSE said {j.get('stat')!r} for {url}")
        except (requests.RequestException, ValueError) as exc:
            log(f"TWSE request failed ({exc!r}); retrying")
        if wait is None:
            break
        time.sleep(wait)
    raise RuntimeError(f"TWSE keeps refusing {url}")


def _months(start: str, end: pd.Timestamp):
    return pd.period_range(start, end, freq="M")


def fetch(end: str | pd.Timestamp, log=print) -> None:
    end = pd.Timestamp(end)
    for m in _months(ETF_FROM, end):
        _get(f"{BASE}/afterTrading/STOCK_DAY?date={m.year}{m.month:02d}01&stockNo=0050&response=json",
             CACHE / "0050" / f"{m}.json", log)
    for m in _months(TR_FROM, end):
        _get(f"{BASE}/TAIEX/MFI94U?date={m.year}{m.month:02d}01&response=json",
             CACHE / "tr" / f"{m}.json", log)
    path = CACHE / "dividends.json"
    if path.exists():
        path.unlink()  # one request covers everything; refresh it every time
    _get(f"{BASE}/exRight/TWT49U?startDate=20050101&endDate={end:%Y%m%d}&response=json", path, log)
    log("TWSE benchmarks cached")


def tr_index() -> pd.Series:
    rows = []
    for p in sorted((CACHE / "tr").glob("*.json")):
        for d, v in json.loads(p.read_text(encoding="utf-8")).get("data") or []:
            rows.append((_roc(d), _num(v)))
    s = pd.Series(dict(rows)).sort_index()
    return s[np.isfinite(s)]


def dividends(stock: str = "0050") -> pd.DataFrame:
    j = json.loads((CACHE / "dividends.json").read_text(encoding="utf-8"))
    rows = [(_roc(r[0]), _num(r[3]), _num(r[4])) for r in j.get("data") or [] if r[1] == stock]
    return pd.DataFrame(rows, columns=["date", "prev_close", "ref"]).set_index("date").sort_index()


def etf_close(stock: str = "0050") -> pd.Series:
    rows = []
    for p in sorted((CACHE / stock).glob("*.json")):
        for r in json.loads(p.read_text(encoding="utf-8")).get("data") or []:
            rows.append((_roc(r[0]), _num(r[6])))
    s = pd.Series(dict(rows)).sort_index()
    return s[np.isfinite(s) & (s > 0)]


def etf_total_return(stock: str = "0050") -> pd.Series:
    """Daily total-return index (dividends reinvested, splits removed)."""
    c = etf_close(stock)
    div = dividends(stock)
    prev = c.shift(1)
    base = prev.copy()
    ex = div.index.intersection(c.index)
    base.loc[ex] = div.loc[ex, "ref"]  # official ex-date reference price
    ratio = prev / c
    split = (c / prev < 0.6) & ~c.index.isin(ex)
    base[split] = prev[split] / ratio[split].round()
    r = (c / base - 1).fillna(0.0)
    return (1 + r).cumprod()


def benchmark_0050() -> pd.Series:
    """0050 total return, spliced onto the TAIEX total return index before 2010."""
    etf = etf_total_return()
    tr = tr_index()
    first = etf.index[0]
    head = tr[tr.index <= first]
    head = head / head.iloc[-1] * etf.iloc[0]
    return pd.concat([head.iloc[:-1], etf]).sort_index()
