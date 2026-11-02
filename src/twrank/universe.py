"""Listed and OTC ordinary shares from the TWSE ISIN service.

Only ordinary shares (CFI code ``ESVUFR``) are kept, so ETFs, preferred shares,
depositary receipts, warrants and beneficiary certificates stay out.
"""

from __future__ import annotations

import io
import time

import pandas as pd
import requests

from .config import UNIVERSE_CSV, ensure_dirs

ISIN_URL = "https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}"
MARKETS = {2: ("TWSE", ".TW"), 4: ("TPEx", ".TWO")}
COMMON_STOCK_CFI = "ESVUFR"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
# The ISIN server answers a request that follows another too closely with a
# short "FOR SECURITY REASONS" page; space requests out and back off.
PAUSE_BETWEEN_PAGES = 15.0
BACKOFF = (30.0, 60.0, 120.0, 240.0)


def _get_page(mode: int) -> str:
    for wait in (*BACKOFF, None):
        resp = requests.get(ISIN_URL.format(mode=mode), headers=HEADERS, timeout=60)
        resp.raise_for_status()
        resp.encoding = "ms950"
        if "<table" in resp.text.lower() and "FOR SECURITY REASONS" not in resp.text:
            return resp.text
        if wait is None:
            break
        time.sleep(wait)
    raise RuntimeError(f"ISIN server keeps blocking strMode={mode}; try again later")


def _fetch_mode(mode: int) -> pd.DataFrame:
    raw = pd.read_html(io.StringIO(_get_page(mode)), flavor="lxml")[0]
    raw.columns = ["code_name", "isin", "listed_date", "market", "industry", "cfi", "note"]
    raw = raw[raw["cfi"] == COMMON_STOCK_CFI].copy()
    split = raw["code_name"].astype(str).str.split(r"[　\s]+", n=1, expand=True)
    raw["stock_id"] = split[0].str.strip()
    raw["name"] = split[1].str.strip() if split.shape[1] > 1 else ""
    market, suffix = MARKETS[mode]
    raw["market"] = market
    raw["ticker"] = raw["stock_id"] + suffix
    raw["listed_date"] = pd.to_datetime(raw["listed_date"], errors="coerce")
    cols = ["stock_id", "name", "ticker", "market", "industry", "listed_date"]
    return raw[cols].reset_index(drop=True)


def fetch_universe() -> pd.DataFrame:
    frames = []
    for i, mode in enumerate(MARKETS):
        if i:
            time.sleep(PAUSE_BETWEEN_PAGES)
        frames.append(_fetch_mode(mode))
    out = pd.concat(frames, ignore_index=True)
    out = out[out["stock_id"].str.fullmatch(r"\d{4}")]
    return out.drop_duplicates("stock_id").sort_values("stock_id").reset_index(drop=True)


def load_universe(refresh: bool = False) -> pd.DataFrame:
    ensure_dirs()
    if refresh or not UNIVERSE_CSV.exists():
        df = fetch_universe()
        df.to_csv(UNIVERSE_CSV, index=False, encoding="utf-8-sig")
    df = pd.read_csv(UNIVERSE_CSV, dtype={"stock_id": str})
    df["listed_date"] = pd.to_datetime(df["listed_date"], errors="coerce")
    return df


DELISTED_URL = "https://www.twse.com.tw/rwd/zh/company/suspendListing?response=json"


def load_delisted(refresh: bool = False) -> pd.DataFrame:
    """TWSE list of companies whose listing was terminated (date, name, code).

    Dates come in the ROC calendar ("115/09/01" = 2026-09-01).  Only four-digit
    codes (ordinary shares) are kept.  The OTC market publishes no equivalent
    machine-readable list, so this covers TWSE only.
    """
    from .config import DELISTED_CSV

    ensure_dirs()
    if refresh or not DELISTED_CSV.exists():
        resp = requests.get(DELISTED_URL, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        rows = resp.json()["data"]
        df = pd.DataFrame(rows, columns=["roc_date", "name", "stock_id"])
        ymd = df["roc_date"].str.split("/", expand=True).astype(int)
        df["date"] = pd.to_datetime(
            dict(year=ymd[0] + 1911, month=ymd[1], day=ymd[2]), errors="coerce")
        df = df[df["stock_id"].astype(str).str.fullmatch(r"\d{4}")]
        df[["stock_id", "name", "date"]].to_csv(DELISTED_CSV, index=False, encoding="utf-8-sig")
    df = pd.read_csv(DELISTED_CSV, dtype={"stock_id": str}, parse_dates=["date"])
    return df


def name_map() -> dict[str, str]:
    uni = load_universe()
    return dict(zip(uni["stock_id"], uni["name"], strict=False))
