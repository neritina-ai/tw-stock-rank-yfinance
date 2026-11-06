"""Project-wide paths and market constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PRICES_DIR = DATA_DIR / "prices"
DERIVED_DIR = DATA_DIR / "derived"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"

UNIVERSE_CSV = DATA_DIR / "universe.csv"
DELISTED_CSV = DATA_DIR / "delisted.csv"
FAILED_TXT = DATA_DIR / "failed.txt"
DATA_LOCK = ROOT / "data.lock"
DATA_MANIFEST = ROOT / "data.manifest.tsv"

TRADING_DAYS_PER_YEAR = 252

INDEX_TICKER = "^TWII"  # TAIEX price index (no dividends)

# Taiwan retail costs: 0.1425% brokerage each side, 0.3% securities tax on sells.
BUY_COST = 0.001425
SELL_COST = 0.001425 + 0.003

# Yahoo's Taiwan daily history begins in January 2000.
DATA_START = "2000-01-01"

# Live (paper trading) data: recent bars layered on the research cache, with its
# own panels, feature table, models and scores, so the research snapshot pinned
# by data.lock is never touched.
LIVE_DIR = DATA_DIR / "live"
LIVE_PRICES_DIR = LIVE_DIR / "prices"
LIVE_DERIVED_DIR = LIVE_DIR / "derived"
_MODE = {"live": False}


def use_live() -> None:
    """Point every derived file (panels, table, scores) at ``data/live/derived``."""
    _MODE["live"] = True
    for d in (LIVE_PRICES_DIR, LIVE_DERIVED_DIR):
        d.mkdir(parents=True, exist_ok=True)


def is_live() -> bool:
    return _MODE["live"]


def derived_dir():
    return LIVE_DERIVED_DIR if _MODE["live"] else DERIVED_DIR


def ensure_dirs() -> None:
    for d in (DATA_DIR, PRICES_DIR, DERIVED_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
