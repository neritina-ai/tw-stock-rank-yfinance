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


def ensure_dirs() -> None:
    for d in (DATA_DIR, PRICES_DIR, DERIVED_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
