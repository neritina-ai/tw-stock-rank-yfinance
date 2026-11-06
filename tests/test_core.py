"""Unit tests on synthetic data (no download needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from twrank import backtest, data, features, portfolio, stacked


def _days(n=400, start="2020-01-01"):
    return pd.bdate_range(start, periods=n)


def _panels(days, n_stocks=6, seed=0):
    rng = np.random.default_rng(seed)
    cols = [f"S{i}" for i in range(n_stocks)]
    r = rng.normal(0.0005, 0.02, (len(days), n_stocks))
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=days, columns=cols)
    openp = close.shift(1).fillna(close) * np.exp(rng.normal(0, 0.005, close.shape))
    high = np.maximum(close, openp) * 1.01
    low = np.minimum(close, openp) * 0.99
    vol = pd.DataFrame(rng.uniform(1e5, 1e6, close.shape), index=days, columns=cols)
    return {"open": openp, "high": high, "low": low, "close": close, "volume": vol,
            "turnover": close * vol}


def test_features_use_only_past_data():
    days = _days()
    P = _panels(days)
    mkt = P["close"].mean(axis=1)
    cut = 300
    full = dict(features.raw_features(P, mkt))
    trunc = dict(features.raw_features({k: v.iloc[:cut] for k, v in P.items()}, mkt.iloc[:cut]))
    assert list(full) == features.FEATURES
    for name in features.FEATURES:
        a = full[name].iloc[:cut].to_numpy()
        b = trunc[name].to_numpy()
        assert np.allclose(a, b, equal_nan=True), name


def test_week_starts_postpone_and_merge():
    days = pd.DatetimeIndex(["2024-01-02", "2024-01-03",  # Monday closed -> Tuesday
                             "2024-01-08", "2024-01-09",
                             # week of 2024-01-15 fully closed
                             "2024-01-24", "2024-01-25"])
    ws = features.week_starts(days)
    assert list(ws.strftime("%m-%d")) == ["01-02", "01-08", "01-24"]


def test_monthly_dates_postpone_only():
    days = pd.bdate_range("2024-01-01", "2024-04-30")
    days = days.drop(pd.DatetimeIndex(["2024-02-09", "2024-02-12", "2024-02-13", "2024-03-01"]))
    got = portfolio.monthly_dates(days, 9)
    assert list(got.strftime("%m-%d")) == ["01-09", "02-14", "03-11", "04-09"]
    got = portfolio.monthly_dates(days, 1)
    assert list(got.strftime("%m-%d")) == ["01-01", "02-01", "03-04", "04-01"]


def test_clean_bars_drops_zero_volume_and_spikes():
    idx = pd.bdate_range("2020-01-01", periods=8)
    px = np.array([10, 10.2, 10.1, 15.0, 10.3, 10.2, 10.4, 10.5])
    df = pd.DataFrame({"Open": px, "High": px, "Low": px, "Close": px, "Adj Close": px,
                       "Volume": [1, 1, 1, 1, 1, 0, 1, 1]}, index=idx)
    out, n_zero, n_spike = data.clean_bars(df)
    assert n_zero == 1 and n_spike == 1
    assert idx[3] not in out.index and idx[5] not in out.index


def test_fix_jumps_clips_unadjusted_action():
    idx = pd.bdate_range("2021-01-01", periods=12)
    px = np.array([100, 101, 102, 103, 104, 105, 106, 53, 54, 55, 56, 57.0])  # 2:1 not adjusted
    df = pd.DataFrame({"Close": px, "Adj Close": px}, index=idx)
    out, n = data.fix_jumps(df, idx)
    assert n == 1
    r = out["Adj Close"].pct_change().dropna()
    assert r.min() == pytest.approx(-0.10, abs=1e-9)
    assert out["Adj Close"].iloc[-1] == pytest.approx(57.0)


def test_backtest_open_execution_and_costs(monkeypatch):
    days = _days(10)
    o = pd.DataFrame({"A": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19.0]}, index=days)
    c = o * 1.01

    class M:
        pass

    m = M()
    m.days, m.open, m.close = days, o, c
    m.arrays = backtest.Market.arrays.__get__(m)
    monkeypatch.setattr(backtest, "_MARKET", m)
    W = pd.DataFrame({"A": [1.0]}, index=[days[2]])
    res = backtest.run(W, days[0], days[-1], limit_guard=False)
    assert res.ret.iloc[:2].abs().sum() == 0
    # day 2: buy at the open, earn open->close, pay the buy cost
    assert res.ret.iloc[2] == pytest.approx(1.01 - 1 - backtest.BUY_COST)
    # day 3: overnight c(2)->o(3) then o(3)->c(3)
    assert res.ret.iloc[3] == pytest.approx(13 * 1.01 / (12 * 1.01) - 1)
    assert res.turnover.sum() == pytest.approx(1.0)


def test_book_buffer_keeps_names_inside_band(monkeypatch):
    days = pd.bdate_range("2024-01-01", periods=15)
    cols = list("ABCDEFGH")
    # week 1: A,B best; week 2: A drops to rank 3 (inside band 2*2=4), C rises
    s = pd.DataFrame(np.nan, index=days, columns=cols)
    s.iloc[:5] = [8, 7, 6, 5, 4, 3, 2, 1]
    s.iloc[5:10] = [6, 7, 8, 5, 4, 3, 2, 1]
    s.iloc[10:] = [1, 7, 8, 5, 4, 3, 2, 6]  # A falls out of the band
    s = s.rank(axis=1, pct=True)
    monkeypatch.setattr(portfolio, "score_panel", lambda k: s)
    monkeypatch.setattr(portfolio, "hold_panel", lambda k, w: s)
    monkeypatch.setattr(portfolio, "risk_on", lambda: pd.Series(True, index=days))
    monkeypatch.setattr(portfolio, "calendar", lambda: days)
    spec = portfolio.Spec("t", "t", top_n=2, band_mult=2.0)
    book = portfolio.build_book(spec, days[0], days[-1], record=True)
    held = [set(t["held"]) for t in book.trades]
    assert held == [{"A", "B"}, {"A", "B"}, {"B", "C"}]


def test_stacked_runs_and_handover():
    holdings = [["A", "B"], ["A", "C"], ["A", "C"], ["D", "C"]]
    runs = stacked.build_runs(holdings)
    assert {(r.stock, r.a, r.b) for r in runs} == {("A", 0, 2), ("B", 0, 0), ("C", 1, 3), ("D", 3, 3)}
    layer = stacked.assign_handover(runs, len(holdings))
    lay = {r.stock: layer[k] for k, r in enumerate(runs)}
    assert lay["C"] == lay["B"] and lay["D"] == lay["A"]


def test_twse_roc_dates_and_etf_total_return(monkeypatch):
    from twrank import twse

    assert twse._roc(" 95/01/02") == pd.Timestamp("2006-01-02")
    assert twse._roc("94年05月19日") == pd.Timestamp("2005-05-19")
    idx = pd.bdate_range("2025-06-02", periods=6)
    close = pd.Series([100.0, 101.0, 98.0, 25.0, 25.5, 26.0], index=idx)  # div on day 2, 4:1 split day 3
    div = pd.DataFrame({"prev_close": [101.0], "ref": [97.0]}, index=[idx[2]])
    monkeypatch.setattr(twse, "etf_close", lambda s="0050": close)
    monkeypatch.setattr(twse, "dividends", lambda s="0050": div)
    tr = twse.etf_total_return()
    r = tr.pct_change().dropna()
    assert r.iloc[1] == pytest.approx(98 / 97 - 1)  # ex-dividend day measured from the reference
    assert r.iloc[2] == pytest.approx(25 * 4 / 98 - 1)  # split removed


def test_kendall_w_and_day_smoothing():
    from twrank import rebalance_report as RR

    idx = pd.bdate_range("2006-01-02", "2025-12-31")
    rng = np.random.default_rng(3)
    noise = pd.DataFrame(rng.normal(0, 0.01, (len(idx), 6)), index=idx, columns=list("abcdef"))
    w, p = RR.kendall_w(noise, list(noise.columns))
    assert w < 0.2 and p > 0.01
    ranked = noise + np.arange(6) * 0.002  # the same order every year
    w, p = RR.kendall_w(ranked, list(ranked.columns))
    assert w > 0.9 and p < 1e-6
    m = pd.DataFrame({20: [1.0] + [float(d) for d in range(1, 29)]}, index=range(0, 29))
    sm = RR.smooth_days(m)
    assert sm.loc[0, 20] == 1.0  # weekly row untouched
    assert sm.loc[10, 20] == pytest.approx(10.0)
    assert sm.loc[1, 20] == pytest.approx((27 + 28 + 1 + 2 + 3) / 5)  # wraps around the month


# ---------------------------------------------------------------------------
# going live: lots, fees, splits, odd-lot parsing, plan helpers
# ---------------------------------------------------------------------------


def _fake_prices(days, raw_open, raw_close, adj_open, adj_close, factor):
    from twrank import lots

    P = lots.Prices.__new__(lots.Prices)
    P.days = days
    P.adj_open = adj_open
    P.adj_close = adj_close
    P.adj_close_ff = adj_close.ffill()
    P.raw_open = raw_open
    P.raw_close_ff = raw_close.ffill()
    P.factor = factor
    return P


def test_order_fees_lots_odd_minimums_and_tax():
    from twrank import lots

    c = lots.Costs(discount=0.6, slip_odd=0.002)
    sh = np.array([2350.0, 10.0, -1500.0])
    px = np.array([50.0, 30.0, 40.0])
    sell = sh < 0
    fee, tax, notional, odd = lots.order_fees(sh, px, sell, "B", c)
    # 2 lots: max(20, floor(100,000 x 0.000855)) = 85; 350 odd at 50.1: floor(17,535 x 0.000855) = 14
    assert fee[0] == 85 + 14 and odd[0] == 350
    assert notional[0] == pytest.approx(100_000 + 350 * 50.1)
    assert fee[1] == 1  # odd-lot minimum
    # sell 1 lot + 500 odd at 40 x 0.998; tax floored per order
    assert tax[2] == np.floor(40_000 * 0.003) + np.floor(500 * 39.92 * 0.003)
    feeA, _, _, oddA = lots.order_fees(sh[:1], px[:1], sell[:1], "A", c)
    # scheme A: 999 + 999 + 352 shares, each its own odd-lot order
    v = 50.1
    assert oddA[0] == 2350
    assert feeA[0] == 2 * np.floor(999 * v * 0.000855) + np.floor(352 * v * 0.000855)


def test_simulate_splits_and_dividends(monkeypatch):
    from twrank import lots

    days = pd.bdate_range("2024-01-01", periods=6)
    raw = pd.DataFrame({"A": [50.0] * 6}, index=days)  # Yahoo: back-adjusted for the split
    k = pd.DataFrame({"A": [2.0, 2.0, 2.0, 1.0, 1.0, 1.0]}, index=days)  # 2:1 split on day 3
    P = _fake_prices(days, raw, raw, raw, raw, k)
    monkeypatch.setattr(lots, "_PRICES", P)
    monkeypatch.setattr(lots, "split_factor", lambda d: k.reindex(d))
    orders = {days[1]: {"A": 1.0}, days[4]: {"A": 1.0}}
    sim = lots.simulate(orders, days[0], days[-1], 10_000, "B", lots.Costs())
    # day 1: 100 real shares at 100 (fee 14); day 4: the split made them 200 at 50,
    # the account is worth 9,986 so the target is 199 shares
    assert sim.orders.iloc[0]["buy"] == pytest.approx(10_000)
    assert sim.shares == {"A": 199.0}
    assert sim.nav.iloc[-1] == pytest.approx(10_000 - 14 - 1)  # two fees, no price change

    # cash dividend: raw drops 5 on day 2, the adjusted series does not
    raw2 = pd.DataFrame({"A": [100.0, 100.0, 95.0, 95.0, 95.0, 95.0]}, index=days)
    adj2 = pd.DataFrame({"A": [95.0] * 6}, index=days)
    one = pd.DataFrame({"A": [1.0] * 6}, index=days)
    P2 = _fake_prices(days, raw2, raw2, adj2, adj2, one)
    monkeypatch.setattr(lots, "_PRICES", P2)
    monkeypatch.setattr(lots, "split_factor", lambda d: one.reindex(d))
    sim2 = lots.simulate({days[1]: {"A": 1.0}, days[3]: {"A": 1.0}}, days[0], days[-1], 10_000,
                         "frac", lots.Costs(discount=0.0))
    o = sim2.orders.set_index("date")
    assert o.loc[days[3], "cash_before"] == pytest.approx(500.0)  # 100 shares x 5 dividend
    assert sim2.nav.iloc[-1] == pytest.approx(10_000)


def test_split_factor_products(tmp_path, monkeypatch):
    p = tmp_path / "splits.csv"
    p.write_text("ticker,date,ratio\nX.TW,2020-01-03,1.1\nX.TW,2020-01-07,0.8\nY.TW,,\n",
                 encoding="utf-8")
    monkeypatch.setattr(data, "SPLITS_CSV", p)
    days = pd.bdate_range("2020-01-01", periods=6)
    k = data.split_factor(days)
    assert list(k.columns) == ["X"]
    assert k["X"].round(6).tolist() == [0.88, 0.88, 0.8, 0.8, 1.0, 1.0]


def test_oddlot_parsing_keeps_ordinary_shares():
    from twrank import oddlot

    j = {"stat": "OK", "fields": ["證券代號", "證券名稱", "成交股數", "成交筆數", "成交金額",
                                  "當日第一次成交價", "當日最後一次成交價", "當日最高價", "當日最低價",
                                  "最後揭示買價", "最後揭示買量", "最後揭示賣價", "最後揭示賣量"],
         "data": [["0050", "ETF", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"],
                  ["2330", "台積電", "8,095,522", "483,888", "18,369,146,800", "2,310.00",
                   "2,265.00", "2,310.00", "2,240.00", "2,265.00", "9,497", "2,270.00", "1,124"],
                  ["2331", "X", "0", "0", "0", "--", "--", "--", "--", "--", "0", "--", "0"]]}
    spec = oddlot.SOURCES["twse_odd"]
    names, rows = oddlot._pick(j, spec["must"])
    df = oddlot._frame(names, rows, spec["map"], spec["cols"])
    assert df["stock"].tolist() == ["2330", "2331"]
    assert df.loc[0, "first"] == 2310.0 and df.loc[0, "value"] == 18_369_146_800
    assert np.isnan(df.loc[1, "first"])


def test_plan_helpers():
    from twrank import live

    assert live.order_lines("A", 2350, "B") == [("lot", 2000), ("odd", 350)]
    assert live.order_lines("A", -2350, "A") == [("odd", 999), ("odd", 999), ("odd", 352)]
    assert live.limit_prices(100.0) == (110.0, 90.0)
    assert live.limit_prices(48.0) == (52.8, 43.2)  # 52.8 sits on the 0.1 tick above 50
    assert live.limit_prices(1000.0) == (1100.0, 900.0)
    weekly = portfolio.Spec("w", "w")
    monthly = portfolio.Spec("m", "m", schedule="monthly", day=11)
    fri, mon, tue = (pd.Timestamp(x) for x in ("2026-11-06", "2026-11-09", "2026-11-10"))
    assert live.is_trade_day(weekly, fri, mon) and live.is_trade_day(weekly, fri, tue)
    assert not live.is_trade_day(weekly, mon, tue)
    assert live.is_trade_day(monthly, mon, tue) is False
    assert live.is_trade_day(monthly, tue, pd.Timestamp("2026-11-11"))
    assert live.is_trade_day(monthly, tue, pd.Timestamp("2026-11-12"))  # the 11th was closed
    assert live.next_trade_day(monthly, pd.Timestamp("2026-11-11")) == pd.Timestamp("2026-12-11")


def test_odd_first_reports_unpublished_market(tmp_path, monkeypatch):
    """A market whose odd-lot file comes back empty is reported missing and not cached."""
    from twrank import live, oddlot

    monkeypatch.setattr(oddlot, "_path", lambda s, d: tmp_path / f"{s}.parquet")

    def fake_fetch(dates, sources, **kw):
        for s in sources:
            df = (pd.DataFrame({"stock": ["2330", "1101"], "first": [100.0, 0.0]})
                  if s == "twse_odd" else pd.DataFrame({"stock": [], "first": []}))
            df.to_parquet(oddlot._path(s, None))

    monkeypatch.setattr(oddlot, "fetch", fake_fetch)
    first, covered, missing = live._odd_first(pd.Timestamp("2026-11-06"))
    assert list(first.index) == ["2330"]
    assert covered == {"2330", "1101"}
    assert missing == ["tpex_odd"]
    assert not (tmp_path / "tpex_odd.parquet").exists()
