"""Command line entry point: ``uv run twrank <command>``.

Commands, in pipeline order::

    fetch      download the stock list and prices (resumable), build panels, data.lock
    panels     rebuild wide panels and data.lock from the cached prices
    lock       rewrite data.lock and data.manifest.tsv
    features   build the feature/label table
    train      rolling LightGBM refits -> out-of-sample scores
    run        backtest the experiments -> results/exp/<id>/
    wfv        per-stage parameter selection on training windows -> results/exp/wfv/
    rebalance  weekly vs monthly sweeps -> results/rebalance/
    rebalance-report  SUMMARY.md, heatmaps and tables from the sweeps
    stacked    stacked equity curve png per experiment
    viewer     export docs/viewer data
    report     results/report.md and results/equity.png
    all        every step after fetch, in order

Going live (genesis/FIAT-deploy.md)::

    odd-lot    odd-lot and whole-lot quotes of the trade days -> results/deploy/oddlot.json
    deploy     capital x lot scheme x costs sweep, settlement cash -> results/deploy/
    deploy-report  results/deploy/SUMMARY.md and figures
    update     recent bars -> live panels, features and scores (data/live/)
    plan       next trade day's book and order list for a paper account
    paper      book a plan at the real open, value the paper account
"""

from __future__ import annotations

import argparse
import sys
import time


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _cmd_fetch(args) -> None:
    from . import data
    from .universe import load_delisted, load_universe

    uni = load_universe(refresh=args.refresh_universe)
    _log(f"universe: {len(uni)} ordinary shares")
    st = data.fetch_all(end=args.end, log=_log)
    _log(st["status"].value_counts().to_string())
    load_delisted(refresh=args.refresh_universe)
    from .twse import fetch as fetch_twse

    fetch_twse(st["last"].dropna().max(), log=_log)
    data.build_panels(log=_log)
    data.write_lock(log=_log)


def _cmd_panels(args) -> None:
    from . import data

    data.build_panels(log=_log)
    data.write_lock(log=_log)


def _cmd_lock(args) -> None:
    from . import data

    data.write_lock(log=_log)


def _cmd_features(args) -> None:
    from .features import build_table

    build_table(log=_log)


def _cmd_train(args) -> None:
    from .train import train_all

    train_all(device=args.device, log=_log)


def _selected(args):
    from . import experiments as X

    specs = X.all_experiments()
    if args.only:
        specs = [X.get(i) for i in args.only]
    return specs


def _cmd_run(args) -> None:
    from . import experiments as X

    for spec in _selected(args):
        t0 = time.time()
        book, res = X.run_spec(spec, record=True)
        s = X.save(spec, book, res)
        m = s["full"]["strategy"]
        _log(f"{spec.id:<12} CAGR {m['cagr']:+.2%} Sharpe {m['sharpe']:.2f} MDD {m['mdd']:.1%} "
             f"turnover {m['turnover']:.1f}x ({time.time() - t0:.0f}s)")


def _cmd_wfv(args) -> None:
    from . import experiments as X

    sel, book, res = X.wfv_select(X.BASE, log=_log)
    s = X.save(X.WFV, book, res, extra={"selection": sel.to_dict("records")})
    m = s["full"]["strategy"]
    _log(f"wfv CAGR {m['cagr']:+.2%} Sharpe {m['sharpe']:.2f} MDD {m['mdd']:.1%}")


def _cmd_stacked(args) -> None:
    from . import experiments as X
    from .stacked import draw
    from .universe import name_map

    names = name_map()
    specs = _selected(args) if args.only else X.viewer_experiments()
    for spec in specs:
        draw(spec.id, names=names, log=_log)


def _cmd_viewer(args) -> None:
    from .viewer_build import build

    build(log=_log)


def _cmd_report(args) -> None:
    from .report import build

    build(log=_log)


def _cmd_rebalance(args) -> None:
    from . import rebalance as R

    R.run_many(R.grid_specs(), "grid", log=_log)
    R.run_many(R.settings_specs(R.REC_SIZE), f"settings_n{R.REC_SIZE}", log=_log)
    R.wf_choice("grid", refresh=True, log=_log)


def _cmd_rebalance_report(args) -> None:
    from .rebalance_report import build

    build(log=_log)


def _cmd_oddlot(args) -> None:
    from . import deploy, oddlot

    deploy.ensure_splits(log=_log)
    dates = set()
    for s in deploy.SPECS:
        dates |= {t["date"] for t in deploy.book(s).trades}
    failed = oddlot.fetch(sorted(dates), log=_log)
    if failed:
        _log(f"{len(failed)} files still missing (listed in data/odd/failed.txt); rerun later")
    deploy.oddlot_report(log=_log)


def _cmd_deploy(args) -> None:
    from . import deploy

    deploy.run(workers=args.workers, log=_log)


def _cmd_deploy_report(args) -> None:
    from .deploy_report import build

    build(log=_log)


def _cmd_update(args) -> None:
    from . import live

    live.update(fetch=not args.no_fetch, log=_log)


def _money(x) -> str:
    return f"{x:,.0f}"


def _cmd_plan(args) -> None:
    from . import live

    out = live.plan(date=args.date, capital=args.capital, ledger=args.ledger, exp=args.exp,
                    scheme=args.scheme, discount=args.discount, slip_odd=args.slip,
                    reserve=args.reserve, force=args.force, log=_log)
    regime = "滿倉" if out["scale"] == 1.0 else "降半倉（加權指數 <= 120 日 EMA）"
    print(f"\n{out['date']}（資料到 {out['data_through']} 收盤）  實驗 {out['experiment']}  "
          f"{regime}")
    print(f"帳戶淨值（前一日收盤）{_money(out['nav_at_close'])}，現金 {_money(out['cash'])}")
    if not out["orders_issued"]:
        print(f"不下單：{out['why']}。下一個換股日 {out['next_trade_day']}")
        return
    print(f"換股：{out['why']}；新增 {' '.join(out['added']) or '-'}；"
          f"剔除 {' '.join(out['dropped']) or '-'}")
    print(f"\n{'代號':<6}{'名稱':<8}{'買賣':<4}{'盤別':<4}{'數量':>9}{'參考價':>10}"
          f"{'漲停':>10}{'跌停':>10}{'約金額':>12}")
    for o in out["orders"]:
        side = "賣" if o["side"] == "sell" else "買"
        qty = f"{o['lots']} 張" if o["board"] == "整股" else f"{o['shares']} 股"
        print(f"{o['stock']:<6}{o['name'][:6]:<8}{side:<4}{o['board']:<4}{qty:>9}"
              f"{o['ref_close']:>10.2f}{o['limit_up']:>10.2f}{o['limit_down']:>10.2f}"
              f"{o['est_value']:>12,.0f}")
    pay = out["est_settlement"]
    print(f"\n約買進 {_money(out['est_buy'])}、約賣出 {_money(out['est_sell'])}；"
          f"T+2 約{'付' if pay > 0 else '收'} {_money(abs(pay))}（以前一日收盤價估）")
    print(f"委託清單：{live.Ledger.resolve(args.ledger) / 'plans' / (out['date'] + '.csv')}")


def _cmd_paper(args) -> None:
    from . import live

    if args.date is None:
        rows = live.mark_only(ledger=args.ledger, log=_log)
        if rows:
            r = rows[-1]
            print(f"{r['date']} 淨值 {_money(r['nav'])}（現金 {_money(r['cash'])}）")
        else:
            print("沒有新的交易日可以估值")
        return
    out = live.paper(args.date, ledger=args.ledger, odd=args.odd, log=_log)
    for f in out["fills"]:
        side = "賣" if f["side"] == "sell" else "買"
        price = f"{f['price']:.2f}" if f["price"] is not None else "-"
        print(f"{f['stock']:<6}{f['name'][:6]:<8}{side} {f['board']} {f['shares']:>7} 股 "
              f"@ {price:>9}  {f['status']}")
    pay = out["settlement"]
    nav = _money(out["nav"]) if out["nav"] is not None else "-"
    print(f"T+2（{out['settles_on']}）{'付' if pay > 0 else '收'} {_money(abs(pay))}；"
          f"現金 {_money(out['cash'])}；淨值 {nav}")
    if out["unfilled"]:
        print(f"沒成交 {len(out['unfilled'])} 筆，見 fills.csv")


def _cmd_all(args) -> None:
    for fn in (_cmd_features, _cmd_train, _cmd_run, _cmd_wfv, _cmd_rebalance,
               _cmd_rebalance_report, _cmd_stacked, _cmd_viewer, _cmd_report):
        fn(args)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="twrank", description="Taiwan stock LightGBM ranking strategy")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch", help="download the universe and prices (resumable)")
    p.add_argument("--end", default="2026-11-03", help="exclusive end date (default: %(default)s)")
    p.add_argument("--refresh-universe", action="store_true")
    p.set_defaults(func=_cmd_fetch)

    sub.add_parser("panels", help="rebuild panels and data.lock").set_defaults(func=_cmd_panels)
    sub.add_parser("lock", help="rewrite data.lock and data.manifest.tsv").set_defaults(
        func=_cmd_lock)
    sub.add_parser("features", help="build the feature table").set_defaults(func=_cmd_features)

    for name, fn, hlp in (("train", _cmd_train, "rolling LightGBM refits"),
                          ("all", _cmd_all, "every step after fetch")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--device", choices=["gpu", "cpu"], default="cpu",
                       help="LightGBM device; pip wheels are CPU-only (default: %(default)s)")
        p.add_argument("--only", nargs="+", metavar="ID", help="experiment ids (default: all)")
        p.set_defaults(func=fn)

    for name, fn, hlp in (("run", _cmd_run, "backtest experiments"),
                          ("stacked", _cmd_stacked, "stacked equity curve png")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--only", nargs="+", metavar="ID", help="experiment ids (default: all)")
        p.set_defaults(func=fn)

    sub.add_parser("wfv", help="per-stage parameter selection").set_defaults(func=_cmd_wfv)
    sub.add_parser("rebalance", help="weekly vs monthly sweeps (about 15 minutes)").set_defaults(
        func=_cmd_rebalance)
    sub.add_parser("rebalance-report", help="results/rebalance/SUMMARY.md and figures").set_defaults(
        func=_cmd_rebalance_report)
    sub.add_parser("viewer", help="export docs/viewer").set_defaults(func=_cmd_viewer)
    sub.add_parser("report", help="results/report.md").set_defaults(func=_cmd_report)

    sub.add_parser("odd-lot", help="odd-lot quotes of the trade days and their estimate"
                   ).set_defaults(func=_cmd_oddlot)
    p = sub.add_parser("deploy", help="capital x lots x costs sweep and settlement cash")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=_cmd_deploy)
    sub.add_parser("deploy-report", help="results/deploy/SUMMARY.md and figures").set_defaults(
        func=_cmd_deploy_report)

    p = sub.add_parser("update", help="recent bars -> live panels, features and scores")
    p.add_argument("--no-fetch", action="store_true", help="rebuild from the bars already here")
    p.set_defaults(func=_cmd_update)
    p = sub.add_parser("plan", help="order list of the next trade day for a paper account")
    p.add_argument("--date", help="trade date (default: the session after the latest data)")
    p.add_argument("--capital", type=float, help="opens a new paper account with this much cash")
    p.add_argument("--ledger", help="account name under paper/ or a directory (default: default)")
    p.add_argument("--exp", default="w20", help="experiment id for a new account (default: w20)")
    p.add_argument("--scheme", choices=["A", "B"], default="B",
                   help="A: odd lots only; B: whole lots + odd remainder (default: B)")
    p.add_argument("--discount", type=float, default=0.6, help="fee multiplier (default: 0.6)")
    p.add_argument("--slip", type=float, default=0.002,
                   help="odd-lot slippage when exchange data is missing (default: 0.002)")
    p.add_argument("--reserve", type=float, default=0.0,
                   help="fraction of the account kept out of the book (default: 0)")
    p.add_argument("--force", action="store_true", help="issue orders on a non-trade day")
    p.set_defaults(func=_cmd_plan)
    p = sub.add_parser("paper", help="book a plan at the real open; without --date only value")
    p.add_argument("--date", help="trade date whose plan to book")
    p.add_argument("--ledger", help="account name under paper/ or a directory (default: default)")
    p.add_argument("--odd", choices=["exchange", "slip"], default="exchange",
                   help="odd-lot price: first intraday print (exchange data) or open +/- slippage")
    p.set_defaults(func=_cmd_paper)
    return ap


def main(argv: list[str] | None = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)  # argparse rejects unknown arguments
    if args.func in (_cmd_plan, _cmd_paper):  # user-facing refusals: no traceback
        try:
            args.func(args)
        except (ValueError, FileNotFoundError) as exc:
            sys.exit(f"twrank {args.cmd}: {exc}")
        return
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
