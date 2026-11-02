"""Command line entry point: ``uv run twrank <command>``.

Commands, in pipeline order::

    fetch      download the stock list and prices (resumable), build panels, data.lock
    panels     rebuild wide panels and data.lock from the cached prices
    lock       rewrite data.lock and data.manifest.tsv
    features   build the feature/label table
    train      rolling LightGBM refits -> out-of-sample scores
    run        backtest the experiments -> results/exp/<id>/
    wfv        per-stage parameter selection on training windows -> results/exp/wfv/
    stacked    stacked equity curve png per experiment
    viewer     export docs/viewer data
    report     results/report.md and results/equity.png
    all        features, train, run, wfv, stacked, viewer, report
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
    from .stacked import draw
    from .universe import name_map

    from . import experiments as X

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


def _cmd_all(args) -> None:
    for fn in (_cmd_features, _cmd_train, _cmd_run, _cmd_wfv, _cmd_stacked, _cmd_viewer,
               _cmd_report):
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
    sub.add_parser("viewer", help="export docs/viewer").set_defaults(func=_cmd_viewer)
    sub.add_parser("report", help="results/report.md").set_defaults(func=_cmd_report)
    return ap


def main(argv: list[str] | None = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)  # argparse rejects unknown arguments
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
