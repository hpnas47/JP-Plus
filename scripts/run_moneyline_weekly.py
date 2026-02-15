#!/usr/bin/env python3
"""
Weekly Moneyline EV Runner
==========================

Three modes:
  1. Recommend (default):  Generate ML bet recommendations from an inputs CSV.
  2. Settle (--settle):    Update results after games finish.
  3. Fit sigma (--fit-sigma): Calibrate margin_sigma from backtest residuals.

Examples:
  # Recommend
  python3 scripts/run_moneyline_weekly.py --year 2025 --week 8 \\
      --inputs-path data/moneyline_selection/inputs/week8.csv

  # Settle
  python3 scripts/run_moneyline_weekly.py --settle --year 2025 --week 8 \\
      --scores-path data/moneyline_selection/scores/week8_scores.csv

  # Fit sigma
  python3 scripts/run_moneyline_weekly.py --fit-sigma \\
      --backtest-path data/backtest_results.csv --week-min 4
"""

import argparse
import sys

from src.spread_selection.moneyline_weekly import (
    fit_sigma,
    run_recommend,
    run_settle,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Weekly Moneyline EV Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode flags
    parser.add_argument("--settle", action="store_true", help="Settlement mode")
    parser.add_argument("--fit-sigma", action="store_true", help="Fit margin_sigma from backtest")

    # Common
    parser.add_argument("--year", type=int)
    parser.add_argument("--week", type=int)
    parser.add_argument("--log-path", type=str)

    # Recommend mode
    parser.add_argument("--inputs-path", type=str)
    parser.add_argument("--bankroll", type=float)
    parser.add_argument("--ev-min", type=float)
    parser.add_argument("--max-bets-per-week", type=int)
    parser.add_argument("--min-disagreement", type=float)
    parser.add_argument("--require-flip", action="store_true", default=None)
    parser.add_argument("--gate", choices=["AND", "OR"])
    parser.add_argument("--round-to", type=float)
    parser.add_argument("--rounding-mode", choices=["floor", "nearest"])
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--one-bet-per-game", action="store_true", default=None)
    grp.add_argument("--allow-multiple", action="store_true", default=None)
    parser.add_argument("--include-missing-odds-in-listB", action="store_true", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--kelly-fraction", type=float)
    parser.add_argument("--max-bet-fraction", type=float)
    parser.add_argument("--min-bet", type=float)

    # Settle mode
    parser.add_argument("--scores-path", type=str)

    # Fit-sigma mode
    parser.add_argument("--backtest-path", type=str)
    parser.add_argument("--week-min", type=int, default=4)
    parser.add_argument("--artifact-out", type=str)

    args = parser.parse_args(argv)

    # --- Determine mode ---
    if args.settle and args.fit_sigma:
        print("ERROR: --settle and --fit-sigma are mutually exclusive.", file=sys.stderr)
        return 2

    if args.fit_sigma:
        # Fit-sigma mode
        if not args.backtest_path:
            print("ERROR: --backtest-path is required in fit-sigma mode.", file=sys.stderr)
            return 2
        fit_sigma(args.backtest_path, args.week_min, args.artifact_out)
        return 0

    if args.settle:
        # Settle mode
        if args.year is None or args.week is None:
            print("ERROR: --year and --week are required in settle mode.", file=sys.stderr)
            return 2
        if not args.scores_path:
            print("ERROR: --scores-path is required in settle mode.", file=sys.stderr)
            return 2
        run_settle(args.year, args.week, args.scores_path, args.log_path)
        return 0

    # Recommend mode (default)
    if args.year is None or args.week is None:
        print("ERROR: --year and --week are required in recommend mode.", file=sys.stderr)
        return 2
    if not args.inputs_path:
        print("ERROR: --inputs-path is required in recommend mode.", file=sys.stderr)
        return 2

    # Build config overrides
    overrides = {}
    if args.bankroll is not None:
        overrides["bankroll"] = args.bankroll
    if args.ev_min is not None:
        overrides["ev_min"] = args.ev_min
    if args.max_bets_per_week is not None:
        overrides["max_bets_per_week"] = args.max_bets_per_week
    if args.min_disagreement is not None:
        overrides["min_disagreement"] = args.min_disagreement
    if args.require_flip is True:
        overrides["require_flip"] = True
    if args.gate is not None:
        overrides["gate"] = args.gate
    if args.round_to is not None:
        overrides["round_to"] = args.round_to
    if args.rounding_mode is not None:
        overrides["rounding_mode"] = args.rounding_mode
    if args.allow_multiple:
        overrides["one_bet_per_game"] = False
    elif args.one_bet_per_game:
        overrides["one_bet_per_game"] = True
    if args.include_missing_odds_in_listB:
        overrides["include_missing_odds_in_listB"] = True
    if args.kelly_fraction is not None:
        overrides["kelly_fraction"] = args.kelly_fraction
    if args.max_bet_fraction is not None:
        overrides["max_bet_fraction"] = args.max_bet_fraction
    if args.min_bet is not None:
        overrides["min_bet"] = args.min_bet

    run_recommend(
        year=args.year,
        week=args.week,
        inputs_path=args.inputs_path,
        log_path=args.log_path,
        dry_run=args.dry_run,
        **overrides,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
