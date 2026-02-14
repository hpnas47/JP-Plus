#!/usr/bin/env python3
"""Calibrate Totals EV Engine using historical walk-forward data.

This script:
1. Runs walk-forward backtest to collect residuals (or loads from existing CSV)
2. Computes multiple sigma estimates
3. Evaluates interval coverage and (if lines available) ROI
4. Tunes parameters via grid search
5. Outputs calibration report and recommended config JSON

Usage:
    # Run full calibration (2023-2025, recommended)
    python3 scripts/calibrate_totals_engine.py

    # Use existing predictions CSV (faster iteration)
    python3 scripts/calibrate_totals_engine.py --from-csv data/totals_preds.csv

    # Include 2022 (transition year, optional)
    python3 scripts/calibrate_totals_engine.py --years 2022 2023 2024 2025

    # Custom sigma range
    python3 scripts/calibrate_totals_engine.py --sigma-min 10.0 --sigma-max 18.0 --sigma-step 0.5
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.cfbd_client import CFBDClient
from src.models.totals_model import TotalsModel
from scripts.backtest import fetch_season_data
from src.spread_selection.totals_calibration import (
    collect_walk_forward_residuals,
    compute_all_sigma_estimates,
    evaluate_interval_coverage,
    compute_coverage_score,
    tune_sigma_for_coverage,
    tune_sigma_for_roi,
    backtest_ev_roi,
    compute_week_bucket_multipliers,
    run_full_calibration,
    save_calibration,
    TotalsCalibrationConfig,
    CalibrationReport,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_walk_forward_backtest(
    years: list[int],
    start_week: int = 1,
    ridge_alpha: float = 10.0,
) -> pd.DataFrame:
    """Run walk-forward backtest to generate predictions.

    Same logic as backtest_totals.py but returns DataFrame directly.
    """
    client = CFBDClient()
    all_predictions = []

    for year in years:
        logger.info(f"Processing {year}...")

        # Fetch data
        games_df, betting_df = fetch_season_data(client, year)
        fbs_teams = client.get_fbs_teams(year=year)
        fbs_set = {t.school for t in fbs_teams if t.school}

        # Convert to pandas
        games = games_df.to_pandas()
        betting = betting_df.to_pandas() if hasattr(betting_df, 'to_pandas') else betting_df

        # Filter to FBS vs FBS with scores
        games = games[
            games['home_team'].isin(fbs_set) &
            games['away_team'].isin(fbs_set) &
            games['home_points'].notna() &
            games['away_points'].notna()
        ].copy()

        # Add year column
        games['year'] = year

        # Merge betting data for over/under
        if 'over_under' in betting.columns:
            cols_to_get = ['game_id', 'over_under']
            if 'over_under_open' in betting.columns:
                cols_to_get.append('over_under_open')
            betting_slim = betting[cols_to_get].drop_duplicates()
            betting_slim = betting_slim.rename(columns={
                'over_under': 'vegas_total_close',
                'over_under_open': 'vegas_total_open',
                'game_id': 'id'
            })
            games = games.merge(betting_slim, on='id', how='left')
            if 'vegas_total_open' not in games.columns:
                games['vegas_total_open'] = games['vegas_total_close']
        else:
            games['vegas_total_close'] = np.nan
            games['vegas_total_open'] = np.nan

        max_week = int(games['week'].max())

        # Walk-forward: for each week, train on previous weeks
        for pred_week in range(start_week, max_week + 1):
            # Skip if no games in this week
            week_games = games[games['week'] == pred_week]
            if len(week_games) == 0:
                continue

            # Train on weeks before pred_week
            train_games = games[games['week'] < pred_week].copy()
            if len(train_games) < 10:
                continue

            # Initialize and train model
            model = TotalsModel(ridge_alpha=ridge_alpha)
            model.train(train_games, fbs_set, max_week=pred_week - 1, year=year)

            # Predict each game in pred_week
            for _, g in week_games.iterrows():
                pred = model.predict_total(g.home_team, g.away_team, year=year)

                if pred:
                    actual_total = g.home_points + g.away_points
                    vegas_total_close = getattr(g, 'vegas_total_close', None)
                    vegas_total_open = getattr(g, 'vegas_total_open', None)

                    # Get team reliability info
                    home_rating = model.team_ratings.get(g.home_team)
                    away_rating = model.team_ratings.get(g.away_team)
                    home_games = home_rating.games_played if home_rating else 0
                    away_games = away_rating.games_played if away_rating else 0

                    all_predictions.append({
                        'year': year,
                        'week': pred_week,
                        'home_team': g.home_team,
                        'away_team': g.away_team,
                        'predicted_total': pred.predicted_total,
                        'adjusted_total': pred.adjusted_total,
                        'home_expected': pred.home_expected,
                        'away_expected': pred.away_expected,
                        'actual_total': actual_total,
                        'vegas_total_close': vegas_total_close,
                        'vegas_total_open': vegas_total_open,
                        'home_games_played': home_games,
                        'away_games_played': away_games,
                        'model_baseline': model.baseline,
                        'n_train_games': len(train_games),
                    })

    return pd.DataFrame(all_predictions)


def print_calibration_report(report: CalibrationReport, preds_df: pd.DataFrame):
    """Print formatted calibration report."""
    print("\n" + "=" * 80)
    print("TOTALS EV ENGINE CALIBRATION REPORT")
    print("=" * 80)

    # Dataset summary
    print(f"\nDataset: {len(preds_df)} games across {sorted(preds_df['year'].unique())}")
    n_with_lines = preds_df['vegas_total_close'].notna().sum()
    print(f"Games with O/U lines: {n_with_lines} ({n_with_lines/len(preds_df)*100:.1f}%)")

    # Sigma estimates
    print("\n" + "-" * 80)
    print("SIGMA ESTIMATES")
    print("-" * 80)
    print(f"{'Method':<35} {'Sigma':>8} {'N Games':>10}")
    print("-" * 55)

    for est in report.sigma_estimates:
        if est.name.startswith('week_bucket_') or est.name.startswith('phase_'):
            print(f"  {est.name:<33} {est.sigma:>8.2f} {est.n_games:>10}")
        else:
            print(f"{est.name:<35} {est.sigma:>8.2f} {est.n_games:>10}")

    # Coverage analysis
    print("\n" + "-" * 80)
    print("INTERVAL COVERAGE BY SIGMA")
    print("-" * 80)
    print(f"{'Sigma':>6} {'50% Cov':>10} {'68% Cov':>10} {'80% Cov':>10} {'90% Cov':>10} {'Score':>10}")
    print("-" * 58)

    for sigma in sorted(report.coverage_results.keys()):
        results = report.coverage_results[sigma]
        row = f"{sigma:>6.1f}"
        for r in results[:4]:  # First 4 targets
            cov_str = f"{r.empirical_coverage*100:>5.1f}%"
            if abs(r.error) > 0.05:
                cov_str += "*"
            else:
                cov_str += " "
            row += f" {cov_str:>9}"
        score = compute_coverage_score(results)
        row += f" {score:>9.4f}"
        if sigma == report.best_sigma:
            row += " <-- BEST"
        print(row)

    # ROI analysis (if available)
    if report.roi_by_sigma:
        print("\n" + "-" * 80)
        print("ROI BACKTEST BY SIGMA")
        print("-" * 80)
        print(f"{'Sigma':>6} {'N Bets':>8} {'Win%':>8} {'ROI':>10} {'CapHit%':>10} {'MeanEV':>10}")
        print("-" * 55)

        for sigma_str in sorted(report.roi_by_sigma.keys(), key=float):
            sigma = float(sigma_str)
            roi_data = report.roi_by_sigma[sigma_str]
            if "error" in roi_data:
                print(f"{sigma:>6.1f} {'--':>8} {'--':>8} {'--':>10} {'--':>10} {'--':>10}")
            else:
                print(f"{sigma:>6.1f} {roi_data['n_bets']:>8} "
                      f"{roi_data['win_rate']*100:>7.1f}% {roi_data['roi']*100:>9.2f}% "
                      f"{roi_data['cap_hit_rate']*100:>9.1f}% {roi_data['mean_ev']*100:>9.2f}%")

    # Best configuration
    print("\n" + "-" * 80)
    print("RECOMMENDED CONFIGURATION")
    print("-" * 80)
    config = report.recommended_config
    print(f"Best sigma: {report.best_sigma:.2f} ({report.best_sigma_method})")
    print(f"Sigma mode: {config.sigma_mode}")
    print(f"\nWeek bucket multipliers (relative to base sigma):")
    for bucket, mult in sorted(config.week_bucket_multipliers.items()):
        effective_sigma = report.best_sigma * mult
        print(f"  {bucket}: x{mult:.2f} -> sigma={effective_sigma:.1f}")

    # Production snippet
    print("\n" + "-" * 80)
    print("PRODUCTION CONFIG SNIPPET (copy to TotalsEVConfig)")
    print("-" * 80)
    print(f"""
# From calibration on {config.years_used}, {config.n_games_calibrated} games
sigma_total={report.best_sigma:.1f},
sigma_mode="{config.sigma_mode}",

# Week bucket multipliers (if using sigma_mode="week_bucket")
week_bucket_multipliers={{
    "1-2": {config.week_bucket_multipliers.get("1-2", 1.0):.2f},
    "3-5": {config.week_bucket_multipliers.get("3-5", 1.0):.2f},
    "6-9": {config.week_bucket_multipliers.get("6-9", 1.0):.2f},
    "10-14": {config.week_bucket_multipliers.get("10-14", 1.0):.2f},
    "15+": {config.week_bucket_multipliers.get("15+", 1.0):.2f},
}}
""")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate Totals EV Engine parameters"
    )
    parser.add_argument(
        "--years", type=int, nargs="+",
        default=[2023, 2024, 2025],
        help="Years to include in calibration (default: 2023-2025)"
    )
    parser.add_argument(
        "--start-week", type=int, default=1,
        help="First week to include (default: 1)"
    )
    parser.add_argument(
        "--alpha", type=float, default=10.0,
        help="Ridge alpha for TotalsModel (default: 10.0)"
    )
    parser.add_argument(
        "--from-csv", type=str, default=None,
        help="Load predictions from CSV instead of running backtest"
    )
    parser.add_argument(
        "--output-csv", type=str, default="data/totals_calibration_preds.csv",
        help="Save predictions to CSV for faster iteration"
    )
    parser.add_argument(
        "--sigma-min", type=float, default=10.0,
        help="Minimum sigma to try (default: 10.0)"
    )
    parser.add_argument(
        "--sigma-max", type=float, default=20.0,
        help="Maximum sigma to try (default: 20.0)"
    )
    parser.add_argument(
        "--sigma-step", type=float, default=0.5,
        help="Sigma step size (default: 0.5)"
    )
    parser.add_argument(
        "--output-config", type=str,
        default="artifacts/totals_calibration_2022_2025.json",
        help="Output path for calibration JSON"
    )

    args = parser.parse_args()

    # Generate sigma candidates
    n_steps = int((args.sigma_max - args.sigma_min) / args.sigma_step) + 1
    sigma_candidates = [args.sigma_min + i * args.sigma_step for i in range(n_steps)]

    # Get predictions
    if args.from_csv:
        logger.info(f"Loading predictions from {args.from_csv}")
        preds_df = pd.read_csv(args.from_csv)
    else:
        logger.info(f"Running walk-forward backtest for {args.years}...")
        preds_df = run_walk_forward_backtest(
            years=args.years,
            start_week=args.start_week,
            ridge_alpha=args.alpha,
        )

        # Save for faster iteration
        if args.output_csv:
            Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
            preds_df.to_csv(args.output_csv, index=False)
            logger.info(f"Saved predictions to {args.output_csv}")

    if len(preds_df) == 0:
        logger.error("No predictions generated!")
        sys.exit(1)

    logger.info(f"Collected {len(preds_df)} predictions")

    # Run full calibration
    logger.info("Running calibration...")
    report = run_full_calibration(preds_df, sigma_candidates)

    # Print report
    print_calibration_report(report, preds_df)

    # Save calibration config
    output_path = Path(args.output_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_calibration(report.recommended_config, output_path)
    print(f"\nSaved calibration config to: {output_path}")

    # Also save full report as JSON
    report_path = output_path.with_suffix('.report.json')
    with open(report_path, 'w') as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    print(f"Saved full report to: {report_path}")


if __name__ == "__main__":
    main()
