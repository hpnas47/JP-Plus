#!/usr/bin/env python3
"""
Backtest totals model against Vegas over/unders.

Walk-forward validation: for each week W, trains on weeks 1 to W-1 and predicts week W.

Usage:
    python3 scripts/backtest_totals.py
    python3 scripts/backtest_totals.py --years 2024 2025
    python3 scripts/backtest_totals.py --start-week 1 --alpha 10.0
"""

import argparse
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_phase(week: int) -> str:
    """Assign phase based on week."""
    if week <= 3:
        return 'Phase 1 (Calibration)'
    elif week <= 15:
        return 'Phase 2 (Core)'
    else:
        return 'Phase 3 (Postseason)'


def backtest_totals_season(
    client: CFBDClient,
    year: int,
    start_week: int = 4,
    ridge_alpha: float = 10.0,
) -> list:
    """Backtest totals for a single season.

    Args:
        client: CFBD API client
        year: Season year
        start_week: First week to predict (need training data from earlier weeks)
        ridge_alpha: Ridge regression regularization strength

    Returns:
        List of prediction dicts
    """
    logger.info(f"Backtesting {year}...")

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
    ]

    # Merge betting data for over/under
    if 'over_under' in betting.columns or 'total' in betting.columns:
        ou_col = 'over_under' if 'over_under' in betting.columns else 'total'
        betting_slim = betting[['game_id', ou_col]].drop_duplicates()
        betting_slim = betting_slim.rename(columns={ou_col: 'vegas_total', 'game_id': 'id'})
        games = games.merge(betting_slim, on='id', how='left')
    else:
        games['vegas_total'] = np.nan

    max_week = int(games['week'].max())
    predictions = []

    for pred_week in range(start_week, max_week + 1):
        # Train on weeks < pred_week (walk-forward)
        model = TotalsModel(ridge_alpha=ridge_alpha)
        train_games = games[games['week'] < pred_week]

        if len(train_games) < 50:
            continue

        model.train(train_games, fbs_set, max_week=pred_week - 1)

        if not model._trained:
            continue

        # Predict games in pred_week
        week_games = games[games['week'] == pred_week]

        for _, g in week_games.iterrows():
            pred = model.predict_total(g['home_team'], g['away_team'])
            if pred:
                actual_total = g['home_points'] + g['away_points']
                vegas_total = g.get('vegas_total')

                predictions.append({
                    'year': year,
                    'week': pred_week,
                    'phase': get_phase(pred_week),
                    'game_id': g['id'],
                    'home_team': g['home_team'],
                    'away_team': g['away_team'],
                    'predicted_total': pred.predicted_total,
                    'home_expected': pred.home_expected,
                    'away_expected': pred.away_expected,
                    'actual_total': actual_total,
                    'vegas_total': vegas_total,
                    'jp_error': pred.predicted_total - actual_total,
                    'jp_abs_error': abs(pred.predicted_total - actual_total),
                })

    logger.info(f"  {year}: {len(predictions)} predictions")
    return predictions


def calculate_ou_ats(preds_df: pd.DataFrame, edge_min: float = 0) -> tuple:
    """Calculate over/under ATS performance.

    Args:
        preds_df: DataFrame with predicted_total, vegas_total, actual_total
        edge_min: Minimum edge (abs difference from Vegas) to include

    Returns:
        Tuple of (wins, losses, pushes)
    """
    valid = preds_df[preds_df['vegas_total'].notna()].copy()
    valid['edge'] = abs(valid['predicted_total'] - valid['vegas_total'])
    valid = valid[valid['edge'] >= edge_min]

    wins, losses, pushes = 0, 0, 0

    for _, r in valid.iterrows():
        jp_total = r['predicted_total']
        vegas_total = r['vegas_total']
        actual = r['actual_total']

        # JP+ pick: over if jp > vegas, under if jp < vegas
        jp_says_over = jp_total > vegas_total

        # Result
        if actual == vegas_total:
            pushes += 1
        elif jp_says_over:
            if actual > vegas_total:
                wins += 1
            else:
                losses += 1
        else:  # JP+ says under
            if actual < vegas_total:
                wins += 1
            else:
                losses += 1

    return wins, losses, pushes


def format_ats(w: int, l: int) -> str:
    """Format ATS record as 'W-L (pct%)'."""
    total = w + l
    if total == 0:
        return "N/A"
    pct = w / total * 100
    return f"{w}-{l} ({pct:.1f}%)"


def main():
    parser = argparse.ArgumentParser(description='Backtest totals model')
    parser.add_argument('--years', type=int, nargs='+', default=[2022, 2023, 2024, 2025],
                        help='Years to backtest')
    parser.add_argument('--start-week', type=int, default=1,
                        help='First week to predict (default: 1)')
    parser.add_argument('--alpha', type=float, default=10.0,
                        help='Ridge alpha (default: 10.0)')
    args = parser.parse_args()

    client = CFBDClient()

    # Run backtest for each year
    all_predictions = []
    for year in args.years:
        preds = backtest_totals_season(
            client, year,
            start_week=args.start_week,
            ridge_alpha=args.alpha
        )
        all_predictions.extend(preds)

    if not all_predictions:
        logger.error("No predictions generated")
        return

    preds_df = pd.DataFrame(all_predictions)

    # Print results
    print("\n" + "=" * 80)
    print("JP+ TOTALS MODEL BACKTEST RESULTS")
    print("=" * 80)
    print(f"Years: {args.years}")
    print(f"Start Week: {args.start_week}")
    print(f"Ridge Alpha: {args.alpha}")

    # Overall metrics
    print(f"\nTotal predictions: {len(preds_df)}")
    print(f"Mean Error (bias): {preds_df['jp_error'].mean():+.2f}")
    print(f"MAE: {preds_df['jp_abs_error'].mean():.2f}")
    print(f"RMSE: {np.sqrt((preds_df['jp_error']**2).mean()):.2f}")

    # Vegas comparison
    valid_ou = preds_df[preds_df['vegas_total'].notna()].copy()
    if len(valid_ou) > 0:
        valid_ou['vegas_error'] = valid_ou['vegas_total'] - valid_ou['actual_total']
        valid_ou['vegas_abs_error'] = abs(valid_ou['vegas_error'])
        print(f"\nGames with Vegas O/U: {len(valid_ou)}")
        print(f"Vegas MAE: {valid_ou['vegas_abs_error'].mean():.2f}")
        print(f"JP+ MAE: {valid_ou['jp_abs_error'].mean():.2f}")

    # Phase breakdown
    print("\n" + "-" * 80)
    print("PERFORMANCE BY PHASE")
    print("-" * 80)
    print(f"{'Phase':<25} {'Weeks':<8} {'Games':<8} {'MAE':<8} {'ATS %':<10} {'3+ Edge':<15} {'5+ Edge':<15}")
    print("-" * 80)

    phases = [
        ('Phase 1 (Calibration)', '1-3'),
        ('Phase 2 (Core)', '4-15'),
        ('Phase 3 (Postseason)', '16+'),
    ]

    for phase, weeks in phases:
        sub = preds_df[preds_df['phase'] == phase]
        if len(sub) == 0:
            continue

        mae = sub['jp_abs_error'].mean()
        w0, l0, _ = calculate_ou_ats(sub, edge_min=0)
        w3, l3, _ = calculate_ou_ats(sub, edge_min=3)
        w5, l5, _ = calculate_ou_ats(sub, edge_min=5)

        ats0 = w0 / (w0 + l0) * 100 if (w0 + l0) > 0 else 0

        print(f"{phase:<25} {weeks:<8} {len(sub):<8} {mae:<8.2f} {ats0:<10.1f} {format_ats(w3, l3):<15} {format_ats(w5, l5):<15}")

    # Full season totals
    print("-" * 80)
    mae = preds_df['jp_abs_error'].mean()
    w0, l0, _ = calculate_ou_ats(preds_df, edge_min=0)
    w3, l3, _ = calculate_ou_ats(preds_df, edge_min=3)
    w5, l5, _ = calculate_ou_ats(preds_df, edge_min=5)
    ats0 = w0 / (w0 + l0) * 100 if (w0 + l0) > 0 else 0
    print(f"{'Full Season':<25} {'All':<8} {len(preds_df):<8} {mae:<8.2f} {ats0:<10.1f} {format_ats(w3, l3):<15} {format_ats(w5, l5):<15}")

    # Year breakdown (Core phase only)
    print("\n" + "-" * 80)
    print("CORE PHASE BY YEAR (Weeks 4-15)")
    print("-" * 80)
    print(f"{'Year':<8} {'Games':<8} {'MAE':<8} {'ATS %':<10} {'3+ Edge':<15} {'5+ Edge':<15}")
    print("-" * 80)

    core = preds_df[preds_df['phase'] == 'Phase 2 (Core)']
    for year in args.years:
        yr = core[core['year'] == year]
        if len(yr) == 0:
            continue

        mae = yr['jp_abs_error'].mean()
        w0, l0, _ = calculate_ou_ats(yr, edge_min=0)
        w3, l3, _ = calculate_ou_ats(yr, edge_min=3)
        w5, l5, _ = calculate_ou_ats(yr, edge_min=5)
        ats0 = w0 / (w0 + l0) * 100 if (w0 + l0) > 0 else 0

        print(f"{year:<8} {len(yr):<8} {mae:<8.2f} {ats0:<10.1f} {format_ats(w3, l3):<15} {format_ats(w5, l5):<15}")


if __name__ == '__main__':
    main()
