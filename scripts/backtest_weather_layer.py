#!/usr/bin/env python3
"""
Backtest Weather Layer for Totals Model.

Compares betting performance with and without weather adjustments:
- Scenario A: mu_used = mu_model (no weather)
- Scenario B: mu_used = mu_model + weather_adj

If historical weather adjustments are not available, the script outputs
diagnostic information and gracefully skips the comparison.

Usage:
    python3 scripts/backtest_weather_layer.py
    python3 scripts/backtest_weather_layer.py --years 2024 2025
    python3 scripts/backtest_weather_layer.py --weather-csv data/historical_weather.csv
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
from src.spread_selection.totals_calibration import (
    collect_walk_forward_residuals,
    estimate_sigma_global,
    evaluate_interval_coverage,
    backtest_ev_roi,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backtest_totals_with_weather(
    client: CFBDClient,
    year: int,
    weather_df: pd.DataFrame | None,
    ridge_alpha: float = 10.0,
    start_week: int = 4,
) -> list:
    """Backtest totals for a single season with optional weather.

    Args:
        client: CFBD API client
        year: Season year
        weather_df: DataFrame with columns: game_id, weather_adj
            If None or empty, weather_adj defaults to 0.0
        ridge_alpha: Ridge regression regularization
        start_week: First week to predict

    Returns:
        List of prediction dicts with both mu_model and weather_adj
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
    ].copy()

    games['year'] = year

    # Merge betting data
    if 'over_under' in betting.columns:
        cols = ['game_id', 'over_under']
        if 'over_under_open' in betting.columns:
            cols.append('over_under_open')
        betting_slim = betting[cols].drop_duplicates()
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

    # Merge weather adjustments (if available)
    if weather_df is not None and len(weather_df) > 0:
        # Expect columns: game_id, weather_adj
        weather_slim = weather_df[['game_id', 'weather_adj']].copy()
        weather_slim = weather_slim.rename(columns={'game_id': 'id'})
        games = games.merge(weather_slim, on='id', how='left')
        games['weather_adj'] = games['weather_adj'].fillna(0.0)
    else:
        games['weather_adj'] = 0.0

    max_week = int(games['week'].max())
    predictions = []

    # Create model
    model = TotalsModel(ridge_alpha=ridge_alpha)
    model.set_team_universe(fbs_set)

    for pred_week in range(start_week, max_week + 1):
        n_train = (games['week'] < pred_week).sum()
        if n_train < 50:
            continue

        model.train(games, fbs_set, max_week=pred_week - 1)

        if not model._trained:
            continue

        week_games = games[games['week'] == pred_week]

        for g in week_games.itertuples():
            pred = model.predict_total(g.home_team, g.away_team, year=year)

            if pred:
                actual_total = g.home_points + g.away_points
                weather_adj = getattr(g, 'weather_adj', 0.0)

                predictions.append({
                    'year': year,
                    'week': pred_week,
                    'game_id': g.id,
                    'home_team': g.home_team,
                    'away_team': g.away_team,
                    # Mu composition
                    'mu_model': pred.predicted_total,
                    'weather_adj': weather_adj,
                    'mu_used': pred.predicted_total + weather_adj,
                    # Outcomes
                    'actual_total': actual_total,
                    'vegas_total_close': getattr(g, 'vegas_total_close', np.nan),
                    'vegas_total_open': getattr(g, 'vegas_total_open', np.nan),
                })

    logger.info(f"  {year}: {len(predictions)} predictions")
    return predictions


def compute_sanity_diagnostics(preds_df: pd.DataFrame) -> dict:
    """Compute sanity diagnostics for weather adjustment validation.

    Args:
        preds_df: DataFrame with mu_model, weather_adj, actual_total

    Returns:
        Dict with diagnostic metrics
    """
    df = preds_df.copy()

    # Filter to games with weather adjustments
    has_weather = df['weather_adj'].abs() > 0.1
    n_with_weather = has_weather.sum()

    if n_with_weather < 10:
        return {
            "n_games_with_weather": n_with_weather,
            "insufficient_data": True,
            "message": "Not enough games with weather adjustments for diagnostics"
        }

    weather_games = df[has_weather].copy()

    # Error before weather adjustment
    weather_games['error_model_only'] = weather_games['mu_model'] - weather_games['actual_total']

    # Error after weather adjustment
    weather_games['error_with_weather'] = weather_games['mu_used'] - weather_games['actual_total']

    # Correlation: weather_adj vs model error
    # If weather_adj is meaningful:
    # - Negative weather_adj (lower scoring) should correlate with NEGATIVE model error
    #   (model over-predicted before weather adjustment was applied)
    # Positive correlation = weather_adj is directionally correct
    correlation = weather_games['weather_adj'].corr(weather_games['error_model_only'])

    # Variance reduction: does weather help?
    var_before = weather_games['error_model_only'].var()
    var_after = weather_games['error_with_weather'].var()
    variance_reduction_pct = (var_before - var_after) / var_before * 100 if var_before > 0 else 0

    # MAE comparison
    mae_model = weather_games['error_model_only'].abs().mean()
    mae_weather = weather_games['error_with_weather'].abs().mean()
    mae_improvement = mae_model - mae_weather

    # Over/Under skew
    n_over = (weather_games['weather_adj'] > 0).sum()
    n_under = (weather_games['weather_adj'] < 0).sum()

    return {
        "n_games_with_weather": n_with_weather,
        "insufficient_data": False,
        # Correlation diagnostic
        "weather_adj_vs_model_error_corr": round(correlation, 4),
        "correlation_interpretation": (
            "GOOD (weather directionally correct)" if correlation > 0.1 else
            "NEUTRAL" if abs(correlation) <= 0.1 else
            "BAD (weather wrong sign)"
        ),
        # Variance reduction
        "variance_before_weather": round(var_before, 2),
        "variance_after_weather": round(var_after, 2),
        "variance_reduction_pct": round(variance_reduction_pct, 2),
        # MAE comparison
        "mae_model_only": round(mae_model, 2),
        "mae_with_weather": round(mae_weather, 2),
        "mae_improvement": round(mae_improvement, 2),
        # Skew
        "n_over_adjustments": int(n_over),
        "n_under_adjustments": int(n_under),
        "over_under_ratio": round(n_over / n_under, 2) if n_under > 0 else float('inf'),
    }


def compare_scenarios(preds_df: pd.DataFrame, sigma: float = 13.0) -> dict:
    """Compare Scenario A (no weather) vs Scenario B (with weather).

    Args:
        preds_df: DataFrame with mu_model, weather_adj, mu_used, actual_total, vegas_total_close
        sigma: Sigma for probability calculations

    Returns:
        Dict with comparison metrics
    """
    # Scenario A: mu_used = mu_model (no weather)
    df_a = preds_df.copy()
    df_a['adjusted_total'] = df_a['mu_model']  # For ROI backtest
    df_a['predicted_total'] = df_a['mu_model']

    # Scenario B: mu_used = mu_model + weather_adj
    df_b = preds_df.copy()
    df_b['adjusted_total'] = df_b['mu_used']
    df_b['predicted_total'] = df_b['mu_used']

    # Collect residuals for each scenario
    residuals_a = collect_walk_forward_residuals(df_a, calibration_mode="model_only")
    residuals_b = collect_walk_forward_residuals(df_b, calibration_mode="model_only")  # Same mode, different mu

    # Sigma estimates
    sigma_a = estimate_sigma_global(residuals_a).sigma
    sigma_b = estimate_sigma_global(residuals_b).sigma

    # Coverage evaluation
    coverage_a = evaluate_interval_coverage(residuals_a, sigma)
    coverage_b = evaluate_interval_coverage(residuals_b, sigma)

    # ROI backtest (if lines available)
    roi_a = backtest_ev_roi(df_a, sigma)
    roi_b = backtest_ev_roi(df_b, sigma)

    return {
        "scenario_a_no_weather": {
            "sigma_fit": round(sigma_a, 2),
            "mae": round(residuals_a['abs_error'].mean(), 2),
            "mean_error": round(residuals_a['error'].mean(), 2),
            "coverage": {f"{c.target_coverage:.0%}": round(c.empirical_coverage, 3) for c in coverage_a},
            "roi": roi_a if "error" not in roi_a else {"error": roi_a["error"]},
        },
        "scenario_b_with_weather": {
            "sigma_fit": round(sigma_b, 2),
            "mae": round(residuals_b['abs_error'].mean(), 2),
            "mean_error": round(residuals_b['error'].mean(), 2),
            "coverage": {f"{c.target_coverage:.0%}": round(c.empirical_coverage, 3) for c in coverage_b},
            "roi": roi_b if "error" not in roi_b else {"error": roi_b["error"]},
        },
        "comparison": {
            "sigma_diff": round(sigma_b - sigma_a, 3),
            "mae_diff": round(residuals_b['abs_error'].mean() - residuals_a['abs_error'].mean(), 3),
            "weather_helps_mae": residuals_b['abs_error'].mean() < residuals_a['abs_error'].mean(),
        }
    }


def print_comparison_report(comparison: dict, sanity: dict) -> None:
    """Print formatted comparison report."""
    print("\n" + "=" * 80)
    print("WEATHER LAYER BACKTEST RESULTS")
    print("=" * 80)

    # Sanity diagnostics
    print("\n--- SANITY DIAGNOSTICS ---")
    if sanity.get("insufficient_data"):
        print(f"  {sanity['message']}")
        print(f"  Games with weather adjustments: {sanity['n_games_with_weather']}")
    else:
        print(f"  Games with weather adjustments: {sanity['n_games_with_weather']}")
        print(f"  Weather vs Model Error correlation: {sanity['weather_adj_vs_model_error_corr']:+.4f}")
        print(f"  Interpretation: {sanity['correlation_interpretation']}")
        print(f"  Variance reduction: {sanity['variance_reduction_pct']:+.2f}%")
        print(f"  MAE improvement: {sanity['mae_improvement']:+.2f} pts")
        print(f"  Over/Under adjustments: {sanity['n_over_adjustments']}/{sanity['n_under_adjustments']}")

    # Scenario comparison
    print("\n--- SCENARIO COMPARISON ---")
    print(f"{'Metric':<25} {'A: No Weather':<18} {'B: With Weather':<18} {'Delta':<12}")
    print("-" * 80)

    a = comparison['scenario_a_no_weather']
    b = comparison['scenario_b_with_weather']
    c = comparison['comparison']

    print(f"{'Sigma (fit)':<25} {a['sigma_fit']:<18.2f} {b['sigma_fit']:<18.2f} {c['sigma_diff']:+.3f}")
    print(f"{'MAE':<25} {a['mae']:<18.2f} {b['mae']:<18.2f} {c['mae_diff']:+.3f}")
    print(f"{'Mean Error':<25} {a['mean_error']:<+18.2f} {b['mean_error']:<+18.2f} {b['mean_error']-a['mean_error']:+.3f}")

    # Coverage
    print("\n--- INTERVAL COVERAGE ---")
    print(f"{'Target':<10} {'A: No Weather':<18} {'B: With Weather':<18}")
    print("-" * 50)
    for target in ['50%', '68%', '80%', '90%', '95%']:
        cov_a = a['coverage'].get(target, 'N/A')
        cov_b = b['coverage'].get(target, 'N/A')
        print(f"{target:<10} {cov_a:<18} {cov_b:<18}")

    # ROI (if available)
    print("\n--- ROI BACKTEST ---")
    if "error" in a['roi']:
        print(f"  Scenario A: {a['roi']['error']}")
    else:
        print(f"  Scenario A: {a['roi'].get('n_bets', 0)} bets, ROI={a['roi'].get('roi', 0)*100:+.2f}%, "
              f"Win={a['roi'].get('win_rate', 0)*100:.1f}%")

    if "error" in b['roi']:
        print(f"  Scenario B: {b['roi']['error']}")
    else:
        print(f"  Scenario B: {b['roi'].get('n_bets', 0)} bets, ROI={b['roi'].get('roi', 0)*100:+.2f}%, "
              f"Win={b['roi'].get('win_rate', 0)*100:.1f}%")

    # Summary
    print("\n--- SUMMARY ---")
    if c['weather_helps_mae']:
        print("  ✅ Weather layer REDUCES prediction error (MAE)")
    else:
        print("  ❌ Weather layer INCREASES prediction error (MAE)")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Backtest weather layer impact on totals')
    parser.add_argument('--years', type=int, nargs='+', default=[2023, 2024, 2025],
                        help='Years to backtest')
    parser.add_argument('--start-week', type=int, default=4,
                        help='First week to predict')
    parser.add_argument('--alpha', type=float, default=10.0,
                        help='Ridge alpha')
    parser.add_argument('--weather-csv', type=str, default=None,
                        help='Path to CSV with historical weather adjustments (game_id, weather_adj)')
    parser.add_argument('--sigma', type=float, default=13.0,
                        help='Sigma for probability calculations')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV for predictions')
    args = parser.parse_args()

    client = CFBDClient()

    # Load weather data if provided
    weather_df = None
    if args.weather_csv:
        weather_path = Path(args.weather_csv)
        if weather_path.exists():
            weather_df = pd.read_csv(weather_path)
            logger.info(f"Loaded {len(weather_df)} weather adjustments from {weather_path}")
        else:
            logger.warning(f"Weather CSV not found: {weather_path}")

    # Collect predictions across all years
    all_predictions = []
    for year in args.years:
        # Filter weather to this year if available
        year_weather = None
        if weather_df is not None:
            # Assume game_id encodes year or we have a year column
            if 'year' in weather_df.columns:
                year_weather = weather_df[weather_df['year'] == year]
            else:
                year_weather = weather_df

        preds = backtest_totals_with_weather(
            client, year, year_weather,
            ridge_alpha=args.alpha,
            start_week=args.start_week,
        )
        all_predictions.extend(preds)

    if not all_predictions:
        logger.error("No predictions generated")
        return

    preds_df = pd.DataFrame(all_predictions)

    # Save predictions if requested
    if args.output:
        preds_df.to_csv(args.output, index=False)
        logger.info(f"Saved {len(preds_df)} predictions to {args.output}")

    # Compute sanity diagnostics
    sanity = compute_sanity_diagnostics(preds_df)

    # Compare scenarios
    comparison = compare_scenarios(preds_df, sigma=args.sigma)

    # Print report
    print_comparison_report(comparison, sanity)

    # Check if historical weather was available
    has_weather = preds_df['weather_adj'].abs().sum() > 0
    if not has_weather:
        print("\n" + "=" * 80)
        print("⚠️  NO HISTORICAL WEATHER DATA AVAILABLE")
        print("=" * 80)
        print("""
The weather layer backtest requires historical weather adjustments to compare
scenarios. Since no historical weather data was found, Scenario A and B are
identical (weather_adj = 0 for all games).

To run a meaningful comparison:
1. Create a CSV with columns: game_id, weather_adj (optionally: year)
2. Run: python3 scripts/backtest_weather_layer.py --weather-csv data/historical_weather.csv

For 2026+ production, weather adjustments will be captured by weather_thursday_capture.py
and stored for future backtesting.
""")


if __name__ == '__main__':
    main()
