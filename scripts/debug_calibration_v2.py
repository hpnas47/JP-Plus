#!/usr/bin/env python3
"""Follow-up diagnostic: per-year edge vs cover relationship.

The issue is clear: calibration trains on 2022-2023, evaluates on 2024-2025.
This checks if the edge-cover relationship differs by year.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.spread_selection.calibration import (
    load_and_normalize_game_data,
    calibrate_cover_probability,
    get_spread_bucket,
)


def main():
    print("=" * 80)
    print("PER-YEAR EDGE vs COVER ANALYSIS")
    print("=" * 80)

    # Load and normalize
    csv_path = "data/spread_selection/ats_export.csv"
    raw_df = pd.read_csv(csv_path)
    df = load_and_normalize_game_data(raw_df, jp_convention="pos_home_favored")

    # Filter to non-push, edge > 0
    df = df[~df['push'] & (df['edge_abs'] > 0)].copy()

    print(f"\nTotal games (no push, edge > 0): {len(df)}")

    # =========================================================================
    # Per-year edge >= 5 win rates
    # =========================================================================
    print("\n" + "=" * 80)
    print("PER-YEAR WIN RATES (edge_abs >= 5)")
    print("=" * 80)

    print("\n| Year | N Games | Wins | Losses | Win Rate |")
    print("|------|---------|------|--------|----------|")

    for year in sorted(df['year'].unique()):
        year_df = df[(df['year'] == year) & (df['edge_abs'] >= 5)]
        wins = year_df['jp_side_covered'].sum()
        n = len(year_df)
        losses = n - wins
        rate = wins / n if n > 0 else 0
        print(f"| {year} | {n:>7} | {wins:>4} | {losses:>6} | {rate*100:>7.1f}% |")

    # Total
    edge5_df = df[df['edge_abs'] >= 5]
    total_wins = edge5_df['jp_side_covered'].sum()
    total_n = len(edge5_df)
    print(f"| ALL  | {total_n:>7} | {total_wins:>4} | {total_n - total_wins:>6} | {total_wins/total_n*100:>7.1f}% |")

    # =========================================================================
    # Per-year calibration slopes
    # =========================================================================
    print("\n" + "=" * 80)
    print("PER-YEAR CALIBRATION (single-year fits)")
    print("=" * 80)

    print("\n| Year | N Games | Slope   | Intercept | P(c)@0 | P(c)@5 | P(c)@10 |")
    print("|------|---------|---------|-----------|--------|--------|---------|")

    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year]

        try:
            cal = calibrate_cover_probability(year_df, min_games_warn=100)
            p_at_0 = expit(cal.intercept)
            p_at_5 = expit(cal.intercept + cal.slope * 5)
            p_at_10 = expit(cal.intercept + cal.slope * 10)
            print(f"| {year} | {cal.n_games:>7} | {cal.slope:>7.4f} | {cal.intercept:>9.4f} | "
                  f"{p_at_0*100:>5.1f}% | {p_at_5*100:>5.1f}% | {p_at_10*100:>6.1f}% |")
        except Exception as e:
            print(f"| {year} | N/A     | Error: {e}")

    # Cumulative (what walk-forward training uses)
    print("\n" + "=" * 80)
    print("CUMULATIVE CALIBRATION (walk-forward training sets)")
    print("=" * 80)

    print("\n| Train Years | N Games | Slope   | Intercept | P(c)@5 |")
    print("|-------------|---------|---------|-----------|--------|")

    for end_year in [2023, 2024, 2025]:
        train_df = df[df['year'] < end_year]
        if len(train_df) < 500:
            continue

        try:
            cal = calibrate_cover_probability(train_df, min_games_warn=500)
            p_at_5 = expit(cal.intercept + cal.slope * 5)
            years_str = f"<{end_year}"
            print(f"| {years_str:<11} | {cal.n_games:>7} | {cal.slope:>7.4f} | {cal.intercept:>9.4f} | {p_at_5*100:>5.1f}% |")
        except Exception as e:
            print(f"| <{end_year}        | Error: {e}")

    # =========================================================================
    # Empirical vs calibrated by bucket, FULL dataset
    # =========================================================================
    print("\n" + "=" * 80)
    print("FULL DATASET: EMPIRICAL vs CALIBRATED BY BUCKET")
    print("=" * 80)

    # Fit on full data
    cal_full = calibrate_cover_probability(df, min_games_warn=1000)
    print(f"\nFull-data calibration: slope={cal_full.slope:.4f}, intercept={cal_full.intercept:.4f}")

    df['bucket'] = df['edge_abs'].apply(get_spread_bucket)
    df['p_cover_full'] = expit(cal_full.intercept + cal_full.slope * df['edge_abs'])

    print("\n| Bucket   | N Games | Empirical | Predicted | Delta   |")
    print("|----------|---------|-----------|-----------|---------|")

    bucket_order = ["[0,3)", "[3,5)", "[5,7)", "[7,10)", "[10,+)"]
    for bucket in bucket_order:
        bucket_df = df[df['bucket'] == bucket]
        if len(bucket_df) == 0:
            continue

        empirical = bucket_df['jp_side_covered'].mean()
        predicted = bucket_df['p_cover_full'].mean()
        delta = empirical - predicted

        print(f"| {bucket:<8} | {len(bucket_df):>7} | {empirical*100:>8.1f}% | {predicted*100:>8.1f}% | {delta*100:>+6.1f}% |")

    # =========================================================================
    # KEY INSIGHT: Training vs Eval year comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("KEY INSIGHT: TRAINING vs EVALUATION YEARS")
    print("=" * 80)

    train_years = df[df['year'].isin([2022, 2023])]
    eval_years = df[df['year'].isin([2024, 2025])]

    train_5plus = train_years[train_years['edge_abs'] >= 5]
    eval_5plus = eval_years[eval_years['edge_abs'] >= 5]

    print(f"\nTraining years (2022-2023), edge >= 5:")
    print(f"  N = {len(train_5plus)}")
    print(f"  Win rate = {train_5plus['jp_side_covered'].mean()*100:.1f}%")

    print(f"\nEvaluation years (2024-2025), edge >= 5:")
    print(f"  N = {len(eval_5plus)}")
    print(f"  Win rate = {eval_5plus['jp_side_covered'].mean()*100:.1f}%")

    diff = eval_5plus['jp_side_covered'].mean() - train_5plus['jp_side_covered'].mean()
    print(f"\nDifference: {diff*100:+.1f}% (eval - train)")

    if diff > 0.02:
        print("\n*** DIAGNOSIS: Evaluation years have HIGHER win rate than training years. ***")
        print("*** The flat calibration slope reflects the training data's weaker edge-cover relationship. ***")
        print("*** This could be: (a) model improvement, (b) random variance, or (c) regime change. ***")

    # =========================================================================
    # Check if backtest's edge calculation matches
    # =========================================================================
    print("\n" + "=" * 80)
    print("BACKTEST EDGE CALCULATION VERIFICATION")
    print("=" * 80)

    raw_with_edge = raw_df[['game_id', 'edge', 'ats_win', 'ats_push']].copy()
    our_with_edge = df[['game_id', 'edge_abs', 'jp_side_covered', 'push']].copy()

    merged = raw_with_edge.merge(our_with_edge, on='game_id')

    print(f"\nComparing 'edge' (backtest) vs 'edge_abs' (calibration module):")
    print(f"  Rows: {len(merged)}")

    # Are they the same?
    edge_match = np.abs(merged['edge'] - merged['edge_abs']) < 0.01
    print(f"  Edge values match: {edge_match.sum()}/{len(merged)} ({edge_match.mean()*100:.1f}%)")

    if not edge_match.all():
        diff_rows = merged[~edge_match]
        print(f"\n  First 5 mismatches:")
        for _, row in diff_rows.head(5).iterrows():
            print(f"    game_id={row['game_id']}: backtest edge={row['edge']:.2f}, our edge_abs={row['edge_abs']:.2f}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"""
The mismatch is explained by:

1. Walk-forward trains on 2022-2023, evaluates on 2024-2025
2. Training years (2022-2023) have ~{train_5plus['jp_side_covered'].mean()*100:.1f}% win rate at 5+ edge
3. Evaluation years (2024-2025) have ~{eval_5plus['jp_side_covered'].mean()*100:.1f}% win rate at 5+ edge
4. The calibration learns a flat slope from training data

The calibrated P(cover) is not "wrong" - it correctly reflects the
training data's edge-cover relationship. The evaluation years just
happen to have better outcomes than the training years predicted.

This is either:
- Model improvement over time (good - but we can't know prospectively)
- Random variance (concerning - 2024-2025 might regress to mean)
- Regime change (requires external explanation)

RECOMMENDATION: Use full-data calibration slope ({cal_full.slope:.4f}) for prospective
predictions, acknowledging that 2022-2023 had weaker edge-cover relationship.
""")


if __name__ == "__main__":
    main()
