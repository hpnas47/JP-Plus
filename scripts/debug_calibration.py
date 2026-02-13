#!/usr/bin/env python3
"""Debug script to diagnose calibration vs strategy evaluation mismatch.

The issue: OLD_5pt shows 55.4% on 653 bets, but calibrated P(cover) at edge=5
is ~50.5% with implied breakeven ~16 pts. This suggests a mismatch in
sample/label/sign/push handling.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.spread_selection.calibration import (
    load_and_normalize_game_data,
    walk_forward_validate,
    predict_cover_probability,
    get_spread_bucket,
)


def main():
    print("=" * 80)
    print("CALIBRATION vs STRATEGY EVALUATION DIAGNOSTIC")
    print("=" * 80)

    # Load raw data
    csv_path = "data/spread_selection/ats_export.csv"
    print(f"\n1. Loading raw data from {csv_path}...")
    raw_df = pd.read_csv(csv_path)
    print(f"   Raw rows: {len(raw_df)}")

    # Normalize
    print("\n2. Normalizing data...")
    normalized_df = load_and_normalize_game_data(raw_df, jp_convention="pos_home_favored")
    print(f"   Normalized rows: {len(normalized_df)}")

    # Run walk-forward
    print("\n3. Running walk-forward validation...")
    wf_result = walk_forward_validate(normalized_df, min_train_seasons=2, exclude_covid=True)
    wf_games = wf_result.game_results
    print(f"   Walk-forward rows: {len(wf_games)}")
    print(f"   Years in walk-forward: {sorted(wf_games['year'].unique())}")

    # =========================================================================
    # DIAGNOSTIC 1: Push and edge==0 exclusion check
    # =========================================================================
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 1: PUSH AND EDGE==0 EXCLUSION CHECK")
    print("=" * 80)

    print(f"\nIn walk-forward game_results:")
    print(f"  Total rows: {len(wf_games)}")
    print(f"  Push rows: {wf_games['push'].sum()}")
    print(f"  edge_abs == 0 rows: {(wf_games['edge_abs'] == 0).sum()}")
    print(f"  Rows with p_cover_no_push NOT NaN: {wf_games['p_cover_no_push'].notna().sum()}")
    print(f"  Rows with p_cover_no_push IS NaN: {wf_games['p_cover_no_push'].isna().sum()}")

    # Check if pushes got predictions
    push_with_pred = wf_games[wf_games['push'] & wf_games['p_cover_no_push'].notna()]
    print(f"  Pushes that got predictions (SHOULD BE 0): {len(push_with_pred)}")

    # Check if edge==0 got predictions
    zero_edge_with_pred = wf_games[(wf_games['edge_abs'] == 0) & wf_games['p_cover_no_push'].notna()]
    print(f"  edge_abs==0 that got predictions (SHOULD BE 0): {len(zero_edge_with_pred)}")

    # =========================================================================
    # DIAGNOSTIC 2: Compare empirical vs predicted by edge bin
    # =========================================================================
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 2: EMPIRICAL vs PREDICTED BY EDGE BIN")
    print("=" * 80)

    # Filter to rows with predictions (excludes pushes and edge==0)
    eval_df = wf_games[wf_games['p_cover_no_push'].notna()].copy()
    print(f"\nEvaluation set: {len(eval_df)} games (has p_cover_no_push)")

    # Add bucket
    eval_df['bucket'] = eval_df['edge_abs'].apply(get_spread_bucket)

    print("\n| Bucket   | N Games | Empirical | Predicted | Delta   |")
    print("|----------|---------|-----------|-----------|---------|")

    bucket_order = ["[0,3)", "[3,5)", "[5,7)", "[7,10)", "[10,+)"]
    for bucket in bucket_order:
        bucket_df = eval_df[eval_df['bucket'] == bucket]
        if len(bucket_df) == 0:
            continue

        empirical = bucket_df['jp_side_covered'].mean()
        predicted = bucket_df['p_cover_no_push'].mean()
        delta = empirical - predicted

        print(f"| {bucket:<8} | {len(bucket_df):>7} | {empirical*100:>8.1f}% | {predicted*100:>8.1f}% | {delta*100:>+6.1f}% |")

    # =========================================================================
    # DIAGNOSTIC 3: edge_abs >= 5 comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 3: EDGE_ABS >= 5 COMPARISON")
    print("=" * 80)

    edge5_df = eval_df[eval_df['edge_abs'] >= 5]
    print(f"\nGames with edge_abs >= 5: {len(edge5_df)}")
    print(f"  - Pushes in this set: {edge5_df['push'].sum()} (should be 0)")
    print(f"  - edge_abs == 0 in this set: {(edge5_df['edge_abs'] == 0).sum()} (should be 0)")

    empirical_5plus = edge5_df['jp_side_covered'].mean()
    predicted_5plus = edge5_df['p_cover_no_push'].mean()

    print(f"\n  Empirical JP-side cover rate: {empirical_5plus*100:.1f}% ({edge5_df['jp_side_covered'].sum()}/{len(edge5_df)})")
    print(f"  Mean predicted p_cover:       {predicted_5plus*100:.1f}%")
    print(f"  Delta (empirical - predicted): {(empirical_5plus - predicted_5plus)*100:+.1f}%")

    # Compare to what compare_strategies reported
    print(f"\n  NOTE: compare_strategies reported OLD_5pt as 55.4% (362-291 on 653 games)")
    print(f"        This diagnostic shows {empirical_5plus*100:.1f}% on {len(edge5_df)} games")

    if abs(len(edge5_df) - 653) > 10:
        print(f"\n  *** MISMATCH: Different game counts! compare_strategies had 653, this has {len(edge5_df)} ***")

    # =========================================================================
    # DIAGNOSTIC 4: Check jp_side_covered calculation
    # =========================================================================
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 4: JP_SIDE_COVERED CALCULATION CHECK")
    print("=" * 80)

    # Manually verify jp_side_covered for a sample
    sample = edge5_df.sample(min(20, len(edge5_df)), random_state=42)

    mismatches = []
    for _, row in sample.iterrows():
        # Recalculate jp_side_covered from first principles
        if row['jp_favored_side'] == 'HOME':
            expected_covered = row['home_covered']
        else:
            expected_covered = row['away_covered']

        if expected_covered != row['jp_side_covered']:
            mismatches.append(row)

    print(f"\nManual verification of jp_side_covered on {len(sample)} samples:")
    print(f"  Mismatches found: {len(mismatches)}")

    # =========================================================================
    # DIAGNOSTIC 5: 20 random row-level examples
    # =========================================================================
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 5: 20 RANDOM ROW-LEVEL EXAMPLES (edge_abs >= 5)")
    print("=" * 80)

    sample = edge5_df.sample(min(20, len(edge5_df)), random_state=123)

    for i, (_, row) in enumerate(sample.iterrows()):
        print(f"\n--- Example {i+1}: {row['away_team']} @ {row['home_team']} (Week {row['week']}, {row['year']}) ---")
        print(f"  jp_spread (Vegas conv):    {row['jp_spread']:+.1f}")
        print(f"  vegas_spread:              {row['vegas_spread']:+.1f}")
        print(f"  edge_pts:                  {row['edge_pts']:+.1f}")
        print(f"  edge_abs:                  {row['edge_abs']:.1f}")
        print(f"  jp_favored_side:           {row['jp_favored_side']}")
        print(f"  actual_margin:             {row['actual_margin']:+.0f}")
        print(f"  cover_margin:              {row['cover_margin']:+.1f}")
        print(f"  home_covered:              {row['home_covered']}")
        print(f"  away_covered:              {row['away_covered']}")
        print(f"  push:                      {row['push']}")
        print(f"  jp_side_covered:           {row['jp_side_covered']}")
        print(f"  p_cover_no_push:           {row['p_cover_no_push']:.3f}")

        # Verify the logic
        if row['jp_favored_side'] == 'HOME':
            expected_covered = row['home_covered']
            print(f"  [VERIFY] JP favors HOME, home_covered={row['home_covered']} -> jp_side_covered should be {expected_covered}")
        else:
            expected_covered = row['away_covered']
            print(f"  [VERIFY] JP favors AWAY, away_covered={row['away_covered']} -> jp_side_covered should be {expected_covered}")

        if expected_covered != row['jp_side_covered']:
            print(f"  *** MISMATCH: Expected {expected_covered}, got {row['jp_side_covered']} ***")

    # =========================================================================
    # DIAGNOSTIC 6: Check fold-specific behavior
    # =========================================================================
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 6: FOLD-SPECIFIC ANALYSIS")
    print("=" * 80)

    for fold in wf_result.fold_summaries:
        year = fold['eval_year']
        year_df = eval_df[eval_df['year'] == year]
        edge5_year = year_df[year_df['edge_abs'] >= 5]

        if len(edge5_year) > 0:
            emp = edge5_year['jp_side_covered'].mean()
            pred = edge5_year['p_cover_no_push'].mean()
            print(f"\n{year}: {len(edge5_year)} games with edge_abs >= 5")
            print(f"  Empirical: {emp*100:.1f}%, Predicted: {pred*100:.1f}%, Delta: {(emp-pred)*100:+.1f}%")
            print(f"  Fold slope: {fold['slope']:.6f}, intercept: {fold['intercept']:.4f}")
            print(f"  Implied P(cover) at edge=5: {1 / (1 + np.exp(-(fold['intercept'] + fold['slope']*5))):.3f}")

    # =========================================================================
    # DIAGNOSTIC 7: Check if compare_strategies uses different data
    # =========================================================================
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 7: RECREATE compare_strategies LOGIC")
    print("=" * 80)

    # This is what compare_strategies does
    from src.spread_selection.calibration import calculate_ev_vectorized

    df_for_strategy = wf_games[
        wf_games['p_cover_no_push'].notna() & ~wf_games['push']
    ].copy()

    df_for_strategy['ev'] = calculate_ev_vectorized(df_for_strategy['p_cover_no_push'].values)

    old_mask = df_for_strategy['edge_abs'] >= 5.0
    old_subset = df_for_strategy[old_mask]

    print(f"\nRecreating OLD_5pt strategy:")
    print(f"  Games with p_cover_no_push & not push: {len(df_for_strategy)}")
    print(f"  Games with edge_abs >= 5: {len(old_subset)}")
    print(f"  jp_side_covered wins: {old_subset['jp_side_covered'].sum()}")
    print(f"  jp_side_covered losses: {len(old_subset) - old_subset['jp_side_covered'].sum()}")
    print(f"  Cover rate: {old_subset['jp_side_covered'].mean()*100:.1f}%")

    # =========================================================================
    # DIAGNOSTIC 8: Check raw backtest ATS calculation
    # =========================================================================
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 8: COMPARE TO RAW BACKTEST ATS COLUMNS")
    print("=" * 80)

    print("\nRaw CSV columns:", list(raw_df.columns))

    if 'ats_win' in raw_df.columns and 'edge' in raw_df.columns:
        raw_edge5 = raw_df[raw_df['edge'] >= 5]
        raw_edge5_no_push = raw_edge5[~raw_edge5['ats_push']] if 'ats_push' in raw_df.columns else raw_edge5

        print(f"\nFrom raw CSV (backtest's own calculation):")
        print(f"  Games with edge >= 5: {len(raw_edge5)}")
        print(f"  Games with edge >= 5 (no push): {len(raw_edge5_no_push)}")
        print(f"  ats_win sum: {raw_edge5_no_push['ats_win'].sum()}")
        print(f"  ATS win rate: {raw_edge5_no_push['ats_win'].mean()*100:.1f}%")

        # Now check how raw 'edge' and 'ats_win' map to our jp_side_covered
        print(f"\nComparing raw 'ats_win' to our 'jp_side_covered':")

        # Merge raw with normalized to compare
        raw_for_merge = raw_df[['game_id', 'edge', 'pick', 'ats_win', 'ats_push']].copy()
        norm_for_merge = normalized_df[['game_id', 'edge_abs', 'jp_favored_side', 'jp_side_covered', 'push']].copy()

        merged = raw_for_merge.merge(norm_for_merge, on='game_id', how='inner')
        print(f"  Merged rows: {len(merged)}")

        # Check agreement
        merged['pick_agrees'] = (
            ((merged['pick'] == 'HOME') & (merged['jp_favored_side'] == 'HOME')) |
            ((merged['pick'] == 'AWAY') & (merged['jp_favored_side'] == 'AWAY'))
        )
        merged['outcome_agrees'] = merged['ats_win'] == merged['jp_side_covered']

        print(f"  Pick agrees (HOME/AWAY): {merged['pick_agrees'].sum()}/{len(merged)} ({merged['pick_agrees'].mean()*100:.1f}%)")
        print(f"  Outcome agrees (ats_win vs jp_side_covered): {merged['outcome_agrees'].sum()}/{len(merged)} ({merged['outcome_agrees'].mean()*100:.1f}%)")

        # Show disagreements
        disagree = merged[~merged['outcome_agrees']]
        if len(disagree) > 0:
            print(f"\n  *** {len(disagree)} OUTCOME DISAGREEMENTS ***")
            print(f"  First 5 disagreements:")
            for _, row in disagree.head(5).iterrows():
                print(f"    game_id={row['game_id']}: pick={row['pick']}, jp_favored={row['jp_favored_side']}, "
                      f"ats_win={row['ats_win']}, jp_side_covered={row['jp_side_covered']}, "
                      f"ats_push={row['ats_push']}, push={row['push']}")

    print("\n" + "=" * 80)
    print("END OF DIAGNOSTIC REPORT")
    print("=" * 80)


if __name__ == "__main__":
    main()
