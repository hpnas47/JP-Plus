#!/usr/bin/env python3
"""Phase 1 Purity Test: Force 100% Prior Weight for Weeks 1-3.

Tests whether removing early-season "small sample noise" improves Phase 1 ATS.

Hypothesis: Week 3's poor performance (+1.43 MSE) coincides with 17% in-season
blend introduction. Forcing pure priors may reduce noise.

Configuration:
- Shrinkage: 0.90 (locked - optimal from shrinkage sweep)
- HFA Offset: -0.50 (standard baseline)
- Prior Weight: 100% for weeks 1, 2, 3 (variable under test)
"""

import sys
sys.path.insert(0, '/Users/jason/Documents/CFB Power Ratings Model')

import subprocess
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

# Import from shrinkage sweep for betting data and ATS calculation
from scripts.phase1_shrinkage_sweep import (
    fetch_betting_data,
    calculate_ats,
    calculate_edge_ats,
)


@dataclass
class PurityResult:
    mode: str
    games: int
    mae: float
    mse: float
    error_std: float
    ats_close: float
    ats_close_record: str
    edge_3_close: float
    edge_3_close_record: str
    edge_5_close: float
    edge_5_close_record: str
    edge_5_open: float
    edge_5_open_record: str


def run_backtest_with_prior_lock(lock_prior_weeks: int = 0) -> str:
    """Run backtest with optional prior lock for early weeks.

    Args:
        lock_prior_weeks: Number of weeks to force 100% prior weight.
                         0 = normal blending, 3 = pure prior for weeks 1-3.
    """
    cmd = [
        "python3", "scripts/backtest.py",
        "--start-week", "1",
        "--end-week", "3",
        "--years", "2022", "2023", "2024", "2025",
        "--hfa-offset", "0.50",  # Standard baseline
        "--qb-continuous",
        "--qb-scale", "5.0",
        "--qb-phase1-only",
        "--output", "/tmp/phase1_purity_predictions.csv",
    ]

    if lock_prior_weeks > 0:
        cmd.extend(["--lock-prior-weeks", str(lock_prior_weeks)])

    print(f"\n{'='*60}")
    print(f"Running backtest: lock_prior_weeks={lock_prior_weeks}")
    print(f"{'='*60}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd="/Users/jason/Documents/CFB Power Ratings Model"
    )

    return result.stdout + result.stderr


def apply_shrinkage(df: pd.DataFrame, shrinkage: float = 0.90) -> pd.Series:
    """Apply shrinkage=0.90 to rating differential.

    Formula: NewSpread = (OldSpread - HFA) × Shrinkage + HFA
    """
    hfa = df["hfa"].values
    original_spread = df["predicted_spread"].values
    rating_diff = original_spread - hfa
    new_rating_diff = rating_diff * shrinkage
    return pd.Series(new_rating_diff + hfa, index=df.index)


def load_and_evaluate(mode: str, shrinkage: float = 0.90) -> PurityResult:
    """Load predictions and evaluate with shrinkage applied."""

    pred_path = Path("/tmp/phase1_purity_predictions.csv")
    if not pred_path.exists():
        print(f"Error: {pred_path} not found")
        sys.exit(1)

    pred_df = pd.read_csv(pred_path)

    # Fetch betting data
    print("Fetching betting lines...")
    betting_df = fetch_betting_data([2022, 2023, 2024, 2025])
    betting_pd = betting_df.to_pandas()

    # Merge
    df = pred_df.merge(
        betting_pd[["game_id", "spread_open", "spread_close"]],
        on="game_id",
        how="left"
    )

    # Apply shrinkage
    df["shrunk_spread"] = apply_shrinkage(df, shrinkage)
    spread_col = "shrunk_spread"

    # Filter valid predictions
    valid_mask = df[spread_col].notna() & df["actual_margin"].notna()
    df_valid = df[valid_mask].copy()

    # Calculate errors
    errors = df_valid[spread_col].values - df_valid["actual_margin"].values
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors))
    error_std = float(np.std(errors))

    # ATS metrics
    close_ats = calculate_ats(df_valid, spread_col, "spread_close")
    edge_3_close = calculate_edge_ats(df_valid, spread_col, "spread_close", 3.0)
    edge_5_close = calculate_edge_ats(df_valid, spread_col, "spread_close", 5.0)
    edge_5_open = calculate_edge_ats(df_valid, spread_col, "spread_open", 5.0)

    return PurityResult(
        mode=mode,
        games=len(df_valid),
        mae=mae,
        mse=mse,
        error_std=error_std,
        ats_close=close_ats["pct"],
        ats_close_record=close_ats["record"],
        edge_3_close=edge_3_close["pct"],
        edge_3_close_record=edge_3_close["record"],
        edge_5_close=edge_5_close["pct"],
        edge_5_close_record=edge_5_close["record"],
        edge_5_open=edge_5_open["pct"],
        edge_5_open_record=edge_5_open["record"],
    )


def main():
    print("=" * 90)
    print("PHASE 1 PURITY TEST: 100% Prior Weight for Weeks 1-3")
    print("Shrinkage locked at 0.90, HFA offset locked at -0.50")
    print("=" * 90)

    results = []

    # Test 1: Normal blending (baseline with shrinkage)
    print("\n" + "="*60)
    print("TEST 1: Normal Blending + Shrinkage=0.90 (BASELINE)")
    print("="*60)
    run_backtest_with_prior_lock(lock_prior_weeks=0)
    result_normal = load_and_evaluate("Normal Blend + Shrink")
    results.append(result_normal)
    print(f"MAE={result_normal.mae:.2f}, MSE={result_normal.mse:+.2f}, "
          f"5+ Edge Close={result_normal.edge_5_close:.1f}%")

    # Test 2: Pure prior (100% prior weight for weeks 1-3)
    print("\n" + "="*60)
    print("TEST 2: Pure Prior (100%) + Shrinkage=0.90")
    print("="*60)
    run_backtest_with_prior_lock(lock_prior_weeks=3)
    result_pure = load_and_evaluate("Pure Prior + Shrink")
    results.append(result_pure)
    print(f"MAE={result_pure.mae:.2f}, MSE={result_pure.mse:+.2f}, "
          f"5+ Edge Close={result_pure.edge_5_close:.1f}%")

    # Results table
    print("\n" + "=" * 110)
    print("PURITY TEST RESULTS (Phase 1: Weeks 1-3, 2022-2025)")
    print("Target: 5+ Edge > 52.4% (profitable)")
    print("=" * 110)

    header = (f"{'Mode':<25} | {'Games':>6} | {'MAE':>6} | {'MSE':>7} | {'ErrStd':>7} | "
              f"{'ATS Cl':>7} | {'5+ Cl':>7} | {'5+ Cl Rec':>10} | {'5+ Op':>7} | {'5+ Op Rec':>10}")
    print(header)
    print("-" * 110)

    baseline = results[0]

    for r in results:
        delta_5c = r.edge_5_close - baseline.edge_5_close
        delta_5o = r.edge_5_open - baseline.edge_5_open
        mark_5c = "↑" if delta_5c > 0.3 else ("↓" if delta_5c < -0.3 else " ")
        mark_5o = "↑" if delta_5o > 0.3 else ("↓" if delta_5o < -0.3 else " ")

        print(f"{r.mode:<25} | {r.games:>6} | {r.mae:>6.2f} | {r.mse:>+7.2f} | "
              f"{r.error_std:>7.2f} | {r.ats_close:>6.1f}% | "
              f"{r.edge_5_close:>6.1f}%{mark_5c}| {r.edge_5_close_record:>10} | "
              f"{r.edge_5_open:>6.1f}%{mark_5o}| {r.edge_5_open_record:>10}")

    print("-" * 110)

    # Delta summary
    print("\nDELTA FROM NORMAL BLENDING:")
    delta_5c = result_pure.edge_5_close - result_normal.edge_5_close
    delta_5o = result_pure.edge_5_open - result_normal.edge_5_open
    delta_mse = result_pure.mse - result_normal.mse
    delta_std = result_pure.error_std - result_normal.error_std

    print(f"  5+ Edge (Close): {delta_5c:+.1f}%")
    print(f"  5+ Edge (Open):  {delta_5o:+.1f}%")
    print(f"  MSE:             {delta_mse:+.2f}")
    print(f"  Error Std:       {delta_std:+.2f}")

    # Verdict
    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)

    if result_pure.edge_5_close > 52.4:
        print(f"✓ APPROVED: Pure Prior achieves profitable 5+ Edge ({result_pure.edge_5_close:.1f}% > 52.4%)")
        print("  Recommendation: Implement --lock-prior-weeks 3 as Phase 1 production mode")
    elif result_pure.edge_5_close > result_normal.edge_5_close + 0.5:
        print(f"↑ IMPROVED: Pure Prior improved 5+ Edge by +{delta_5c:.1f}% but still below 52.4%")
        print("  Recommendation: Consider combining with other Phase 1 fixes")
    else:
        print(f"✗ NO IMPROVEMENT: Pure Prior did not improve ATS ({delta_5c:+.1f}%)")
        print("  Conclusion: Early-season blending is NOT the source of Phase 1 noise")
        print("  The noise is in the SP+ priors themselves")


if __name__ == "__main__":
    main()
