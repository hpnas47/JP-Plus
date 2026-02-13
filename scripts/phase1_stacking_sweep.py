#!/usr/bin/env python3
"""Phase 1 Stacking Interaction Sweep.

Tests whether shrinkage=0.90 + HFA offset adjustments can synergize
to fix Phase 1 ATS performance.

Hypothesis: Shrinkage=0.90 created -0.79 MSE (road bias).
            Removing HFA penalty may restore balance.

Fixed: Shrinkage = 0.90
Variable: HFA Offset (0.0, 0.50, 1.0, 1.5)
"""

import sys
sys.path.insert(0, '/Users/jason/Documents/CFB Power Ratings Model')

import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

from scripts.phase1_shrinkage_sweep import (
    fetch_betting_data,
    calculate_ats,
    calculate_edge_ats,
)


def run_backtest_with_params(hfa_offset: float) -> pd.DataFrame:
    """Run backtest and return predictions DataFrame."""
    cmd = [
        "python3", "scripts/backtest.py",
        "--start-week", "1",
        "--end-week", "3",
        "--years", "2022", "2023", "2024", "2025",
        "--hfa-offset", str(hfa_offset),
        "--qb-continuous",
        "--qb-scale", "5.0",
        "--qb-phase1-only",
        "--output", "/tmp/phase1_stacking_predictions.csv",
        "--no-diagnostics",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd="/Users/jason/Documents/CFB Power Ratings Model"
    )

    if result.returncode != 0:
        print(f"Backtest failed: {result.stderr}")
        return None

    return pd.read_csv("/tmp/phase1_stacking_predictions.csv")


def apply_shrinkage(df: pd.DataFrame, shrinkage: float = 0.90) -> pd.Series:
    """Apply shrinkage to rating differential (post-hoc)."""
    hfa = df["hfa"].values
    original_spread = df["predicted_spread"].values
    rating_diff = original_spread - hfa
    new_rating_diff = rating_diff * shrinkage
    return pd.Series(new_rating_diff + hfa, index=df.index)


def evaluate_config(hfa_offset: float, betting_pd: pd.DataFrame) -> dict:
    """Evaluate a single HFA offset configuration with shrinkage=0.90."""

    print(f"  Running HFA offset = {hfa_offset}...")
    pred_df = run_backtest_with_params(hfa_offset)

    if pred_df is None:
        return None

    # Merge with betting data
    df = pred_df.merge(
        betting_pd[["game_id", "spread_open", "spread_close"]],
        on="game_id",
        how="left"
    )

    # Apply shrinkage
    df["shrunk_spread"] = apply_shrinkage(df, 0.90)
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
    edge_5_close = calculate_edge_ats(df_valid, spread_col, "spread_close", 5.0)
    edge_5_open = calculate_edge_ats(df_valid, spread_col, "spread_open", 5.0)

    return {
        "hfa_offset": hfa_offset,
        "games": len(df_valid),
        "mae": mae,
        "mse": mse,
        "error_std": error_std,
        "ats_close": close_ats["pct"],
        "edge_5_close": edge_5_close["pct"],
        "edge_5_close_record": edge_5_close["record"],
        "edge_5_close_n": edge_5_close.get("n", 0),
        "edge_5_open": edge_5_open["pct"],
        "edge_5_open_record": edge_5_open["record"],
        "edge_5_open_n": edge_5_open.get("n", 0),
    }


def main():
    print("=" * 90)
    print("PHASE 1 STACKING INTERACTION SWEEP")
    print("Fixed: Shrinkage = 0.90")
    print("Variable: HFA Offset")
    print("=" * 90)

    # Fetch betting data once
    print("\nFetching betting lines...")
    betting_df = fetch_betting_data([2022, 2023, 2024, 2025])
    betting_pd = betting_df.to_pandas()
    print(f"  {len(betting_pd)} betting lines loaded")

    # Test configurations
    configs = [
        (0.0, "Test A: Restoration (no HFA penalty)"),
        (0.50, "Baseline: Current default"),
        (1.0, "Test B: Hybrid (moderate penalty)"),
        (1.5, "Test C: Double Down (aggressive)"),
    ]

    results = []
    for hfa_offset, description in configs:
        print(f"\n{description}")
        result = evaluate_config(hfa_offset, betting_pd)
        if result:
            results.append(result)
            print(f"  MAE={result['mae']:.2f}, MSE={result['mse']:+.2f}, "
                  f"Std={result['error_std']:.2f}")
            print(f"  5+ Edge: Close={result['edge_5_close']:.1f}%, "
                  f"Open={result['edge_5_open']:.1f}%")

    # Results table
    print("\n" + "=" * 100)
    print("STACKING SWEEP RESULTS (Shrinkage=0.90 LOCKED, HFA Offset varied)")
    print("Target: MSE → 0, 5+ Edge > 52.4%")
    print("=" * 100)

    header = (f"{'HFA Offset':>10} | {'Games':>6} | {'MAE':>6} | {'MSE':>7} | "
              f"{'ErrStd':>7} | {'5+ Cl':>7} | {'5+ Cl Rec':>12} | "
              f"{'5+ Op':>7} | {'5+ Op Rec':>12}")
    print(header)
    print("-" * 100)

    # Baseline is hfa_offset=0.50 (index 1)
    baseline = results[1] if len(results) > 1 else results[0]

    for r in results:
        delta_5c = r["edge_5_close"] - baseline["edge_5_close"]
        delta_5o = r["edge_5_open"] - baseline["edge_5_open"]
        delta_mse = r["mse"] - baseline["mse"]

        mark_5c = "↑" if delta_5c > 0.5 else ("↓" if delta_5c < -0.5 else " ")
        mark_5o = "↑" if delta_5o > 0.5 else ("↓" if delta_5o < -0.5 else " ")
        mark_mse = "↑" if delta_mse > 0.3 else ("↓" if delta_mse < -0.3 else " ")

        print(f"{r['hfa_offset']:>10.2f} | {r['games']:>6} | {r['mae']:>6.2f} | "
              f"{r['mse']:>+6.2f}{mark_mse}| {r['error_std']:>7.2f} | "
              f"{r['edge_5_close']:>6.1f}%{mark_5c}| {r['edge_5_close_record']:>12} | "
              f"{r['edge_5_open']:>6.1f}%{mark_5o}| {r['edge_5_open_record']:>12}")

    print("-" * 100)

    # Delta table
    print("\nDELTA FROM BASELINE (HFA Offset=0.50):")
    print(f"{'HFA Offset':>10} | {'MSE Δ':>8} | {'5+ Cl Δ':>9} | {'5+ Op Δ':>9} | {'Hypothesis':<40}")
    print("-" * 85)

    hypotheses = {
        0.0: "Restoration: Fix road bias by removing penalty",
        0.50: "Baseline",
        1.0: "Hybrid: Moderate synergy",
        1.5: "Double Down: Test crash theory",
    }

    for r in results:
        delta_mse = r["mse"] - baseline["mse"]
        delta_5c = r["edge_5_close"] - baseline["edge_5_close"]
        delta_5o = r["edge_5_open"] - baseline["edge_5_open"]

        print(f"{r['hfa_offset']:>10.2f} | {delta_mse:>+8.2f} | {delta_5c:>+9.1f}% | "
              f"{delta_5o:>+9.1f}% | {hypotheses.get(r['hfa_offset'], ''):<40}")

    # Verdict
    print("\n" + "=" * 90)
    print("HYPOTHESIS TESTING")
    print("=" * 90)

    best_5c = max(results, key=lambda x: x["edge_5_close"])
    best_5o = max(results, key=lambda x: x["edge_5_open"])
    closest_mse = min(results, key=lambda x: abs(x["mse"]))

    print(f"\nBest 5+ Edge (Close): HFA={best_5c['hfa_offset']} → {best_5c['edge_5_close']:.1f}% ({best_5c['edge_5_close_record']})")
    print(f"Best 5+ Edge (Open):  HFA={best_5o['hfa_offset']} → {best_5o['edge_5_open']:.1f}% ({best_5o['edge_5_open_record']})")
    print(f"Closest to MSE=0:     HFA={closest_mse['hfa_offset']} → MSE={closest_mse['mse']:+.2f}")

    # Test A hypothesis
    test_a = next((r for r in results if r["hfa_offset"] == 0.0), None)
    if test_a:
        print(f"\nTest A (Restoration, HFA=0.0):")
        print(f"  MSE: {test_a['mse']:+.2f} (target: ~0)")
        if abs(test_a["mse"]) < abs(baseline["mse"]):
            print(f"  ✓ MSE improved toward zero ({baseline['mse']:+.2f} → {test_a['mse']:+.2f})")
        else:
            print(f"  ✗ MSE did not improve")

        if test_a["edge_5_close"] > baseline["edge_5_close"]:
            print(f"  ✓ 5+ Edge improved ({baseline['edge_5_close']:.1f}% → {test_a['edge_5_close']:.1f}%)")
        else:
            print(f"  ✗ 5+ Edge degraded ({baseline['edge_5_close']:.1f}% → {test_a['edge_5_close']:.1f}%)")

    # Crash theory test
    test_c = next((r for r in results if r["hfa_offset"] == 1.5), None)
    if test_c:
        print(f"\nTest C (Double Down, HFA=1.5):")
        if test_c["edge_5_close"] < 48.0:
            print(f"  ✓ Crash theory CONFIRMED: 5+ Edge crashed to {test_c['edge_5_close']:.1f}%")
        else:
            print(f"  ✗ Crash theory REJECTED: 5+ Edge at {test_c['edge_5_close']:.1f}% (not crashed)")


if __name__ == "__main__":
    main()
