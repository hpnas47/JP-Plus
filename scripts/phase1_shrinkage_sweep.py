#!/usr/bin/env python3
"""Phase 1 Prior Spread Compression Sweep.

Tests whether shrinking the rating differential (keeping HFA constant)
improves Phase 1 ATS performance by reducing overconfidence.

Formula: NewSpread = (OldSpread - HFA) × Shrinkage + HFA
"""

import sys
sys.path.insert(0, '/Users/jason/Documents/CFB Power Ratings Model')

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

from config.settings import get_settings
from src.api.cfbd_client import CFBDClient


def fetch_betting_data(years: list[int]) -> pl.DataFrame:
    """Fetch betting lines for the specified years."""
    settings = get_settings()
    client = CFBDClient(api_key=settings.cfbd_api_key)

    preferred_providers = ["DraftKings", "ESPN Bet", "Bovada", "consensus"]
    records = []

    for year in years:
        for season_type in ["regular", "postseason"]:
            try:
                lines = client.get_betting_lines(year, season_type=season_type)
                for game_lines in lines:
                    if not game_lines.lines:
                        continue

                    # Find best provider line
                    selected_line = None
                    for provider in preferred_providers:
                        for line in game_lines.lines:
                            if line.provider == provider:
                                selected_line = line
                                break
                        if selected_line:
                            break

                    if not selected_line:
                        selected_line = game_lines.lines[0]

                    records.append({
                        "game_id": game_lines.id,
                        "home_team": game_lines.home_team,
                        "away_team": game_lines.away_team,
                        "spread_close": selected_line.spread,
                        "spread_open": selected_line.spread_open if selected_line.spread_open else selected_line.spread,
                    })
            except Exception as e:
                print(f"Warning: Failed to fetch {season_type} lines for {year}: {e}")

    return pl.DataFrame(records) if records else pl.DataFrame()


def load_predictions_and_lines():
    """Load predictions CSV and merge with betting lines."""

    # Load predictions
    pred_path = Path("/tmp/phase1_predictions.csv")
    if not pred_path.exists():
        print(f"Error: {pred_path} not found. Run backtest first with --output flag.")
        sys.exit(1)

    pred_df = pd.read_csv(pred_path)
    print(f"Loaded {len(pred_df)} predictions")

    # Fetch betting lines
    print("Fetching betting lines...")
    betting_df = fetch_betting_data([2022, 2023, 2024, 2025])
    betting_pd = betting_df.to_pandas()

    # Merge on game_id
    merged = pred_df.merge(
        betting_pd[["game_id", "spread_open", "spread_close"]],
        on="game_id",
        how="left"
    )

    # Filter to games with valid lines
    has_close = merged["spread_close"].notna()
    has_open = merged["spread_open"].notna()
    print(f"Games with close lines: {has_close.sum()}")
    print(f"Games with open lines: {has_open.sum()}")

    return merged


def apply_shrinkage(df: pd.DataFrame, shrinkage: float) -> pd.Series:
    """Apply shrinkage to rating differential.

    Formula: NewSpread = (OldSpread - HFA) × Shrinkage + HFA

    This compresses the non-HFA portion of the spread.
    """
    # Get HFA (already with offset applied)
    hfa = df["hfa"].values

    # Get original spread
    original_spread = df["predicted_spread"].values

    # Extract rating differential (everything except HFA)
    rating_diff = original_spread - hfa

    # Apply shrinkage to rating differential
    new_rating_diff = rating_diff * shrinkage

    # Reconstruct spread
    return pd.Series(new_rating_diff + hfa, index=df.index)


def calculate_ats(df: pd.DataFrame, spread_col: str, vegas_col: str) -> dict:
    """Calculate ATS metrics."""

    # Filter to games with valid lines
    valid = df[vegas_col].notna()
    df_valid = df[valid].copy()

    if len(df_valid) == 0:
        return {"wins": 0, "losses": 0, "pushes": 0, "pct": 0.0, "record": "0-0"}

    # Get values
    model_spread = df_valid[spread_col].values
    vegas_spread = -df_valid[vegas_col].values  # Convert to internal convention
    actual_margin = df_valid["actual_margin"].values

    # Model edge: positive = model favors home more than Vegas
    model_edge = model_spread - vegas_spread

    # Determine if home covered
    # Home covers when actual_margin > vegas_spread (internal convention)
    home_covers = actual_margin > vegas_spread

    # Determine if model was correct
    # Model picks home (edge > 0): correct if home covers
    # Model picks away (edge < 0): correct if home doesn't cover
    picks_home = model_edge > 0
    correct = np.where(picks_home, home_covers, ~home_covers)

    # Handle pushes
    is_push = np.abs(actual_margin - vegas_spread) < 0.01

    wins = int(np.sum(correct & ~is_push))
    losses = int(np.sum(~correct & ~is_push))
    pushes = int(np.sum(is_push))

    pct = 100.0 * wins / (wins + losses) if (wins + losses) > 0 else 0.0

    return {
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "pct": pct,
        "record": f"{wins}-{losses}",
    }


def calculate_edge_ats(df: pd.DataFrame, spread_col: str, vegas_col: str,
                       edge_threshold: float) -> dict:
    """Calculate ATS for games with edge >= threshold."""

    valid = df[vegas_col].notna()
    df_valid = df[valid].copy()

    if len(df_valid) == 0:
        return {"wins": 0, "losses": 0, "pct": 0.0, "record": "0-0", "n": 0}

    model_spread = df_valid[spread_col].values
    vegas_spread = -df_valid[vegas_col].values

    model_edge = np.abs(model_spread - vegas_spread)

    # Filter to games with sufficient edge
    edge_mask = model_edge >= edge_threshold
    n_edge = int(np.sum(edge_mask))

    if n_edge == 0:
        return {"wins": 0, "losses": 0, "pct": 0.0, "record": "0-0", "n": 0}

    df_edge = df_valid[edge_mask].copy()
    result = calculate_ats(df_edge, spread_col, vegas_col)
    result["n"] = n_edge

    return result


def calculate_bucket_mse(df: pd.DataFrame, spread_col: str,
                         min_spread: float, max_spread: float = None) -> tuple:
    """Calculate MSE for games in a specific spread bucket."""

    spread = np.abs(df[spread_col].values)
    actual = df["actual_margin"].values
    predicted = df[spread_col].values

    if max_spread is None:
        mask = spread >= min_spread
    else:
        mask = (spread >= min_spread) & (spread < max_spread)

    n = int(np.sum(mask))
    if n == 0:
        return 0.0, 0

    errors = predicted[mask] - actual[mask]
    mse = float(np.mean(errors))

    return mse, n


def evaluate_shrinkage(df: pd.DataFrame, shrinkage: float) -> dict:
    """Evaluate a single shrinkage value."""

    # Apply shrinkage
    df = df.copy()
    df["shrunk_spread"] = apply_shrinkage(df, shrinkage)

    spread_col = "shrunk_spread"

    # Calculate prediction errors (filter out any rows with missing predictions or actual_margin)
    valid_mask = df[spread_col].notna() & df["actual_margin"].notna()
    errors = df.loc[valid_mask, spread_col].values - df.loc[valid_mask, "actual_margin"].values
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors))  # Mean signed error
    error_std = float(np.std(errors))
    rmse = float(np.sqrt(np.mean(errors**2)))

    # ATS vs Close
    close_ats = calculate_ats(df, spread_col, "spread_close")
    edge_3_close = calculate_edge_ats(df, spread_col, "spread_close", 3.0)
    edge_5_close = calculate_edge_ats(df, spread_col, "spread_close", 5.0)

    # ATS vs Open
    open_ats = calculate_ats(df, spread_col, "spread_open")
    edge_5_open = calculate_edge_ats(df, spread_col, "spread_open", 5.0)

    # Bucket-specific: 14+ spread games
    bucket_14plus_mse, bucket_14plus_n = calculate_bucket_mse(df, spread_col, 14.0)

    # 21+ spread bucket (the problematic one from diagnostic)
    bucket_21plus_mse, bucket_21plus_n = calculate_bucket_mse(df, spread_col, 21.0)

    return {
        "shrinkage": shrinkage,
        "games": len(df),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "error_std": error_std,
        "ats_close": close_ats["pct"],
        "ats_close_record": close_ats["record"],
        "edge_3_close": edge_3_close["pct"],
        "edge_3_close_record": edge_3_close["record"],
        "edge_5_close": edge_5_close["pct"],
        "edge_5_close_record": edge_5_close["record"],
        "ats_open": open_ats["pct"],
        "ats_open_record": open_ats["record"],
        "edge_5_open": edge_5_open["pct"],
        "edge_5_open_record": edge_5_open["record"],
        "bucket_14plus_mse": bucket_14plus_mse,
        "bucket_14plus_n": bucket_14plus_n,
        "bucket_21plus_mse": bucket_21plus_mse,
        "bucket_21plus_n": bucket_21plus_n,
    }


def main():
    print("=" * 90)
    print("PHASE 1 PRIOR SPREAD COMPRESSION SWEEP")
    print("Formula: NewSpread = (OldSpread - HFA) × Shrinkage + HFA")
    print("=" * 90)

    # Load data
    df = load_predictions_and_lines()

    # Shrinkage values to test - include fine-grained around 0.90
    shrinkages = [1.00, 0.95, 0.92, 0.90, 0.88, 0.85, 0.80]

    results = []
    for shrink in shrinkages:
        print(f"\nEvaluating shrinkage = {shrink:.2f}...")
        result = evaluate_shrinkage(df, shrink)
        results.append(result)
        print(f"  MAE={result['mae']:.2f}, MSE={result['mse']:+.2f}, ErrStd={result['error_std']:.2f}")
        print(f"  ATS Close={result['ats_close']:.1f}%, 5+ Edge Close={result['edge_5_close']:.1f}%")

    # Print results table
    print("\n" + "=" * 120)
    print("PHASE 1 SHRINKAGE SWEEP RESULTS (Weeks 1-3, 2022-2025)")
    print("Targets: Error Std 17.55→15.81, 5+ Edge >53%, 21+ bucket MSE +2.52→0")
    print("=" * 120)

    header = (f"{'Shrink':>8} | {'MAE':>6} | {'MSE':>7} | {'ErrStd':>7} | "
              f"{'ATS Cl':>7} | {'3+ Cl':>7} | {'5+ Cl':>7} | {'5+ Cl Rec':>10} | "
              f"{'5+ Op':>7} | {'5+ Op Rec':>10} | {'21+ MSE':>8}")
    print(header)
    print("-" * 120)

    baseline = results[0]

    for r in results:
        delta_5_close = r["edge_5_close"] - baseline["edge_5_close"]
        delta_5_open = r["edge_5_open"] - baseline["edge_5_open"]
        delta_std = r["error_std"] - baseline["error_std"]
        delta_21_mse = r["bucket_21plus_mse"] - baseline["bucket_21plus_mse"]

        mark_5c = "↑" if delta_5_close > 0.5 else ("↓" if delta_5_close < -0.5 else " ")
        mark_5o = "↑" if delta_5_open > 0.5 else ("↓" if delta_5_open < -0.5 else " ")
        mark_std = "↓" if delta_std < -0.3 else ("↑" if delta_std > 0.3 else " ")
        mark_21 = "↓" if delta_21_mse < -0.3 else ("↑" if delta_21_mse > 0.3 else " ")

        print(f"{r['shrinkage']:>8.2f} | {r['mae']:>6.2f} | {r['mse']:>+7.2f} | "
              f"{r['error_std']:>6.2f}{mark_std}| {r['ats_close']:>6.1f}% | "
              f"{r['edge_3_close']:>6.1f}% | {r['edge_5_close']:>6.1f}%{mark_5c}| {r['edge_5_close_record']:>10} | "
              f"{r['edge_5_open']:>6.1f}%{mark_5o}| {r['edge_5_open_record']:>10} | "
              f"{r['bucket_21plus_mse']:>+7.2f}{mark_21}")

    print("-" * 120)

    # Find best
    best_5_close = max(results, key=lambda x: x["edge_5_close"])
    best_5_open = max(results, key=lambda x: x["edge_5_open"])
    best_ats = max(results, key=lambda x: x["ats_close"])
    lowest_std = min(results, key=lambda x: x["error_std"])

    print("\nBEST SETTINGS:")
    print(f"  Best 5+ Edge (Close): shrinkage={best_5_close['shrinkage']} → {best_5_close['edge_5_close']:.1f}% ({best_5_close['edge_5_close_record']})")
    print(f"  Best 5+ Edge (Open):  shrinkage={best_5_open['shrinkage']} → {best_5_open['edge_5_open']:.1f}% ({best_5_open['edge_5_open_record']})")
    print(f"  Best ATS (Close):     shrinkage={best_ats['shrinkage']} → {best_ats['ats_close']:.1f}% ({best_ats['ats_close_record']})")
    print(f"  Lowest Error Std:     shrinkage={lowest_std['shrinkage']} → {lowest_std['error_std']:.2f}")

    print("\nDELTA FROM BASELINE (shrinkage=1.00):")
    print(f"{'Shrink':>8} | {'5+ Cl Δ':>9} | {'5+ Op Δ':>9} | {'ErrStd Δ':>10} | {'21+ MSE Δ':>10}")
    print("-" * 55)
    for r in results:
        print(f"{r['shrinkage']:>8.2f} | {r['edge_5_close'] - baseline['edge_5_close']:>+9.1f}% | "
              f"{r['edge_5_open'] - baseline['edge_5_open']:>+9.1f}% | "
              f"{r['error_std'] - baseline['error_std']:>+10.2f} | "
              f"{r['bucket_21plus_mse'] - baseline['bucket_21plus_mse']:>+10.2f}")

    # Diagnostic summary
    print("\n" + "=" * 90)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 90)
    print(f"Baseline (shrinkage=1.00):")
    print(f"  Error Std:    {baseline['error_std']:.2f} (target: 15.81, gap: {baseline['error_std'] - 15.81:+.2f})")
    print(f"  5+ Edge (Cl): {baseline['edge_5_close']:.1f}% (target: 53%+, gap: {baseline['edge_5_close'] - 53.0:+.1f}%)")
    print(f"  21+ MSE:      {baseline['bucket_21plus_mse']:+.2f} (target: ~0, N={baseline['bucket_21plus_n']})")

    # Check if any shrinkage meets targets
    target_met = False
    for r in results:
        if r["edge_5_close"] >= 53.0:
            print(f"\n✓ shrinkage={r['shrinkage']} achieves 5+ Edge >= 53% ({r['edge_5_close']:.1f}%)")
            target_met = True
        if r["error_std"] <= 16.0:
            print(f"✓ shrinkage={r['shrinkage']} reduces Error Std to {r['error_std']:.2f}")
            target_met = True

    if not target_met:
        print("\n✗ No shrinkage value meets the target criteria.")
        print("  Spread compression alone may not be sufficient to fix Phase 1.")


if __name__ == "__main__":
    main()
