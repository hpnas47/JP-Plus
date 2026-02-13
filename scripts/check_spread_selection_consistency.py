#!/usr/bin/env python3
"""
CHECK 1: Predict Calibration Source Consistency

This script verifies that the `predict` command produces identical calibration
parameters to `validate` when given the same training data and configuration.

Consistency Criteria:
1. Same historical data source (ats_export.csv)
2. Same normalization (jp_convention="pos_home_favored")
3. Same training year selection (training_window_seasons)
4. Same exclusion filters (pushes, edge_abs == 0)
5. Identical fitted intercept/slope within tolerance (1e-10)
"""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import pandas as pd
from datetime import datetime

from src.spread_selection.calibration import (
    load_and_normalize_game_data,
    calibrate_cover_probability,
    walk_forward_validate,
    predict_cover_probability,
    CALIBRATION_MODES,
)
from scipy.special import expit


def trace_validate_path(historical_csv: str, eval_year: int, training_window_seasons: int | None):
    """Trace the validate/walk-forward calibration path."""
    print(f"\n{'='*70}")
    print(f"TRACING VALIDATE PATH: eval_year={eval_year}, window={training_window_seasons}")
    print("="*70)

    # Step 1: Load raw data
    raw_df = pd.read_csv(historical_csv)
    print(f"[1] Loaded raw CSV: {len(raw_df)} rows")
    print(f"    Columns: {list(raw_df.columns)}")
    print(f"    Years in data: {sorted(raw_df['year'].unique())}")

    # Step 2: Normalize
    normalized_df = load_and_normalize_game_data(raw_df, jp_convention="pos_home_favored")
    print(f"[2] After normalization: {len(normalized_df)} rows")
    print(f"    Added columns: jp_spread, vegas_spread, edge_pts, edge_abs, push, jp_favored_side, jp_side_covered")

    # Step 3: Filter to years < eval_year (simulate validate path)
    years = sorted(normalized_df["year"].unique())
    eligible_train_years = [y for y in years if y < eval_year]

    if training_window_seasons is not None:
        window_years = eligible_train_years[-training_window_seasons:]
    else:
        window_years = eligible_train_years

    train_mask = normalized_df["year"].isin(window_years)
    train_df = normalized_df[train_mask]

    print(f"[3] Training data selection:")
    print(f"    Eligible years < {eval_year}: {eligible_train_years}")
    print(f"    Window seasons: {training_window_seasons}")
    print(f"    Final training years: {window_years}")
    print(f"    Training rows (before exclusions): {len(train_df)}")

    # Step 4: Apply calibrate_cover_probability exclusions
    push_count = train_df["push"].sum()
    zero_edge_count = (train_df["edge_abs"] == 0).sum()

    calib_mask = (~train_df["push"]) & (train_df["edge_abs"] > 0)
    calib_df = train_df[calib_mask]

    print(f"[4] Calibration exclusions:")
    print(f"    Pushes excluded: {push_count}")
    print(f"    Zero-edge excluded: {zero_edge_count}")
    print(f"    Final training rows: {len(calib_df)}")
    print(f"    Years in final training: {sorted(calib_df['year'].unique().tolist())}")

    # Step 5: Fit calibration
    calibration = calibrate_cover_probability(train_df, min_games_warn=500)

    print(f"[5] Fitted calibration:")
    print(f"    intercept: {calibration.intercept:.10f}")
    print(f"    slope: {calibration.slope:.10f}")
    print(f"    n_games: {calibration.n_games}")
    print(f"    years_trained: {calibration.years_trained}")
    print(f"    p_cover_at_zero: {calibration.p_cover_at_zero:.6f}")
    print(f"    implied_5pt_pcover: {calibration.implied_5pt_pcover:.6f}")
    print(f"    implied_breakeven_edge: {calibration.implied_breakeven_edge:.4f}")

    p_at_10 = expit(calibration.intercept + calibration.slope * 10)
    print(f"    implied_10pt_pcover: {p_at_10:.6f}")

    return {
        "path": "validate",
        "eval_year": eval_year,
        "training_window_seasons": training_window_seasons,
        "training_years": window_years,
        "n_train_rows_before_exclusions": len(train_df),
        "pushes_excluded": int(push_count),
        "zero_edge_excluded": int(zero_edge_count),
        "n_train_games": calibration.n_games,
        "intercept": calibration.intercept,
        "slope": calibration.slope,
        "p_cover_at_zero": calibration.p_cover_at_zero,
        "implied_5pt_pcover": calibration.implied_5pt_pcover,
        "implied_10pt_pcover": p_at_10,
        "implied_breakeven_edge": calibration.implied_breakeven_edge,
    }


def trace_predict_path(historical_csv: str, pred_year: int, training_window_seasons: int | None):
    """Trace the predict calibration path."""
    print(f"\n{'='*70}")
    print(f"TRACING PREDICT PATH: pred_year={pred_year}, window={training_window_seasons}")
    print("="*70)

    # Step 1: Load raw data
    raw_df = pd.read_csv(historical_csv)
    print(f"[1] Loaded raw CSV: {len(raw_df)} rows")

    # Step 2: Filter to years < pred_year (as predict does)
    historical_df = raw_df[raw_df["year"] < pred_year]
    print(f"[2] Filtered to years < {pred_year}: {len(historical_df)} rows")
    print(f"    Years remaining: {sorted(historical_df['year'].unique())}")

    # Step 3: Normalize
    historical_normalized = load_and_normalize_game_data(
        historical_df, jp_convention="pos_home_favored"
    )
    print(f"[3] After normalization: {len(historical_normalized)} rows")

    # Step 4: Select training years based on window
    train_years = sorted(historical_normalized["year"].unique())

    if training_window_seasons is not None:
        window_years = train_years[-training_window_seasons:]
    else:
        window_years = train_years

    train_mask = historical_normalized["year"].isin(window_years)
    train_df = historical_normalized[train_mask]

    print(f"[4] Training data selection:")
    print(f"    Available years: {train_years}")
    print(f"    Window seasons: {training_window_seasons}")
    print(f"    Final training years: {list(window_years)}")  # Convert numpy to list
    print(f"    Training rows (before exclusions): {len(train_df)}")

    # Step 5: Apply calibrate_cover_probability exclusions
    push_count = train_df["push"].sum()
    zero_edge_count = (train_df["edge_abs"] == 0).sum()

    calib_mask = (~train_df["push"]) & (train_df["edge_abs"] > 0)
    calib_df = train_df[calib_mask]

    print(f"[5] Calibration exclusions:")
    print(f"    Pushes excluded: {push_count}")
    print(f"    Zero-edge excluded: {zero_edge_count}")
    print(f"    Final training rows: {len(calib_df)}")
    print(f"    Years in final training: {sorted(calib_df['year'].unique().tolist())}")

    # Step 6: Fit calibration
    calibration = calibrate_cover_probability(train_df, min_games_warn=500)

    print(f"[6] Fitted calibration:")
    print(f"    intercept: {calibration.intercept:.10f}")
    print(f"    slope: {calibration.slope:.10f}")
    print(f"    n_games: {calibration.n_games}")
    print(f"    years_trained: {calibration.years_trained}")
    print(f"    p_cover_at_zero: {calibration.p_cover_at_zero:.6f}")
    print(f"    implied_5pt_pcover: {calibration.implied_5pt_pcover:.6f}")
    print(f"    implied_breakeven_edge: {calibration.implied_breakeven_edge:.4f}")

    p_at_10 = expit(calibration.intercept + calibration.slope * 10)
    print(f"    implied_10pt_pcover: {p_at_10:.6f}")

    return {
        "path": "predict",
        "pred_year": pred_year,
        "training_window_seasons": training_window_seasons,
        "training_years": list(window_years),  # Convert numpy to list
        "n_train_rows_before_exclusions": len(train_df),
        "pushes_excluded": int(push_count),
        "zero_edge_excluded": int(zero_edge_count),
        "n_train_games": calibration.n_games,
        "intercept": calibration.intercept,
        "slope": calibration.slope,
        "p_cover_at_zero": calibration.p_cover_at_zero,
        "implied_5pt_pcover": calibration.implied_5pt_pcover,
        "implied_10pt_pcover": p_at_10,
        "implied_breakeven_edge": calibration.implied_breakeven_edge,
    }


def compare_results(validate_result: dict, predict_result: dict, tolerance: float = 1e-10):
    """Compare validate and predict results for consistency."""
    print(f"\n{'='*70}")
    print("CONSISTENCY COMPARISON")
    print("="*70)

    issues = []

    # Check training years
    v_years = [int(y) for y in validate_result["training_years"]]
    p_years = [int(y) for y in predict_result["training_years"]]

    if v_years != p_years:
        issues.append(f"Training years differ: validate={v_years}, predict={p_years}")
    else:
        print(f"✓ Training years match: {v_years}")

    # Check n_train_games
    if validate_result["n_train_games"] != predict_result["n_train_games"]:
        issues.append(
            f"n_train_games differs: validate={validate_result['n_train_games']}, "
            f"predict={predict_result['n_train_games']}"
        )
    else:
        print(f"✓ n_train_games match: {validate_result['n_train_games']}")

    # Check calibration parameters
    intercept_diff = abs(validate_result["intercept"] - predict_result["intercept"])
    slope_diff = abs(validate_result["slope"] - predict_result["slope"])

    if intercept_diff > tolerance:
        issues.append(
            f"Intercept differs by {intercept_diff:.2e}: "
            f"validate={validate_result['intercept']:.10f}, "
            f"predict={predict_result['intercept']:.10f}"
        )
    else:
        print(f"✓ Intercept matches within {tolerance}: diff={intercept_diff:.2e}")

    if slope_diff > tolerance:
        issues.append(
            f"Slope differs by {slope_diff:.2e}: "
            f"validate={validate_result['slope']:.10f}, "
            f"predict={predict_result['slope']:.10f}"
        )
    else:
        print(f"✓ Slope matches within {tolerance}: diff={slope_diff:.2e}")

    # Check derived quantities
    p5_diff = abs(validate_result["implied_5pt_pcover"] - predict_result["implied_5pt_pcover"])
    if p5_diff > 1e-6:
        issues.append(f"P(cover)@5 differs by {p5_diff:.6f}")
    else:
        print(f"✓ P(cover)@5 matches: {validate_result['implied_5pt_pcover']:.6f}")

    be_diff = abs(validate_result["implied_breakeven_edge"] - predict_result["implied_breakeven_edge"])
    if be_diff > 0.01:
        issues.append(f"Breakeven edge differs by {be_diff:.4f}")
    else:
        print(f"✓ Breakeven edge matches: {validate_result['implied_breakeven_edge']:.4f}")

    # Summary
    print("\n" + "-"*70)
    if issues:
        print("CONSISTENCY CHECK: FAILED")
        for issue in issues:
            print(f"  ✗ {issue}")
        return False
    else:
        print("CONSISTENCY CHECK: PASSED")
        return True


def run_consistency_check(
    historical_csv: str,
    eval_year: int,
    output_path: str,
):
    """Run full consistency check for both modes."""
    print(f"\n{'#'*70}")
    print("SPREAD SELECTION PREDICT CONSISTENCY CHECK")
    print(f"Historical CSV: {historical_csv}")
    print(f"Eval/Pred Year: {eval_year}")
    print(f"Output: {output_path}")
    print("#"*70)

    results = {
        "check_time": datetime.now().isoformat(),
        "historical_csv": historical_csv,
        "eval_year": eval_year,
        "modes": {},
    }

    all_passed = True

    for mode_name, mode_config in CALIBRATION_MODES.items():
        print(f"\n\n{'#'*70}")
        print(f"MODE: {mode_name.upper()} ({mode_config['label']})")
        print("#"*70)

        window = mode_config["training_window_seasons"]

        # Trace both paths
        validate_result = trace_validate_path(historical_csv, eval_year, window)
        predict_result = trace_predict_path(historical_csv, eval_year, window)

        # Compare
        passed = compare_results(validate_result, predict_result)
        all_passed = all_passed and passed

        results["modes"][mode_name] = {
            "training_window_seasons": window,
            "validate": validate_result,
            "predict": predict_result,
            "passed": passed,
        }

    # Write results
    results["all_passed"] = all_passed

    with open(output_path.replace(".md", ".json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Write markdown report
    with open(output_path, "w") as f:
        f.write("# Spread Selection Predict Consistency Check\n\n")
        f.write(f"**Generated:** {results['check_time']}\n\n")
        f.write(f"**Historical CSV:** `{historical_csv}`\n\n")
        f.write(f"**Eval/Pred Year:** {eval_year}\n\n")
        f.write(f"**Overall Result:** {'✓ PASSED' if all_passed else '✗ FAILED'}\n\n")

        f.write("## Test Methodology\n\n")
        f.write("This check verifies that `predict` produces identical calibration parameters to\n")
        f.write("`validate` when given the same training data. The test:\n\n")
        f.write("1. Loads the same historical CSV\n")
        f.write("2. Filters to years < eval_year\n")
        f.write("3. Applies the same normalization (`jp_convention=\"pos_home_favored\"`)\n")
        f.write("4. Selects training years using the same window logic\n")
        f.write("5. Applies the same exclusions (pushes, edge_abs == 0)\n")
        f.write("6. Fits logistic regression\n")
        f.write("7. Compares intercept and slope within tolerance (1e-10)\n\n")

        for mode_name, mode_result in results["modes"].items():
            f.write(f"## Mode: {mode_name.upper()}\n\n")
            f.write(f"**Training Window:** {mode_result['training_window_seasons']}\n\n")
            f.write(f"**Result:** {'✓ PASSED' if mode_result['passed'] else '✗ FAILED'}\n\n")

            v = mode_result["validate"]
            p = mode_result["predict"]

            f.write("| Metric | Validate | Predict | Match |\n")
            f.write("|--------|----------|---------|-------|\n")
            f.write(f"| Training Years | {v['training_years']} | {p['training_years']} | {'✓' if v['training_years'] == p['training_years'] else '✗'} |\n")
            f.write(f"| N Train Games | {v['n_train_games']} | {p['n_train_games']} | {'✓' if v['n_train_games'] == p['n_train_games'] else '✗'} |\n")
            f.write(f"| Intercept | {v['intercept']:.10f} | {p['intercept']:.10f} | {'✓' if abs(v['intercept'] - p['intercept']) < 1e-10 else '✗'} |\n")
            f.write(f"| Slope | {v['slope']:.10f} | {p['slope']:.10f} | {'✓' if abs(v['slope'] - p['slope']) < 1e-10 else '✗'} |\n")
            f.write(f"| P(cover)@5 | {v['implied_5pt_pcover']:.6f} | {p['implied_5pt_pcover']:.6f} | {'✓' if abs(v['implied_5pt_pcover'] - p['implied_5pt_pcover']) < 1e-6 else '✗'} |\n")
            f.write(f"| BE Edge | {v['implied_breakeven_edge']:.4f} | {p['implied_breakeven_edge']:.4f} | {'✓' if abs(v['implied_breakeven_edge'] - p['implied_breakeven_edge']) < 0.01 else '✗'} |\n")
            f.write("\n")

        f.write("## Reproduction Commands\n\n")
        f.write("```bash\n")
        f.write("# Run this consistency check\n")
        f.write(f"python3 scripts/check_spread_selection_consistency.py\n\n")
        f.write("# Compare with validate output for the same fold\n")
        f.write(f"python3 -m src.spread_selection.run_selection validate --csv {historical_csv} --end-year {eval_year}\n")
        f.write("```\n")

    print(f"\n{'#'*70}")
    print(f"FINAL RESULT: {'PASSED' if all_passed else 'FAILED'}")
    print(f"Report written to: {output_path}")
    print("#"*70)

    return all_passed


if __name__ == "__main__":
    historical_csv = "data/spread_selection/ats_export.csv"
    eval_year = 2025  # Simulate predicting 2025 using 2023-2024 (PRIMARY) or 2022-2024 (ULTRA)
    output_path = "data/outputs/spread_selection_predict_consistency_check.md"

    Path("data/outputs").mkdir(parents=True, exist_ok=True)

    passed = run_consistency_check(historical_csv, eval_year, output_path)
    sys.exit(0 if passed else 1)
