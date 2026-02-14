#!/usr/bin/env python3
"""Out-of-Sample Spread EV Calibration Validation with Statistical Support.

This script implements production-grade calibration validation:
1. Fit calibration on training years (2022-2024)
2. Evaluate on held-out test year (2025)
3. Phase-aware routing (Phase 1 skip/weighted, Phase 2 only)
4. Placeholder odds detection and separate reporting
5. Wilson CI for ATS%, Bootstrap CI for ROI
6. Matched comparisons (apples-to-apples)
7. EV decile reliability diagnostics
8. Recommendation for 2026 policy

Usage:
    python3 scripts/validate_spread_calibration_oos.py
    python3 scripts/validate_spread_calibration_oos.py --train-years 2022 2023 2024 --test-years 2025
    python3 scripts/validate_spread_calibration_oos.py --phase1-policy skip

Output:
    - Calibration artifacts with train_years metadata
    - OOS report with CIs, matched comparisons, recommendations
"""

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.spread_selection.calibration import (
    CalibrationResult,
    load_and_normalize_game_data,
    predict_cover_probability,
    breakeven_prob,
)
from src.spread_selection.stats_utils import (
    wilson_ci,
    compute_ats_with_ci,
    bootstrap_mean_ci,
    compute_roi_with_ci,
    matched_top_n_by_week,
    compare_at_thresholds,
    compute_ev_decile_reliability,
    generate_recommendation,
    ATSWithCI,
    ROIWithCI,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_TRAIN_YEARS = [2022, 2023, 2024]
DEFAULT_TEST_YEARS = [2025]
DEFAULT_ARTIFACT_DIR = "data/spread_selection/artifacts"
DEFAULT_ATS_EXPORT = "data/spread_selection/ats_export.csv"

PHASE1_WEEKS = (1, 3)
PHASE2_WEEKS = (4, 15)
PHASE3_WEEKS = (16, 99)


# =============================================================================
# PLACEHOLDER ODDS DETECTION
# =============================================================================

def detect_placeholder_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Add odds_placeholder column to detect assumed -110 odds."""
    df = df.copy()
    # All historical CFBD data uses assumed -110 odds
    df['odds_placeholder'] = True
    logger.warning(
        "All historical odds are placeholders (assumed -110). "
        "ROI calculations should be interpreted with caution."
    )
    return df


# =============================================================================
# CALIBRATION FITTING
# =============================================================================

def fit_calibration_on_years(
    df: pd.DataFrame,
    train_years: list[int],
    week_filter: Optional[tuple[int, int]] = None,
    sample_weights: Optional[np.ndarray] = None,
) -> CalibrationResult:
    """Fit logistic calibration on specific training years."""
    train_mask = df['year'].isin(train_years)
    train_df = df[train_mask].copy()

    if week_filter:
        min_week, max_week = week_filter
        week_mask = (train_df['week'] >= min_week) & (train_df['week'] <= max_week)
        train_df = train_df[week_mask]

    valid_mask = (~train_df['push']) & (train_df['edge_abs'] > 0)
    train_df = train_df[valid_mask]

    if len(train_df) < 100:
        raise ValueError(f"Insufficient training data: {len(train_df)} games")

    X = train_df[['edge_abs']].values
    y = train_df['jp_side_covered'].astype(int).values

    model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, fit_intercept=True)

    if sample_weights is not None:
        w = sample_weights[valid_mask.values] if len(sample_weights) == len(df[train_mask]) else sample_weights
        model.fit(X, y, sample_weight=w)
    else:
        model.fit(X, y)

    intercept = model.intercept_[0]
    slope = model.coef_[0][0]

    p_cover_at_zero = expit(intercept)
    implied_5pt_pcover = expit(intercept + slope * 5)

    breakeven = breakeven_prob(-110)
    logit_breakeven = np.log(breakeven / (1 - breakeven))

    if slope > 0:
        implied_breakeven_edge = (logit_breakeven - intercept) / slope
    else:
        implied_breakeven_edge = float('inf')

    years_trained = sorted(train_df['year'].unique().tolist())

    return CalibrationResult(
        intercept=intercept,
        slope=slope,
        n_games=len(train_df),
        years_trained=years_trained,
        implied_breakeven_edge=implied_breakeven_edge,
        implied_5pt_pcover=implied_5pt_pcover,
        p_cover_at_zero=p_cover_at_zero,
    )


# =============================================================================
# OUT-OF-SAMPLE EVALUATION
# =============================================================================

@dataclass
class OOSMetrics:
    """Out-of-sample evaluation metrics with CIs."""
    n_bets: int
    wins: int
    losses: int
    pushes: int
    ats: ATSWithCI
    roi: ROIWithCI
    avg_ev: float
    avg_edge: float
    breakeven_edge: float
    phase: str
    years: list[int]
    odds_placeholder_pct: float


def apply_calibration_to_df(
    df: pd.DataFrame,
    calibration: CalibrationResult,
) -> pd.DataFrame:
    """Apply calibration to compute EV for each row."""
    df = df.copy()

    has_edge = (df['edge_abs'] > 0) & ~df['push']
    df['p_cover_cal'] = np.nan
    df['ev_cal'] = np.nan

    if has_edge.sum() > 0:
        edge_vals = df.loc[has_edge, 'edge_abs'].values
        p_cover = predict_cover_probability(edge_vals, calibration)
        ev = p_cover * (100/110) - (1 - p_cover)

        df.loc[has_edge, 'p_cover_cal'] = p_cover
        df.loc[has_edge, 'ev_cal'] = ev

    return df


def evaluate_oos_with_ci(
    df: pd.DataFrame,
    calibration: CalibrationResult,
    test_years: list[int],
    week_filter: Optional[tuple[int, int]] = None,
    min_ev: float = 0.03,
) -> tuple[OOSMetrics, pd.DataFrame]:
    """Evaluate calibration out-of-sample with confidence intervals.

    Returns:
        (OOSMetrics, bets_df) - metrics and the filtered bets DataFrame
    """
    test_mask = df['year'].isin(test_years)
    test_df = df[test_mask].copy()

    if week_filter:
        min_week, max_week = week_filter
        week_mask = (test_df['week'] >= min_week) & (test_df['week'] <= max_week)
        test_df = test_df[week_mask]
        phase = f"weeks_{min_week}_{max_week}"
    else:
        phase = "all_weeks"

    # Apply calibration
    test_df = apply_calibration_to_df(test_df, calibration)

    # Select bets with EV >= threshold
    ev_mask = test_df['ev_cal'] >= min_ev
    bets = test_df[ev_mask].copy()

    n_bets = len(bets)

    if n_bets == 0:
        return (OOSMetrics(
            n_bets=0, wins=0, losses=0, pushes=0,
            ats=ATSWithCI(0, 0, 0, 0.0, 0.0, 100.0),
            roi=ROIWithCI(0.0, 0.0, 0.0, 0, 0.0),
            avg_ev=0.0, avg_edge=0.0,
            breakeven_edge=calibration.implied_breakeven_edge,
            phase=phase, years=test_years,
            odds_placeholder_pct=100.0,
        ), bets)

    # Exclude pushes for ATS
    non_push = bets[~bets['push']]
    wins = int(non_push['jp_side_covered'].sum())
    losses = len(non_push) - wins
    pushes = int(bets['push'].sum())

    # Compute with CIs
    ats = compute_ats_with_ci(wins, losses)
    roi = compute_roi_with_ci(wins, losses, pushes)

    avg_ev = bets['ev_cal'].mean()
    avg_edge = bets['edge_abs'].mean()
    odds_placeholder_pct = 100 * bets['odds_placeholder'].mean() if 'odds_placeholder' in bets.columns else 100.0

    return (OOSMetrics(
        n_bets=n_bets,
        wins=wins,
        losses=losses,
        pushes=pushes,
        ats=ats,
        roi=roi,
        avg_ev=avg_ev,
        avg_edge=avg_edge,
        breakeven_edge=calibration.implied_breakeven_edge,
        phase=phase,
        years=test_years,
        odds_placeholder_pct=odds_placeholder_pct,
    ), bets)


# =============================================================================
# ARTIFACT SAVING
# =============================================================================

def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def save_calibration_artifact(
    calibration: CalibrationResult,
    name: str,
    train_years: list[int],
    week_filter: Optional[tuple[int, int]],
    artifact_dir: str,
    oos_metrics: Optional[OOSMetrics] = None,
) -> str:
    """Save calibration artifact with full metadata."""
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_years_str = "_".join(str(y) for y in train_years)
    filename = f"spread_calibration_{name}_train_{train_years_str}.json"
    filepath = output_dir / filename

    artifact = {
        "artifact_id": f"{name}_{train_years_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "created_at": datetime.now().isoformat(),
        "model_type": "logistic_regression",
        "calibration_name": name,
        "parameters": {
            "intercept": float(calibration.intercept),
            "slope": float(calibration.slope),
        },
        "metadata": {
            "n_games": calibration.n_games,
            "train_years": train_years,
            "week_filter": list(week_filter) if week_filter else None,
            "breakeven_edge_at_110": float(calibration.implied_breakeven_edge),
            "p_cover_at_zero": float(calibration.p_cover_at_zero),
            "p_cover_at_5pt": float(calibration.implied_5pt_pcover),
            "git_commit": get_git_commit(),
        },
    }

    if oos_metrics:
        artifact["oos_validation"] = {
            "test_years": oos_metrics.years,
            "n_bets": oos_metrics.n_bets,
            "wins": oos_metrics.wins,
            "losses": oos_metrics.losses,
            "ats_pct": oos_metrics.ats.ats_pct,
            "ats_ci": [oos_metrics.ats.ci_low, oos_metrics.ats.ci_high],
            "roi_pct": oos_metrics.roi.roi_pct,
            "roi_ci": [oos_metrics.roi.ci_low, oos_metrics.roi.ci_high],
        }

    with open(filepath, 'w') as f:
        json.dump(artifact, f, indent=2)

    logger.info(f"Saved artifact: {filepath}")
    return str(filepath)


def save_enhanced_oos_report(
    in_sample_metrics: dict[str, OOSMetrics],
    out_sample_metrics: dict[str, OOSMetrics],
    matched_metrics: dict[str, OOSMetrics],
    threshold_comparison: pd.DataFrame,
    decile_reliability: dict,
    recommendation,
    artifact_dir: str,
    test_years: list[int],
) -> str:
    """Save enhanced OOS validation report with CIs and recommendations."""
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_years_str = "_".join(str(y) for y in test_years)
    filename = f"spread_calibration_oos_report_{test_years_str}.md"
    filepath = output_dir / filename

    lines = []
    lines.append("# Spread EV Calibration Out-of-Sample Validation Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().isoformat()}")
    lines.append(f"**Git Commit:** {get_git_commit() or 'N/A'}")
    lines.append("")

    # Warning
    lines.append("## ⚠️ Odds Warning")
    lines.append("")
    lines.append("All historical odds are **placeholders** (assumed -110).")
    lines.append("ROI calculations should be interpreted with caution.")
    lines.append("")

    # Summary with CIs
    lines.append("## Summary Comparison (with 95% Confidence Intervals)")
    lines.append("")
    lines.append("| Calibration | Sample | Bets | ATS% | ATS CI | ROI% | ROI CI |")
    lines.append("|-------------|--------|------|------|--------|------|--------|")

    for name in sorted(set(in_sample_metrics.keys()) | set(out_sample_metrics.keys())):
        if name in in_sample_metrics:
            m = in_sample_metrics[name]
            lines.append(
                f"| {name} | In-Sample | {m.n_bets} | {m.ats.ats_pct:.1f}% | "
                f"[{m.ats.ci_low:.1f}, {m.ats.ci_high:.1f}] | {m.roi.roi_pct:+.1f}% | "
                f"[{m.roi.ci_low:+.1f}, {m.roi.ci_high:+.1f}] |"
            )

        if name in out_sample_metrics:
            m = out_sample_metrics[name]
            lines.append(
                f"| {name} | **OOS** | {m.n_bets} | **{m.ats.ats_pct:.1f}%** | "
                f"[{m.ats.ci_low:.1f}, {m.ats.ci_high:.1f}] | **{m.roi.roi_pct:+.1f}%** | "
                f"[{m.roi.ci_low:+.1f}, {m.roi.ci_high:+.1f}] |"
            )

    lines.append("")

    # Matched comparison
    lines.append("## Matched Comparison (Apples-to-Apples)")
    lines.append("")
    lines.append("Matching bet counts by week (top-N per week where N = min(phase2, weighted)):")
    lines.append("")
    lines.append("| Calibration | Matched Bets | ATS% | ATS CI | ROI% | ROI CI |")
    lines.append("|-------------|--------------|------|--------|------|--------|")

    for name in sorted(matched_metrics.keys()):
        m = matched_metrics[name]
        lines.append(
            f"| {name} | {m.n_bets} | {m.ats.ats_pct:.1f}% | "
            f"[{m.ats.ci_low:.1f}, {m.ats.ci_high:.1f}] | {m.roi.roi_pct:+.1f}% | "
            f"[{m.roi.ci_low:+.1f}, {m.roi.ci_high:+.1f}] |"
        )

    lines.append("")

    # Threshold comparison
    lines.append("## EV Threshold Comparison (OOS)")
    lines.append("")
    lines.append("| Threshold | Calibration | Bets | ATS% | ROI% |")
    lines.append("|-----------|-------------|------|------|------|")

    for _, row in threshold_comparison.iterrows():
        cal_name = "phase2" if row['calibration'] == 'cal1' else "weighted"
        lines.append(
            f"| {row['threshold']:.0%} | {cal_name} | {row['n_bets']} | "
            f"{row['ats_pct']:.1f}% | {row['roi_pct']:+.1f}% |"
        )

    lines.append("")

    # EV Decile Reliability
    lines.append("## EV Decile Reliability (OOS)")
    lines.append("")

    for name, reliability in decile_reliability.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- **Spearman Correlation (EV vs ATS):** {reliability.spearman_corr:.3f}")
        lines.append(f"- **Monotonic:** {'Yes ✓' if reliability.is_monotonic else 'No ⚠️'}")
        lines.append(f"- **Violations:** {reliability.monotonicity_violations}")
        lines.append("")

        if len(reliability.decile_stats) > 0:
            lines.append("| Decile | EV Range | N | ATS% | ROI% |")
            lines.append("|--------|----------|---|------|------|")

            for _, row in reliability.decile_stats.iterrows():
                lines.append(
                    f"| {int(row['decile'])} | [{row['ev_min']:.1%}, {row['ev_max']:.1%}] | "
                    f"{int(row['n_bets'])} | {row['ats_pct']:.1f}% | {row['roi_pct']:+.1f}% |"
                )

            lines.append("")

    # Recommendation
    lines.append("## 🎯 2026 Recommendation")
    lines.append("")
    lines.append(f"**Recommended Calibration:** `{recommendation.recommended_calibration}`")
    lines.append(f"**Confidence:** {recommendation.confidence.upper()}")
    lines.append(f"**Mode:** {recommendation.mode}")
    lines.append("")
    lines.append(f"**Rationale:** {recommendation.rationale}")
    lines.append("")
    lines.append("**Operational Constraints:**")
    for c in recommendation.constraints:
        lines.append(f"- {c}")
    lines.append("")

    with open(filepath, 'w') as f:
        f.write("\n".join(lines))

    logger.info(f"Saved report: {filepath}")
    return str(filepath)


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_oos_validation(
    train_years: list[int],
    test_years: list[int],
    ats_export_path: str,
    artifact_dir: str,
    phase1_policy: str = "skip",
    min_ev: float = 0.03,
) -> dict:
    """Run full out-of-sample validation workflow with statistical support."""
    print("\n" + "=" * 80)
    print("SPREAD EV CALIBRATION OOS VALIDATION (with Statistical Support)")
    print("=" * 80)
    print(f"Train years: {train_years}")
    print(f"Test years: {test_years}")
    print(f"Phase 1 policy: {phase1_policy}")
    print(f"Min EV: {min_ev:.1%}")
    print("=" * 80)

    if set(train_years) & set(test_years):
        raise ValueError(f"Train and test years overlap: {set(train_years) & set(test_years)}")

    # Load data
    print("\n--- Loading Data ---")
    raw_df = pd.read_csv(ats_export_path)
    print(f"Loaded {len(raw_df)} rows from {ats_export_path}")

    df = load_and_normalize_game_data(raw_df)
    df = detect_placeholder_odds(df)
    print(f"Normalized: {len(df)} games with valid spreads")

    for year in train_years + test_years:
        if year not in df['year'].values:
            raise ValueError(f"Year {year} not in data")

    # Fit calibrations on training data only
    print("\n--- Fitting Calibrations (Train Only) ---")
    calibrations = {}

    # Phase 2 only
    print("\nPhase 2 Only (weeks 4-15):")
    cal_phase2 = fit_calibration_on_years(df, train_years, week_filter=PHASE2_WEEKS)
    calibrations['phase2_only'] = cal_phase2
    print(f"  slope={cal_phase2.slope:.4f}, breakeven={cal_phase2.implied_breakeven_edge:.1f}p")

    # Weighted
    print("\nWeighted (Phase 1 = 0.25):")
    train_mask = df['year'].isin(train_years)
    weights = np.where(df.loc[train_mask, 'week'] <= 3, 0.25, 1.0)
    cal_weighted = fit_calibration_on_years(df[train_mask], train_years, week_filter=None, sample_weights=weights)
    calibrations['weighted'] = cal_weighted
    print(f"  slope={cal_weighted.slope:.4f}, breakeven={cal_weighted.implied_breakeven_edge:.1f}p")

    # Evaluate in-sample
    print("\n--- In-Sample Evaluation (Train Years) ---")
    in_sample_metrics = {}

    for name, cal in calibrations.items():
        if name == 'phase2_only':
            week_filter = PHASE2_WEEKS
        else:
            week_filter = None

        metrics, _ = evaluate_oos_with_ci(df, cal, train_years, week_filter=week_filter, min_ev=min_ev)
        in_sample_metrics[name] = metrics
        print(f"  {name}: {metrics.n_bets} bets, {metrics.ats}")

    # Evaluate out-of-sample
    print("\n--- OUT-OF-SAMPLE Evaluation (Test Years) ---")
    out_sample_metrics = {}
    oos_bets = {}

    for name, cal in calibrations.items():
        if name == 'phase2_only':
            week_filter = PHASE2_WEEKS
        else:
            week_filter = None

        metrics, bets_df = evaluate_oos_with_ci(df, cal, test_years, week_filter=week_filter, min_ev=min_ev)
        out_sample_metrics[name] = metrics
        oos_bets[name] = bets_df
        print(f"  {name}: {metrics.n_bets} bets, {metrics.ats}")

    # Matched comparison (Mode 1: top-N by week)
    print("\n--- Matched Comparison (Top-N by Week) ---")
    matched_metrics = {}

    if 'phase2_only' in oos_bets and 'weighted' in oos_bets:
        # Apply calibrations to full test data for matched comparison
        test_df = df[df['year'].isin(test_years)].copy()

        test_df_p2 = apply_calibration_to_df(test_df, calibrations['phase2_only'])
        test_df_w = apply_calibration_to_df(test_df, calibrations['weighted'])

        # Filter to EV >= threshold
        bets_p2 = test_df_p2[(test_df_p2['ev_cal'] >= min_ev) & ~test_df_p2['push']]
        bets_w = test_df_w[(test_df_w['ev_cal'] >= min_ev) & ~test_df_w['push']]

        matched_p2, matched_w = matched_top_n_by_week(bets_p2, bets_w)

        for name, matched_df in [('phase2_only', matched_p2), ('weighted', matched_w)]:
            if len(matched_df) > 0:
                wins = int(matched_df['jp_side_covered'].sum())
                losses = len(matched_df) - wins
                ats = compute_ats_with_ci(wins, losses)
                roi = compute_roi_with_ci(wins, losses, 0)

                matched_metrics[name] = OOSMetrics(
                    n_bets=len(matched_df),
                    wins=wins,
                    losses=losses,
                    pushes=0,
                    ats=ats,
                    roi=roi,
                    avg_ev=matched_df['ev_cal'].mean(),
                    avg_edge=matched_df['edge_abs'].mean(),
                    breakeven_edge=calibrations[name].implied_breakeven_edge,
                    phase="matched",
                    years=test_years,
                    odds_placeholder_pct=100.0,
                )
                print(f"  {name} (matched): {len(matched_df)} bets, {ats}")

    # Threshold comparison (Mode 2)
    print("\n--- EV Threshold Comparison ---")
    test_df = df[df['year'].isin(test_years)].copy()
    test_df_p2 = apply_calibration_to_df(test_df, calibrations['phase2_only'])
    test_df_w = apply_calibration_to_df(test_df, calibrations['weighted'])

    thresholds = [0.01, 0.02, 0.03, 0.05, 0.07]
    threshold_comparison = compare_at_thresholds(
        test_df_p2, test_df_w, thresholds,
        ev_col='ev_cal', outcome_col='jp_side_covered', push_col='push'
    )
    print(threshold_comparison.to_string(index=False))

    # EV Decile Reliability
    print("\n--- EV Decile Reliability (OOS) ---")
    decile_reliability = {}

    for name, bets_df in oos_bets.items():
        if len(bets_df) >= 10:
            reliability = compute_ev_decile_reliability(
                bets_df, ev_col='ev_cal', outcome_col='jp_side_covered', push_col='push'
            )
            decile_reliability[name] = reliability
            print(f"  {name}: Spearman r={reliability.spearman_corr:.3f}, Monotonic={reliability.is_monotonic}")

    # Generate recommendation
    print("\n--- Generating Recommendation ---")
    recommendation = generate_recommendation(
        phase2_oos=out_sample_metrics['phase2_only'].roi,
        weighted_oos=out_sample_metrics['weighted'].roi,
        phase2_ats=out_sample_metrics['phase2_only'].ats,
        weighted_ats=out_sample_metrics['weighted'].ats,
        matched_phase2_ats=matched_metrics.get('phase2_only', {}).ats if 'phase2_only' in matched_metrics else None,
        matched_weighted_ats=matched_metrics.get('weighted', {}).ats if 'weighted' in matched_metrics else None,
    )
    print(f"  Recommended: {recommendation.recommended_calibration} ({recommendation.confidence} confidence)")

    # Save artifacts
    print("\n--- Saving Artifacts ---")
    artifact_paths = []

    for name, cal in calibrations.items():
        if name == 'phase2_only':
            week_filter = PHASE2_WEEKS
        else:
            week_filter = None

        oos_m = out_sample_metrics.get(name)
        path = save_calibration_artifact(cal, name, train_years, week_filter, artifact_dir, oos_m)
        artifact_paths.append(path)

    # Save enhanced report
    report_path = save_enhanced_oos_report(
        in_sample_metrics, out_sample_metrics, matched_metrics,
        threshold_comparison, decile_reliability, recommendation,
        artifact_dir, test_years
    )
    artifact_paths.append(report_path)

    # Print summary
    print("\n" + "=" * 80)
    print("OOS VALIDATION SUMMARY (with 95% CIs)")
    print("=" * 80)
    print()
    print(f"{'Calibration':<15} {'Sample':<10} {'Bets':>5} | {'ATS%':>7} {'ATS CI':<15} | {'ROI%':>7} {'ROI CI':<18}")
    print("-" * 90)

    for name in sorted(calibrations.keys()):
        if name in in_sample_metrics:
            m = in_sample_metrics[name]
            ats_ci = f"[{m.ats.ci_low:.1f}, {m.ats.ci_high:.1f}]"
            roi_ci = f"[{m.roi.ci_low:+.1f}, {m.roi.ci_high:+.1f}]"
            print(f"{name:<15} {'In-Sample':<10} {m.n_bets:>5} | {m.ats.ats_pct:>6.1f}% {ats_ci:<15} | {m.roi.roi_pct:>+6.1f}% {roi_ci:<18}")

        if name in out_sample_metrics:
            m = out_sample_metrics[name]
            ats_ci = f"[{m.ats.ci_low:.1f}, {m.ats.ci_high:.1f}]"
            roi_ci = f"[{m.roi.ci_low:+.1f}, {m.roi.ci_high:+.1f}]"
            print(f"{name:<15} {'**OOS**':<10} {m.n_bets:>5} | {m.ats.ats_pct:>6.1f}% {ats_ci:<15} | {m.roi.roi_pct:>+6.1f}% {roi_ci:<18}")

    print("-" * 90)

    # Matched summary
    print()
    print("MATCHED COMPARISON (equal bets per week):")
    print("-" * 70)
    for name in sorted(matched_metrics.keys()):
        m = matched_metrics[name]
        ats_ci = f"[{m.ats.ci_low:.1f}, {m.ats.ci_high:.1f}]"
        roi_ci = f"[{m.roi.ci_low:+.1f}, {m.roi.ci_high:+.1f}]"
        print(f"  {name:<15} {m.n_bets:>5} bets | {m.ats.ats_pct:>6.1f}% {ats_ci:<15} | {m.roi.roi_pct:>+6.1f}% {roi_ci:<18}")
    print("-" * 70)

    print()
    print("🎯 RECOMMENDATION FOR 2026:")
    print(f"   Calibration: {recommendation.recommended_calibration}")
    print(f"   Confidence:  {recommendation.confidence}")
    print(f"   Mode:        {recommendation.mode}")
    print(f"   Rationale:   {recommendation.rationale}")
    print()
    print("   Constraints:")
    for c in recommendation.constraints:
        print(f"     - {c}")
    print()
    print("⚠️  All ROI calculations assume -110 juice (placeholder odds)")
    print()

    return {
        'calibrations': calibrations,
        'in_sample_metrics': in_sample_metrics,
        'out_sample_metrics': out_sample_metrics,
        'matched_metrics': matched_metrics,
        'threshold_comparison': threshold_comparison,
        'decile_reliability': decile_reliability,
        'recommendation': recommendation,
        'artifact_paths': artifact_paths,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Out-of-sample spread EV calibration validation with statistical support"
    )
    parser.add_argument(
        "--train-years", nargs="+", type=int, default=DEFAULT_TRAIN_YEARS,
        help="Years to train calibration on"
    )
    parser.add_argument(
        "--test-years", nargs="+", type=int, default=DEFAULT_TEST_YEARS,
        help="Years to evaluate (held out)"
    )
    parser.add_argument(
        "--ats-export", type=str, default=DEFAULT_ATS_EXPORT,
        help="Path to ATS export CSV"
    )
    parser.add_argument(
        "--artifact-dir", type=str, default=DEFAULT_ARTIFACT_DIR,
        help="Directory for output artifacts"
    )
    parser.add_argument(
        "--phase1-policy", type=str, default="skip",
        choices=["skip", "weighted", "phase1_only"],
        help="How to handle Phase 1 (weeks 1-3)"
    )
    parser.add_argument(
        "--min-ev", type=float, default=0.03,
        help="Minimum EV threshold for bet selection"
    )

    args = parser.parse_args()

    run_oos_validation(
        train_years=args.train_years,
        test_years=args.test_years,
        ats_export_path=args.ats_export,
        artifact_dir=args.artifact_dir,
        phase1_policy=args.phase1_policy,
        min_ev=args.min_ev,
    )


if __name__ == "__main__":
    main()
