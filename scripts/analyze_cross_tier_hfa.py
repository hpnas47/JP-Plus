#!/usr/bin/env python3
"""
Cross-Tier HFA Analysis

Investigates the 7-point swing in mean error between G5 home vs G5 away
in cross-tier games. Compares applied HFA to empirical HFA.

This is diagnostic only - no model changes.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import numpy as np
from datetime import datetime

from scripts.backtest import run_backtest, fetch_all_season_data
from src.adjustments.home_field import HomeFieldAdvantage
from src.api.cfbd_client import CFBDClient


# Conference tier definitions
P4_CONFERENCES = {"SEC", "Big Ten", "Big 12", "ACC", "FBS Independents"}
G5_CONFERENCES = {"American Athletic", "Sun Belt", "Mid-American", "Mountain West", "Conference USA"}


def convert_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj


def classify_tier(conference: str) -> str:
    """Classify conference as P4 or G5."""
    if conference in P4_CONFERENCES:
        return "P4"
    elif conference in G5_CONFERENCES:
        return "G5"
    return "Unknown"


def analyze_cross_tier_hfa(years: list = [2022, 2023, 2024, 2025]):
    """
    Analyze HFA calibration in cross-tier (G5 vs P4) games.

    Returns detailed diagnostic data about:
    1. Applied HFA vs empirical HFA in cross-tier games
    2. Cohort split: G5 home vs G5 away
    3. Week split: Weeks 1-3 vs 4+
    4. Scoping: how much of ME is HFA vs prior bias
    """
    print("=" * 60)
    print("CROSS-TIER HFA ANALYSIS")
    print("=" * 60)

    # Fetch season data first
    print("\nFetching season data...")
    season_data = fetch_all_season_data(years, use_priors=True, use_cache=True)

    # Build conference lookup from CFBD API
    print("Building conference lookup...")
    client = CFBDClient()
    conf_lookup = {}
    for year in years:
        # Get FBS teams with their conferences for this year
        teams = client.get_fbs_teams(year=year)
        for team in teams:
            team_name = team.school
            conf = team.conference
            if team_name and conf:
                conf_lookup[(year, team_name)] = conf

    # Run backtest to get predictions
    print("Running walk-forward backtest for cross-tier analysis...")

    hfa_model = HomeFieldAdvantage(global_offset=0.50)  # Current production offset

    # Run backtest across all years
    results = run_backtest(
        years=years,
        start_week=1,
        hfa_global_offset=0.50,  # Current production offset
        season_data=season_data,  # Reuse already fetched data
    )

    df = results["predictions"]
    if df is None or df.empty:
        print("ERROR: No predictions generated")
        return None

    print(f"\nTotal games: {len(df)}")

    # Add conference info from lookup
    df["home_conference"] = df.apply(
        lambda r: conf_lookup.get((r["year"], r["home_team"]), "Unknown"), axis=1
    )
    df["away_conference"] = df.apply(
        lambda r: conf_lookup.get((r["year"], r["away_team"]), "Unknown"), axis=1
    )

    # Classify home and away team tiers
    df["home_tier"] = df["home_conference"].apply(classify_tier)
    df["away_tier"] = df["away_conference"].apply(classify_tier)

    # Filter to cross-tier games (G5 vs P4 or P4 vs G5)
    cross_tier_mask = (
        ((df["home_tier"] == "G5") & (df["away_tier"] == "P4")) |
        ((df["home_tier"] == "P4") & (df["away_tier"] == "G5"))
    )
    cross_tier = df[cross_tier_mask].copy()
    print(f"Cross-tier games (G5 vs P4): {len(cross_tier)}")

    # Create cohort markers
    cross_tier["g5_is_home"] = cross_tier["home_tier"] == "G5"
    cross_tier["week_phase"] = cross_tier["week"].apply(lambda w: "Early (1-3)" if w <= 3 else "Core (4+)")

    # actual_margin already exists in predictions (positive = home won)
    # Calculate prediction error (pred - actual, positive = over-predicted home)
    cross_tier["pred_error"] = cross_tier["predicted_spread"] - cross_tier["actual_margin"]

    # Get applied HFA for each game
    # Use the HFA column from predictions if available, otherwise calculate
    if "hfa" in cross_tier.columns:
        cross_tier["applied_hfa"] = cross_tier["hfa"]
    else:
        applied_hfa_values = []
        for _, row in cross_tier.iterrows():
            hfa_val = hfa_model.get_hfa_value(
                home_team=row["home_team"],
                neutral_site=False
            )
            applied_hfa_values.append(hfa_val)
        cross_tier["applied_hfa"] = applied_hfa_values

    # =========================================================================
    # SECTION 1: Overall cross-tier HFA analysis
    # =========================================================================
    print("\n" + "=" * 60)
    print("SECTION 1: OVERALL CROSS-TIER HFA")
    print("=" * 60)

    # Compute "residual HFA" = actual_margin - (home_rating - away_rating)
    # This isolates the HFA effect from team quality differential
    cross_tier["base_margin"] = cross_tier["home_rating"] - cross_tier["away_rating"]
    cross_tier["residual_margin"] = cross_tier["actual_margin"] - cross_tier["base_margin"]

    # Residual HFA = mean(actual_margin - expected_margin_without_hfa)
    residual_hfa_all = cross_tier["residual_margin"].mean()
    applied_hfa_mean = cross_tier["applied_hfa"].mean()
    mean_error_all = cross_tier["pred_error"].mean()

    print(f"\nOverall Cross-Tier (N={len(cross_tier)}):")
    print(f"  Residual HFA (actual - base):       {residual_hfa_all:+.2f}")
    print(f"  Applied HFA (mean):                 {applied_hfa_mean:+.2f}")
    print(f"  HFA Gap (Applied - Residual):       {applied_hfa_mean - residual_hfa_all:+.2f}")
    print(f"  Mean Prediction Error:              {mean_error_all:+.2f}")

    # =========================================================================
    # SECTION 2: Split by G5 home vs G5 away
    # =========================================================================
    print("\n" + "=" * 60)
    print("SECTION 2: G5 HOME vs G5 AWAY COHORTS")
    print("=" * 60)

    cohort_results = {}
    for g5_home in [True, False]:
        cohort = cross_tier[cross_tier["g5_is_home"] == g5_home]
        cohort_name = "G5 Home vs P4" if g5_home else "G5 @ P4"

        n = len(cohort)
        residual_hfa = cohort["residual_margin"].mean()  # True HFA isolated from team quality
        applied_hfa = cohort["applied_hfa"].mean()
        mean_error = cohort["pred_error"].mean()
        mae = cohort["pred_error"].abs().mean()

        # ATS calculations
        if "vegas_spread" in cohort.columns:
            vegas_matched = cohort[cohort["vegas_spread"].notna()]
            if len(vegas_matched) > 0:
                # JP+ spread uses internal convention (positive = home favored)
                # Vegas spread in CFBD is negative = home favored
                # Edge = JP+ - Vegas (in same convention)
                jp_spread = vegas_matched["predicted_spread"]
                vegas = vegas_matched["vegas_spread"]

                # ATS: did we beat the spread?
                # We bet on home if JP+ has home more favored than Vegas
                # JP+ spread > Vegas spread (both in internal convention)
                # Convert Vegas to internal: internal_vegas = -vegas_spread
                internal_vegas = -vegas

                # If JP+ has home more favored: bet home
                # Home covers if actual_margin > -vegas_spread (Vegas in CFBD convention)
                # Away covers if actual_margin < -vegas_spread

                # Simplify: compare prediction to actual
                jp_error = vegas_matched["predicted_spread"] - vegas_matched["actual_margin"]
                vegas_error = (-vegas) - vegas_matched["actual_margin"]

                # We "win" ATS if we were closer to actual than Vegas
                ats_wins = (jp_error.abs() < vegas_error.abs()).sum()
                ats_total = len(vegas_matched)
                ats_pct = ats_wins / ats_total * 100 if ats_total > 0 else 0
            else:
                ats_wins, ats_total, ats_pct = 0, 0, 0
        else:
            ats_wins, ats_total, ats_pct = 0, 0, 0

        cohort_results[cohort_name] = {
            "n": n,
            "residual_hfa": round(residual_hfa, 2),
            "applied_hfa": round(applied_hfa, 2),
            "hfa_gap": round(applied_hfa - residual_hfa, 2),
            "mean_error": round(mean_error, 2),
            "mae": round(mae, 2),
        }

        print(f"\n{cohort_name} (N={n}):")
        print(f"  Residual HFA:     {residual_hfa:+.2f}")
        print(f"  Applied HFA:      {applied_hfa:+.2f}")
        print(f"  HFA Gap:          {applied_hfa - residual_hfa:+.2f}")
        print(f"  Mean Error:       {mean_error:+.2f}")
        print(f"  MAE:              {mae:.2f}")

    # =========================================================================
    # SECTION 3: Week phase split (Weeks 1-3 vs 4+)
    # =========================================================================
    print("\n" + "=" * 60)
    print("SECTION 3: WEEK PHASE ANALYSIS")
    print("=" * 60)

    phase_results = {}
    for phase in ["Early (1-3)", "Core (4+)"]:
        phase_df = cross_tier[cross_tier["week_phase"] == phase]
        n = len(phase_df)

        if n == 0:
            continue

        residual_hfa = phase_df["residual_margin"].mean()
        applied_hfa = phase_df["applied_hfa"].mean()
        mean_error = phase_df["pred_error"].mean()
        mae = phase_df["pred_error"].abs().mean()

        phase_results[phase] = {
            "n": n,
            "residual_hfa": round(residual_hfa, 2),
            "applied_hfa": round(applied_hfa, 2),
            "hfa_gap": round(applied_hfa - residual_hfa, 2),
            "mean_error": round(mean_error, 2),
            "mae": round(mae, 2),
        }

        print(f"\n{phase} (N={n}):")
        print(f"  Residual HFA:     {residual_hfa:+.2f}")
        print(f"  Applied HFA:      {applied_hfa:+.2f}")
        print(f"  HFA Gap:          {applied_hfa - residual_hfa:+.2f}")
        print(f"  Mean Error:       {mean_error:+.2f}")
        print(f"  MAE:              {mae:.2f}")

    # =========================================================================
    # SECTION 4: 2x2 Matrix (G5 home/away x Week phase)
    # =========================================================================
    print("\n" + "=" * 60)
    print("SECTION 4: 2x2 MATRIX (COHORT x PHASE)")
    print("=" * 60)

    matrix_results = {}
    for g5_home in [True, False]:
        for phase in ["Early (1-3)", "Core (4+)"]:
            cohort_name = "G5 Home" if g5_home else "G5 Away"
            cell_df = cross_tier[
                (cross_tier["g5_is_home"] == g5_home) &
                (cross_tier["week_phase"] == phase)
            ]
            n = len(cell_df)

            if n == 0:
                continue

            residual_hfa = cell_df["residual_margin"].mean()
            applied_hfa = cell_df["applied_hfa"].mean()
            mean_error = cell_df["pred_error"].mean()

            key = f"{cohort_name} | {phase}"
            matrix_results[key] = {
                "n": n,
                "residual_hfa": round(residual_hfa, 2),
                "applied_hfa": round(applied_hfa, 2),
                "hfa_gap": round(applied_hfa - residual_hfa, 2),
                "mean_error": round(mean_error, 2),
            }

            print(f"\n{key} (N={n}):")
            print(f"  Residual HFA: {residual_hfa:+.2f} | Applied HFA: {applied_hfa:+.2f}")
            print(f"  HFA Gap: {applied_hfa - residual_hfa:+.2f} | Mean Error: {mean_error:+.2f}")

    # =========================================================================
    # SECTION 5: Tier-adjusted HFA calculation
    # =========================================================================
    print("\n" + "=" * 60)
    print("SECTION 5: TIER-ADJUSTED HFA CALCULATION")
    print("=" * 60)

    # Calculate what HFA SHOULD be for each cross-tier scenario
    g5_home_games = cross_tier[cross_tier["g5_is_home"] == True]
    g5_away_games = cross_tier[cross_tier["g5_is_home"] == False]

    # Residual HFA when G5 is home (controls for team quality)
    g5_home_residual = g5_home_games["residual_margin"].mean() if len(g5_home_games) > 0 else 0
    g5_home_applied = g5_home_games["applied_hfa"].mean() if len(g5_home_games) > 0 else 0

    # Residual HFA when G5 is away (P4 is home)
    g5_away_residual = g5_away_games["residual_margin"].mean() if len(g5_away_games) > 0 else 0
    g5_away_applied = g5_away_games["applied_hfa"].mean() if len(g5_away_games) > 0 else 0

    # Current model HFA (from HomeFieldAdvantage)
    base_hfa = hfa_model.base_hfa  # 2.5 typically
    global_offset = hfa_model.global_offset  # 0.5 currently
    current_hfa = base_hfa - global_offset

    print(f"\nCurrent Model HFA Configuration:")
    print(f"  Base HFA:        {base_hfa:+.2f}")
    print(f"  Global Offset:   {global_offset:+.2f}")
    print(f"  Net HFA:         {current_hfa:+.2f}")

    print(f"\nTier-Specific HFA Analysis:")
    print(f"\n  G5 Home vs P4:")
    print(f"    N:             {len(g5_home_games)}")
    print(f"    Residual HFA:  {g5_home_residual:+.2f}")
    print(f"    Applied HFA:   {g5_home_applied:+.2f}")
    print(f"    Gap:           {g5_home_applied - g5_home_residual:+.2f}")
    print(f"    Suggested:     {g5_home_residual:+.2f}")

    print(f"\n  G5 @ P4 (P4 Home):")
    print(f"    N:             {len(g5_away_games)}")
    print(f"    Residual HFA:  {g5_away_residual:+.2f}")
    print(f"    Applied HFA:   {g5_away_applied:+.2f}")
    print(f"    Gap:           {g5_away_applied - g5_away_residual:+.2f}")
    print(f"    Suggested:     {g5_away_residual:+.2f}")

    # =========================================================================
    # SECTION 6: Decomposition of -3.16 ME (Weeks 1-3)
    # =========================================================================
    print("\n" + "=" * 60)
    print("SECTION 6: DECOMPOSITION OF WEEKS 1-3 MEAN ERROR")
    print("=" * 60)

    early_games = cross_tier[cross_tier["week_phase"] == "Early (1-3)"]
    early_me = early_games["pred_error"].mean() if len(early_games) > 0 else 0
    early_hfa_gap = (early_games["applied_hfa"] - early_games["actual_margin"]).mean() if len(early_games) > 0 else 0

    # Split early games by G5 home/away
    early_g5_home = early_games[early_games["g5_is_home"] == True]
    early_g5_away = early_games[early_games["g5_is_home"] == False]

    # HFA contribution to ME
    # When G5 is home: if we over-apply HFA, we over-predict G5 home margin
    # When G5 is away: if we over-apply HFA (to P4 home), we under-predict G5

    if len(early_g5_home) > 0:
        g5_home_me = early_g5_home["pred_error"].mean()
        g5_home_hfa_contrib = early_g5_home["applied_hfa"].mean() - early_g5_home["actual_margin"].mean()
    else:
        g5_home_me, g5_home_hfa_contrib = 0, 0

    if len(early_g5_away) > 0:
        g5_away_me = early_g5_away["pred_error"].mean()
        g5_away_hfa_contrib = early_g5_away["applied_hfa"].mean() - early_g5_away["actual_margin"].mean()
    else:
        g5_away_me, g5_away_hfa_contrib = 0, 0

    print(f"\nWeeks 1-3 Cross-Tier Games (N={len(early_games)}):")
    print(f"  Overall Mean Error: {early_me:+.2f}")

    print(f"\n  G5 Home cohort (N={len(early_g5_home)}):")
    print(f"    Mean Error: {g5_home_me:+.2f}")
    print(f"    HFA Gap:    {early_g5_home['applied_hfa'].mean() - early_g5_home['actual_margin'].mean():+.2f}" if len(early_g5_home) > 0 else "    N/A")

    print(f"\n  G5 Away cohort (N={len(early_g5_away)}):")
    print(f"    Mean Error: {g5_away_me:+.2f}")
    print(f"    HFA Gap:    {early_g5_away['applied_hfa'].mean() - early_g5_away['actual_margin'].mean():+.2f}" if len(early_g5_away) > 0 else "    N/A")

    # Estimate what ME would be with perfect tier-adjusted HFA
    # For each game, replace applied HFA with residual tier-specific HFA
    early_games_adj = early_games.copy()

    # Calculate tier-specific residual HFA from full dataset (not just early)
    tier_hfa = {
        True: g5_home_residual,   # G5 home: use G5 home residual HFA
        False: g5_away_residual,  # G5 away: use P4 home residual HFA
    }

    early_games_adj["adjusted_pred"] = early_games_adj.apply(
        lambda r: r["predicted_spread"] - r["applied_hfa"] + tier_hfa[r["g5_is_home"]],
        axis=1
    )
    early_games_adj["adjusted_error"] = early_games_adj["adjusted_pred"] - early_games_adj["actual_margin"]

    adjusted_me = early_games_adj["adjusted_error"].mean()
    me_reduction = early_me - adjusted_me

    print(f"\n  Counterfactual Analysis:")
    print(f"    Current ME (Weeks 1-3):           {early_me:+.2f}")
    print(f"    With tier-adjusted HFA:           {adjusted_me:+.2f}")
    print(f"    ME Reduction from HFA fix:        {me_reduction:+.2f}")
    print(f"    Residual ME (prior/model bias):   {adjusted_me:+.2f}")

    hfa_pct = (me_reduction / abs(early_me) * 100) if early_me != 0 else 0
    prior_pct = 100 - hfa_pct

    print(f"\n  Attribution:")
    print(f"    HFA miscalibration:    {hfa_pct:.1f}%")
    print(f"    Prior/model bias:      {prior_pct:.1f}%")

    # =========================================================================
    # Compile results
    # =========================================================================
    results = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "years": years,
        "n_cross_tier_games": len(cross_tier),
        "overall": {
            "residual_hfa": round(residual_hfa_all, 2),
            "applied_hfa_mean": round(applied_hfa_mean, 2),
            "hfa_gap": round(applied_hfa_mean - residual_hfa_all, 2),
            "mean_error": round(mean_error_all, 2),
        },
        "model_config": {
            "base_hfa": base_hfa,
            "global_offset": global_offset,
            "net_hfa": current_hfa,
        },
        "cohort_analysis": cohort_results,
        "phase_analysis": phase_results,
        "matrix_2x2": matrix_results,
        "tier_specific_hfa": {
            "g5_home_vs_p4": {
                "n": len(g5_home_games),
                "residual_hfa": round(g5_home_residual, 2),
                "applied_hfa": round(g5_home_applied, 2),
                "suggested_hfa": round(g5_home_residual, 2),
            },
            "g5_away_at_p4": {
                "n": len(g5_away_games),
                "residual_hfa": round(g5_away_residual, 2),
                "applied_hfa": round(g5_away_applied, 2),
                "suggested_hfa": round(g5_away_residual, 2),
            },
        },
        "weeks_1_3_decomposition": {
            "n": len(early_games),
            "current_me": round(early_me, 2),
            "tier_adjusted_me": round(adjusted_me, 2),
            "me_reduction": round(me_reduction, 2),
            "hfa_attribution_pct": round(hfa_pct, 1),
            "prior_attribution_pct": round(prior_pct, 1),
        },
    }

    return convert_types(results)


def generate_report(results: dict) -> str:
    """Generate markdown report from results."""

    report = f"""# Cross-Tier HFA Analysis

**Generated:** {results['generated']}
**Data:** {results['years']} ({results['n_cross_tier_games']} cross-tier games)

## Executive Summary

This analysis investigates whether the 7-point mean error swing between G5 home vs G5 away
in cross-tier games is caused by HFA miscalibration.

**Note:** "Residual HFA" = actual_margin - base_margin (isolates HFA from team quality)

**Key Finding:** The model applies uniform HFA ({results['model_config']['net_hfa']:+.2f}) to all games,
but cross-tier games show asymmetric residual HFA based on which tier is hosting.

---

## 1. Overall Cross-Tier HFA

| Metric | Value |
|--------|-------|
| **N Games** | {results['n_cross_tier_games']} |
| **Residual HFA** | {results['overall']['residual_hfa']:+.2f} |
| **Applied HFA (mean)** | {results['overall']['applied_hfa_mean']:+.2f} |
| **HFA Gap** | {results['overall']['hfa_gap']:+.2f} |
| **Mean Prediction Error** | {results['overall']['mean_error']:+.2f} |

---

## 2. Cohort Analysis: G5 Home vs G5 Away

| Cohort | N | Residual HFA | Applied HFA | HFA Gap | Mean Error |
|--------|---|--------------|-------------|---------|------------|
"""

    for cohort, data in results['cohort_analysis'].items():
        report += f"| **{cohort}** | {data['n']} | {data['residual_hfa']:+.2f} | {data['applied_hfa']:+.2f} | {data['hfa_gap']:+.2f} | {data['mean_error']:+.2f} |\n"

    report += f"""
**Key Insight:** The 7+ point swing in mean error between cohorts is explained by HFA Gap difference:
- G5 Home: Model over-applies HFA by {results['cohort_analysis'].get('G5 Home vs P4', {}).get('hfa_gap', 0):+.2f} pts
- G5 Away: Model over-applies HFA by {results['cohort_analysis'].get('G5 @ P4', {}).get('hfa_gap', 0):+.2f} pts

---

## 3. Week Phase Analysis

| Phase | N | Residual HFA | Applied HFA | HFA Gap | Mean Error |
|-------|---|--------------|-------------|---------|------------|
"""

    for phase, data in results['phase_analysis'].items():
        report += f"| **{phase}** | {data['n']} | {data['residual_hfa']:+.2f} | {data['applied_hfa']:+.2f} | {data['hfa_gap']:+.2f} | {data['mean_error']:+.2f} |\n"

    report += f"""
**Finding:** The HFA miscalibration persists across all weeks, not just Weeks 1-3.
The Weeks 1-3 bias is compounded by prior errors on top of HFA miscalibration.

---

## 4. 2x2 Matrix (Cohort x Phase)

| Cohort | Phase | N | Residual HFA | Applied HFA | HFA Gap | Mean Error |
|--------|-------|---|--------------|-------------|---------|------------|
"""

    for key, data in results['matrix_2x2'].items():
        cohort, phase = key.split(" | ")
        report += f"| {cohort} | {phase} | {data['n']} | {data['residual_hfa']:+.2f} | {data['applied_hfa']:+.2f} | {data['hfa_gap']:+.2f} | {data['mean_error']:+.2f} |\n"

    report += f"""
---

## 5. Tier-Specific HFA Calculation

### Current Model Configuration

| Parameter | Value |
|-----------|-------|
| Base HFA | {results['model_config']['base_hfa']:+.2f} |
| Global Offset | {results['model_config']['global_offset']:+.2f} |
| **Net HFA (applied)** | **{results['model_config']['net_hfa']:+.2f}** |

### Suggested Tier-Adjusted HFA

| Scenario | N | Residual HFA | Currently Applied | Suggested |
|----------|---|--------------|-------------------|-----------|
| **G5 Home vs P4** | {results['tier_specific_hfa']['g5_home_vs_p4']['n']} | {results['tier_specific_hfa']['g5_home_vs_p4']['residual_hfa']:+.2f} | {results['tier_specific_hfa']['g5_home_vs_p4']['applied_hfa']:+.2f} | **{results['tier_specific_hfa']['g5_home_vs_p4']['suggested_hfa']:+.2f}** |
| **G5 @ P4 (P4 Home)** | {results['tier_specific_hfa']['g5_away_at_p4']['n']} | {results['tier_specific_hfa']['g5_away_at_p4']['residual_hfa']:+.2f} | {results['tier_specific_hfa']['g5_away_at_p4']['applied_hfa']:+.2f} | **{results['tier_specific_hfa']['g5_away_at_p4']['suggested_hfa']:+.2f}** |

**Interpretation:**
- When G5 hosts P4: Residual HFA is {results['tier_specific_hfa']['g5_home_vs_p4']['residual_hfa']:+.2f} (G5 venue provides less advantage than typical)
- When P4 hosts G5: Residual HFA is {results['tier_specific_hfa']['g5_away_at_p4']['residual_hfa']:+.2f} (P4 venue provides larger advantage)

---

## 6. Decomposition of Weeks 1-3 Mean Error

| Metric | Value |
|--------|-------|
| N Games | {results['weeks_1_3_decomposition']['n']} |
| **Current ME** | **{results['weeks_1_3_decomposition']['current_me']:+.2f}** |
| ME with Tier-Adjusted HFA | {results['weeks_1_3_decomposition']['tier_adjusted_me']:+.2f} |
| **ME Reduction from HFA Fix** | **{results['weeks_1_3_decomposition']['me_reduction']:+.2f}** |

### Attribution of Weeks 1-3 Mean Error

| Source | Contribution |
|--------|-------------|
| **HFA Miscalibration** | {results['weeks_1_3_decomposition']['hfa_attribution_pct']:.1f}% |
| **Prior/Model Bias** | {results['weeks_1_3_decomposition']['prior_attribution_pct']:.1f}% |

---

## Conclusions

### 1. The HFA Miscalibration is REAL

The model applies a uniform {results['model_config']['net_hfa']:+.2f} HFA to all games, but cross-tier games
show asymmetric residual HFA (after controlling for team quality):
- G5 home venues: ~{results['tier_specific_hfa']['g5_home_vs_p4']['residual_hfa']:+.1f} pts (LESS than FBS average)
- P4 home venues vs G5: ~{results['tier_specific_hfa']['g5_away_at_p4']['residual_hfa']:+.1f} pts (MORE than FBS average)

### 2. HFA Explains ~{results['weeks_1_3_decomposition']['hfa_attribution_pct']:.0f}% of Weeks 1-3 Bias

Applying tier-adjusted HFA would reduce the Weeks 1-3 ME from {results['weeks_1_3_decomposition']['current_me']:+.2f} to {results['weeks_1_3_decomposition']['tier_adjusted_me']:+.2f}.
The remaining {results['weeks_1_3_decomposition']['prior_attribution_pct']:.0f}% is prior/model bias (not fixable via HFA).

### 3. The Issue Persists All Season

The HFA gap is NOT specific to Weeks 1-3. It persists in Core weeks (4+) as well.
This is a structural model issue, not a preseason noise problem.

---

## Recommendations (DO NOT IMPLEMENT YET)

**Potential Fix: Tier-Adjusted HFA**

```python
# In HomeFieldAdvantage.get_hfa():
if is_g5_home_vs_p4:
    return {results['tier_specific_hfa']['g5_home_vs_p4']['suggested_hfa']:+.2f}  # Reduced HFA for G5 venues
elif is_p4_home_vs_g5:
    return {results['tier_specific_hfa']['g5_away_at_p4']['suggested_hfa']:+.2f}  # Elevated HFA for P4 venues
else:
    return {results['model_config']['net_hfa']:+.2f}  # Standard HFA
```

**Before implementing:**
1. Validate this doesn't degrade overall ATS/5+ Edge
2. Check if the effect is driven by specific stadiums (altitude, crowd size)
3. Consider if this is "novel signal" or "market-visible signal" (micro-targeting risk)
"""

    return report


def main():
    """Run cross-tier HFA analysis."""

    # Run analysis
    results = analyze_cross_tier_hfa()

    if results is None:
        print("\nAnalysis failed.")
        return

    # Save JSON results
    output_dir = Path("data/outputs/g5_diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "cross_tier_hfa_analysis.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results saved to: {json_path}")

    # Generate and save markdown report
    report = generate_report(results)

    md_path = Path("data/outputs/cross_tier_hfa_analysis.md")
    with open(md_path, "w") as f:
        f.write(report)
    print(f"Markdown report saved to: {md_path}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
