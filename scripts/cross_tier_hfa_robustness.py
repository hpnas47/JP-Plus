#!/usr/bin/env python3
"""
Cross-Tier HFA Robustness Checks

Validates the +7.16 P4 home vs G5 finding before implementation:
1a. Blowout check: exclude 30+ pt margins
1b. Conference pair breakdown
1c. Year stability
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

# Sub-tier definitions for conference pair analysis
ELITE_P4 = {"SEC", "Big Ten"}
OTHER_P4 = {"ACC", "Big 12", "FBS Independents"}


def classify_tier(conference: str) -> str:
    """Classify conference as P4 or G5."""
    if conference in P4_CONFERENCES:
        return "P4"
    elif conference in G5_CONFERENCES:
        return "G5"
    return "Unknown"


def classify_subtier(conference: str) -> str:
    """Classify conference into sub-tiers."""
    if conference in ELITE_P4:
        return "Elite P4"
    elif conference in OTHER_P4:
        return "Other P4"
    elif conference in G5_CONFERENCES:
        return "G5"
    return "Unknown"


def run_robustness_checks(years: list = [2022, 2023, 2024, 2025]):
    """Run robustness checks on cross-tier HFA finding."""

    print("=" * 60)
    print("CROSS-TIER HFA ROBUSTNESS CHECKS")
    print("=" * 60)

    # Fetch data and run backtest
    print("\nFetching data and running backtest...")
    season_data = fetch_all_season_data(years, use_priors=True, use_cache=True)

    # Build conference lookup
    client = CFBDClient()
    conf_lookup = {}
    for year in years:
        teams = client.get_fbs_teams(year=year)
        for team in teams:
            if team.school and team.conference:
                conf_lookup[(year, team.school)] = team.conference

    # Run backtest
    results = run_backtest(
        years=years,
        start_week=1,
        hfa_global_offset=0.50,
        season_data=season_data,
    )

    df = results["predictions"]
    print(f"\nTotal games: {len(df)}")

    # Add conference info
    df["home_conference"] = df.apply(
        lambda r: conf_lookup.get((r["year"], r["home_team"]), "Unknown"), axis=1
    )
    df["away_conference"] = df.apply(
        lambda r: conf_lookup.get((r["year"], r["away_team"]), "Unknown"), axis=1
    )

    # Classify tiers
    df["home_tier"] = df["home_conference"].apply(classify_tier)
    df["away_tier"] = df["away_conference"].apply(classify_tier)
    df["home_subtier"] = df["home_conference"].apply(classify_subtier)
    df["away_subtier"] = df["away_conference"].apply(classify_subtier)

    # Filter to cross-tier games
    cross_tier_mask = (
        ((df["home_tier"] == "G5") & (df["away_tier"] == "P4")) |
        ((df["home_tier"] == "P4") & (df["away_tier"] == "G5"))
    )
    cross_tier = df[cross_tier_mask].copy()
    print(f"Cross-tier games: {len(cross_tier)}")

    # Compute residual margin (actual - base, isolates HFA from team quality)
    cross_tier["base_margin"] = cross_tier["home_rating"] - cross_tier["away_rating"]
    cross_tier["residual_margin"] = cross_tier["actual_margin"] - cross_tier["base_margin"]
    cross_tier["g5_is_home"] = cross_tier["home_tier"] == "G5"
    cross_tier["abs_margin"] = cross_tier["actual_margin"].abs()

    # =========================================================================
    # CHECK 1A: BLOWOUT EXCLUSION
    # =========================================================================
    print("\n" + "=" * 60)
    print("CHECK 1A: BLOWOUT EXCLUSION (30+ pt margins)")
    print("=" * 60)

    # Full dataset
    p4_home_all = cross_tier[~cross_tier["g5_is_home"]]
    g5_home_all = cross_tier[cross_tier["g5_is_home"]]

    p4_home_residual_all = p4_home_all["residual_margin"].mean()
    g5_home_residual_all = g5_home_all["residual_margin"].mean()

    print(f"\nFull dataset:")
    print(f"  P4 Home vs G5 (N={len(p4_home_all)}): Residual HFA = {p4_home_residual_all:+.2f}")
    print(f"  G5 Home vs P4 (N={len(g5_home_all)}): Residual HFA = {g5_home_residual_all:+.2f}")

    # Exclude blowouts
    non_blowout = cross_tier[cross_tier["abs_margin"] < 30]
    p4_home_no_blow = non_blowout[~non_blowout["g5_is_home"]]
    g5_home_no_blow = non_blowout[non_blowout["g5_is_home"]]

    p4_home_residual_no_blow = p4_home_no_blow["residual_margin"].mean()
    g5_home_residual_no_blow = g5_home_no_blow["residual_margin"].mean()

    blowout_pct = (len(cross_tier) - len(non_blowout)) / len(cross_tier) * 100

    print(f"\nExcluding 30+ pt margins ({len(cross_tier) - len(non_blowout)} games, {blowout_pct:.1f}%):")
    print(f"  P4 Home vs G5 (N={len(p4_home_no_blow)}): Residual HFA = {p4_home_residual_no_blow:+.2f}")
    print(f"  G5 Home vs P4 (N={len(g5_home_no_blow)}): Residual HFA = {g5_home_residual_no_blow:+.2f}")

    blowout_check_passed = p4_home_residual_no_blow >= 4.0
    print(f"\n  BLOWOUT CHECK: {'PASSED' if blowout_check_passed else 'FAILED'} (P4 home residual {'>=' if blowout_check_passed else '<'} +4.0)")

    blowout_results = {
        "full_dataset": {
            "p4_home_n": len(p4_home_all),
            "p4_home_residual": round(p4_home_residual_all, 2),
            "g5_home_n": len(g5_home_all),
            "g5_home_residual": round(g5_home_residual_all, 2),
        },
        "excluding_blowouts": {
            "blowout_games_excluded": len(cross_tier) - len(non_blowout),
            "blowout_pct": round(blowout_pct, 1),
            "p4_home_n": len(p4_home_no_blow),
            "p4_home_residual": round(p4_home_residual_no_blow, 2),
            "g5_home_n": len(g5_home_no_blow),
            "g5_home_residual": round(g5_home_residual_no_blow, 2),
        },
        "passed": blowout_check_passed,
    }

    # =========================================================================
    # CHECK 1B: CONFERENCE PAIR BREAKDOWN
    # =========================================================================
    print("\n" + "=" * 60)
    print("CHECK 1B: CONFERENCE PAIR BREAKDOWN")
    print("=" * 60)

    pair_results = {}

    # P4 home scenarios
    p4_home = cross_tier[~cross_tier["g5_is_home"]]

    # Elite P4 (SEC/Big Ten) home vs G5
    elite_home = p4_home[p4_home["home_subtier"] == "Elite P4"]
    elite_home_residual = elite_home["residual_margin"].mean() if len(elite_home) > 0 else 0

    # Other P4 (ACC/Big 12/Ind) home vs G5
    other_home = p4_home[p4_home["home_subtier"] == "Other P4"]
    other_home_residual = other_home["residual_margin"].mean() if len(other_home) > 0 else 0

    print(f"\nP4 Home vs G5:")
    print(f"  SEC/Big Ten home (N={len(elite_home)}): Residual HFA = {elite_home_residual:+.2f}")
    print(f"  ACC/Big12/Ind home (N={len(other_home)}): Residual HFA = {other_home_residual:+.2f}")

    pair_results["p4_home_vs_g5"] = {
        "sec_bigten_home": {"n": len(elite_home), "residual_hfa": round(elite_home_residual, 2)},
        "acc_big12_ind_home": {"n": len(other_home), "residual_hfa": round(other_home_residual, 2)},
    }

    # G5 home scenarios
    g5_home = cross_tier[cross_tier["g5_is_home"]]

    # G5 home vs Elite P4 (SEC/Big Ten)
    g5_vs_elite = g5_home[g5_home["away_subtier"] == "Elite P4"]
    g5_vs_elite_residual = g5_vs_elite["residual_margin"].mean() if len(g5_vs_elite) > 0 else 0

    # G5 home vs Other P4 (ACC/Big 12/Ind)
    g5_vs_other = g5_home[g5_home["away_subtier"] == "Other P4"]
    g5_vs_other_residual = g5_vs_other["residual_margin"].mean() if len(g5_vs_other) > 0 else 0

    print(f"\nG5 Home vs P4:")
    print(f"  vs SEC/Big Ten (N={len(g5_vs_elite)}): Residual HFA = {g5_vs_elite_residual:+.2f}")
    print(f"  vs ACC/Big12/Ind (N={len(g5_vs_other)}): Residual HFA = {g5_vs_other_residual:+.2f}")

    pair_results["g5_home_vs_p4"] = {
        "vs_sec_bigten": {"n": len(g5_vs_elite), "residual_hfa": round(g5_vs_elite_residual, 2)},
        "vs_acc_big12_ind": {"n": len(g5_vs_other), "residual_hfa": round(g5_vs_other_residual, 2)},
    }

    # Check consistency (within 3 pts)
    p4_home_diff = abs(elite_home_residual - other_home_residual)
    g5_home_diff = abs(g5_vs_elite_residual - g5_vs_other_residual)

    pair_check_passed = (p4_home_diff <= 3.0) and (g5_home_diff <= 3.0)
    print(f"\n  P4 home spread: {p4_home_diff:.2f} pts")
    print(f"  G5 home spread: {g5_home_diff:.2f} pts")
    print(f"  PAIR CHECK: {'PASSED' if pair_check_passed else 'FAILED'} (both spreads <= 3.0 pts)")

    pair_results["p4_home_spread"] = round(p4_home_diff, 2)
    pair_results["g5_home_spread"] = round(g5_home_diff, 2)
    pair_results["passed"] = pair_check_passed

    # =========================================================================
    # CHECK 1C: YEAR STABILITY
    # =========================================================================
    print("\n" + "=" * 60)
    print("CHECK 1C: YEAR STABILITY")
    print("=" * 60)

    year_results = {}
    p4_home_by_year = []
    g5_home_by_year = []

    for year in years:
        year_data = cross_tier[cross_tier["year"] == year]
        p4_home_year = year_data[~year_data["g5_is_home"]]
        g5_home_year = year_data[year_data["g5_is_home"]]

        p4_residual = p4_home_year["residual_margin"].mean() if len(p4_home_year) > 0 else 0
        g5_residual = g5_home_year["residual_margin"].mean() if len(g5_home_year) > 0 else 0

        p4_home_by_year.append(p4_residual)
        g5_home_by_year.append(g5_residual)

        print(f"\n{year}:")
        print(f"  P4 Home vs G5 (N={len(p4_home_year)}): Residual HFA = {p4_residual:+.2f}")
        print(f"  G5 Home vs P4 (N={len(g5_home_year)}): Residual HFA = {g5_residual:+.2f}")

        year_results[year] = {
            "p4_home_n": len(p4_home_year),
            "p4_home_residual": round(p4_residual, 2),
            "g5_home_n": len(g5_home_year),
            "g5_home_residual": round(g5_residual, 2),
        }

    # Check year-over-year stability
    p4_std = np.std(p4_home_by_year)
    g5_std = np.std(g5_home_by_year)
    p4_range = max(p4_home_by_year) - min(p4_home_by_year)
    g5_range = max(g5_home_by_year) - min(g5_home_by_year)

    # Check: no single year driving result (all years same sign, range not too large)
    p4_all_positive = all(r > 0 for r in p4_home_by_year)
    g5_range_ok = g5_range <= 5.0  # More lenient for G5 home (smaller sample)
    p4_range_ok = p4_range <= 6.0  # P4 home is the main signal

    year_check_passed = p4_all_positive and p4_range_ok

    print(f"\nYear Stability:")
    print(f"  P4 Home: std={p4_std:.2f}, range={p4_range:.2f}")
    print(f"  G5 Home: std={g5_std:.2f}, range={g5_range:.2f}")
    print(f"  P4 all positive: {p4_all_positive}")
    print(f"  YEAR CHECK: {'PASSED' if year_check_passed else 'FAILED'} (P4 all positive, range <= 6.0)")

    year_results["stability"] = {
        "p4_std": round(p4_std, 2),
        "p4_range": round(p4_range, 2),
        "g5_std": round(g5_std, 2),
        "g5_range": round(g5_range, 2),
        "passed": year_check_passed,
    }

    # =========================================================================
    # OVERALL ASSESSMENT
    # =========================================================================
    print("\n" + "=" * 60)
    print("OVERALL ASSESSMENT")
    print("=" * 60)

    all_passed = blowout_check_passed and pair_check_passed and year_check_passed

    print(f"\n  Blowout Check:    {'PASSED' if blowout_check_passed else 'FAILED'}")
    print(f"  Pair Check:       {'PASSED' if pair_check_passed else 'FAILED'}")
    print(f"  Year Check:       {'PASSED' if year_check_passed else 'FAILED'}")
    print(f"\n  OVERALL:          {'PROCEED TO IMPLEMENTATION' if all_passed else 'NEEDS REVIEW'}")

    # Calculate recommended adjustments (with 0.75 shrinkage)
    base_hfa = 2.30  # Current net HFA

    # Use blowout-excluded values for robustness
    p4_home_empirical = p4_home_residual_no_blow
    g5_home_empirical = g5_home_residual_no_blow

    cross_tier_bonus = (p4_home_empirical - base_hfa) * 0.75
    cross_tier_penalty = (base_hfa - g5_home_empirical) * 0.75

    print(f"\n  Recommended Adjustments (0.75 shrinkage on blowout-excluded):")
    print(f"    P4 Home empirical: {p4_home_empirical:+.2f}")
    print(f"    G5 Home empirical: {g5_home_empirical:+.2f}")
    print(f"    cross_tier_bonus (P4 home): {cross_tier_bonus:+.2f}")
    print(f"    cross_tier_penalty (G5 home): {cross_tier_penalty:+.2f}")

    results = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "years": years,
        "n_cross_tier_games": len(cross_tier),
        "blowout_check": blowout_results,
        "conference_pairs": pair_results,
        "year_stability": year_results,
        "all_passed": all_passed,
        "recommended_adjustments": {
            "base_hfa": base_hfa,
            "p4_home_empirical": round(p4_home_empirical, 2),
            "g5_home_empirical": round(g5_home_empirical, 2),
            "shrinkage": 0.75,
            "cross_tier_bonus": round(cross_tier_bonus, 2),
            "cross_tier_penalty": round(cross_tier_penalty, 2),
        },
    }

    return results


def generate_report(results: dict) -> str:
    """Generate markdown report."""

    report = f"""# Cross-Tier HFA Robustness Checks

**Generated:** {results['generated']}
**Data:** {results['years']} ({results['n_cross_tier_games']} cross-tier games)

## Summary

| Check | Result | Details |
|-------|--------|---------|
| **Blowout Exclusion** | {'PASSED' if results['blowout_check']['passed'] else 'FAILED'} | P4 home residual {results['blowout_check']['excluding_blowouts']['p4_home_residual']:+.2f} after excluding 30+ pt margins |
| **Conference Pairs** | {'PASSED' if results['conference_pairs']['passed'] else 'FAILED'} | P4 spread: {results['conference_pairs']['p4_home_spread']:.2f}, G5 spread: {results['conference_pairs']['g5_home_spread']:.2f} |
| **Year Stability** | {'PASSED' if results['year_stability']['stability']['passed'] else 'FAILED'} | P4 range: {results['year_stability']['stability']['p4_range']:.2f}, all years positive: {all(r['p4_home_residual'] > 0 for r in results['year_stability'].values() if isinstance(r, dict) and 'p4_home_residual' in r)} |

**Overall: {'PROCEED TO IMPLEMENTATION' if results['all_passed'] else 'NEEDS REVIEW'}**

---

## 1a. Blowout Check

Does the P4 home advantage hold when excluding 30+ point margins?

### Full Dataset

| Scenario | N | Residual HFA |
|----------|---|--------------|
| P4 Home vs G5 | {results['blowout_check']['full_dataset']['p4_home_n']} | {results['blowout_check']['full_dataset']['p4_home_residual']:+.2f} |
| G5 Home vs P4 | {results['blowout_check']['full_dataset']['g5_home_n']} | {results['blowout_check']['full_dataset']['g5_home_residual']:+.2f} |

### Excluding 30+ Point Margins

| Scenario | N | Residual HFA |
|----------|---|--------------|
| P4 Home vs G5 | {results['blowout_check']['excluding_blowouts']['p4_home_n']} | {results['blowout_check']['excluding_blowouts']['p4_home_residual']:+.2f} |
| G5 Home vs P4 | {results['blowout_check']['excluding_blowouts']['g5_home_n']} | {results['blowout_check']['excluding_blowouts']['g5_home_residual']:+.2f} |

**Games excluded:** {results['blowout_check']['excluding_blowouts']['blowout_games_excluded']} ({results['blowout_check']['excluding_blowouts']['blowout_pct']:.1f}%)

**Verdict:** {'The effect holds above +4.0 after blowout exclusion.' if results['blowout_check']['passed'] else 'Effect drops below +4.0 — may be inflated by garbage time.'}

---

## 1b. Conference Pair Breakdown

Is the effect consistent across different P4 conferences?

### P4 Home vs G5

| Home Conference | N | Residual HFA |
|-----------------|---|--------------|
| SEC/Big Ten | {results['conference_pairs']['p4_home_vs_g5']['sec_bigten_home']['n']} | {results['conference_pairs']['p4_home_vs_g5']['sec_bigten_home']['residual_hfa']:+.2f} |
| ACC/Big12/Ind | {results['conference_pairs']['p4_home_vs_g5']['acc_big12_ind_home']['n']} | {results['conference_pairs']['p4_home_vs_g5']['acc_big12_ind_home']['residual_hfa']:+.2f} |

**Spread:** {results['conference_pairs']['p4_home_spread']:.2f} pts

### G5 Home vs P4

| Visitor Conference | N | Residual HFA |
|--------------------|---|--------------|
| vs SEC/Big Ten | {results['conference_pairs']['g5_home_vs_p4']['vs_sec_bigten']['n']} | {results['conference_pairs']['g5_home_vs_p4']['vs_sec_bigten']['residual_hfa']:+.2f} |
| vs ACC/Big12/Ind | {results['conference_pairs']['g5_home_vs_p4']['vs_acc_big12_ind']['n']} | {results['conference_pairs']['g5_home_vs_p4']['vs_acc_big12_ind']['residual_hfa']:+.2f} |

**Spread:** {results['conference_pairs']['g5_home_spread']:.2f} pts

**Verdict:** {'Effect is consistent across conference pairs (within 3 pts).' if results['conference_pairs']['passed'] else 'Effect varies too much across conference pairs — may need granular adjustment.'}

---

## 1c. Year Stability

Is the effect consistent across years?

| Year | P4 Home N | P4 Home Residual | G5 Home N | G5 Home Residual |
|------|-----------|------------------|-----------|------------------|
"""

    for year in results['years']:
        yr = results['year_stability'][year]
        report += f"| {year} | {yr['p4_home_n']} | {yr['p4_home_residual']:+.2f} | {yr['g5_home_n']} | {yr['g5_home_residual']:+.2f} |\n"

    report += f"""
**P4 Home Stability:** std={results['year_stability']['stability']['p4_std']:.2f}, range={results['year_stability']['stability']['p4_range']:.2f}
**G5 Home Stability:** std={results['year_stability']['stability']['g5_std']:.2f}, range={results['year_stability']['stability']['g5_range']:.2f}

**Verdict:** {'Effect is stable across years — no single year driving the result.' if results['year_stability']['stability']['passed'] else 'Effect varies too much across years — may be unstable.'}

---

## Recommended Implementation

Based on blowout-excluded empirical values with 0.75 shrinkage:

| Parameter | Value |
|-----------|-------|
| Base HFA | {results['recommended_adjustments']['base_hfa']:+.2f} |
| P4 Home Empirical | {results['recommended_adjustments']['p4_home_empirical']:+.2f} |
| G5 Home Empirical | {results['recommended_adjustments']['g5_home_empirical']:+.2f} |
| **cross_tier_bonus** (P4 home) | **{results['recommended_adjustments']['cross_tier_bonus']:+.2f}** |
| **cross_tier_penalty** (G5 home) | **{results['recommended_adjustments']['cross_tier_penalty']:+.2f}** |

### Application

```python
# For cross-tier games:
if p4_is_home:
    hfa = base_hfa + cross_tier_bonus  # {results['recommended_adjustments']['base_hfa']:+.2f} + {results['recommended_adjustments']['cross_tier_bonus']:+.2f} = {results['recommended_adjustments']['base_hfa'] + results['recommended_adjustments']['cross_tier_bonus']:+.2f}
elif g5_is_home:
    hfa = base_hfa - cross_tier_penalty  # {results['recommended_adjustments']['base_hfa']:+.2f} - {results['recommended_adjustments']['cross_tier_penalty']:+.2f} = {results['recommended_adjustments']['base_hfa'] - results['recommended_adjustments']['cross_tier_penalty']:+.2f}
```
"""

    return report


def main():
    """Run robustness checks."""

    results = run_robustness_checks()

    # Save JSON
    output_dir = Path("data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "cross_tier_hfa_robustness.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nJSON saved to: {json_path}")

    # Generate and save markdown report
    report = generate_report(results)
    md_path = output_dir / "cross_tier_hfa_robustness.md"
    with open(md_path, "w") as f:
        f.write(report)
    print(f"Report saved to: {md_path}")

    print("\n" + "=" * 60)
    print("ROBUSTNESS CHECKS COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
