#!/usr/bin/env python3
"""
G5 Prior Analysis - Connect G5 vs P4 bias to priors system.

Analyzes whether own-priors (JP+ historical) vs SP+ priors
explain the -3.16 mean error in Weeks 1-3 G5 vs P4 games.

Output: data/outputs/g5_prior_analysis.md
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import polars as pl

from scripts.backtest import fetch_all_season_data, run_backtest
from src.api.cfbd_client import CFBDClient
from src.models.preseason_priors import PreseasonPriors
from src.models.own_priors import OwnPriorGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Conference tiers
P4_CONFERENCES = {"SEC", "Big Ten", "Big 12", "ACC", "FBS Independents"}
G5_CONFERENCES = {"American Athletic", "Sun Belt", "Mid-American", "Mountain West", "Conference USA"}
P4_INDEPENDENTS = {"Notre Dame", "BYU"}


def get_team_tier(team: str, conf: str) -> str:
    if team in P4_INDEPENDENTS:
        return "P4"
    if conf in P4_CONFERENCES:
        return "P4"
    if conf in G5_CONFERENCES:
        return "G5"
    return "OTHER"


def load_historical_ratings() -> dict:
    """Load historical JP+ ratings."""
    path = Path("data/historical_jp_ratings.json")
    if not path.exists():
        logger.error(f"Historical ratings not found: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def fetch_sp_ratings(client: CFBDClient, year: int) -> dict[str, float]:
    """Fetch SP+ ratings for prior year (as preseason priors)."""
    try:
        sp_ratings = client.get_sp_ratings(year=year - 1)
        return {r.team: r.rating for r in sp_ratings if r.team and r.rating}
    except Exception as e:
        logger.warning(f"Could not fetch SP+ for {year - 1}: {e}")
        return {}


def analyze_g5_priors(years: list[int]) -> dict:
    """Analyze SP+ vs own-prior performance for G5 teams in early season."""
    client = CFBDClient()
    historical_ratings = load_historical_ratings()

    if not historical_ratings:
        logger.error("Cannot proceed without historical ratings")
        return {}

    # Fetch all season data
    logger.info("Fetching season data...")
    season_data = fetch_all_season_data(years, use_priors=True, use_portal=True)

    # Run backtest to get predictions
    logger.info("Running backtest...")
    results = run_backtest(years=years, start_week=1, season_data=season_data)

    pred_df = results.get("predictions")
    if pred_df is None or pred_df.empty:
        logger.error("No predictions!")
        return {}

    # Get conference mappings by year
    conf_by_year = {}
    for year in years:
        try:
            teams = client.get_fbs_teams(year=year)
            conf_by_year[year] = {t.school: t.conference for t in teams if t.school and t.conference}
        except Exception as e:
            logger.warning(f"Could not fetch conferences for {year}: {e}")
            conf_by_year[year] = {}

    # Add tier info to predictions
    def get_conf(row, team_col):
        return conf_by_year.get(row["year"], {}).get(row[team_col], "UNKNOWN")

    pred_df = pred_df.copy()
    pred_df["home_conf"] = pred_df.apply(lambda r: get_conf(r, "home_team"), axis=1)
    pred_df["away_conf"] = pred_df.apply(lambda r: get_conf(r, "away_team"), axis=1)
    pred_df["home_tier"] = pred_df.apply(lambda r: get_team_tier(r["home_team"], r["home_conf"]), axis=1)
    pred_df["away_tier"] = pred_df.apply(lambda r: get_team_tier(r["away_team"], r["away_conf"]), axis=1)

    # Filter to G5 vs P4, Weeks 1-3
    def is_g5_vs_p4(row):
        return (
            (row["home_tier"] == "G5" and row["away_tier"] == "P4") or
            (row["home_tier"] == "P4" and row["away_tier"] == "G5")
        )

    pred_df["is_g5_vs_p4"] = pred_df.apply(is_g5_vs_p4, axis=1)
    early_g5_p4 = pred_df[(pred_df["is_g5_vs_p4"]) & (pred_df["week"] <= 3)].copy()

    logger.info(f"Found {len(early_g5_p4)} Weeks 1-3 G5 vs P4 games")

    # Identify which team is G5
    early_g5_p4["g5_team"] = early_g5_p4.apply(
        lambda r: r["home_team"] if r["home_tier"] == "G5" else r["away_team"], axis=1
    )
    early_g5_p4["p4_team"] = early_g5_p4.apply(
        lambda r: r["home_team"] if r["home_tier"] == "P4" else r["away_team"], axis=1
    )
    early_g5_p4["g5_is_home"] = early_g5_p4["home_tier"] == "G5"
    early_g5_p4["g5_conf"] = early_g5_p4.apply(
        lambda r: r["home_conf"] if r["g5_is_home"] else r["away_conf"], axis=1
    )

    # Calculate SP+ and own-prior ratings for each G5 team
    # and compare to end-of-season JP+ rating
    analysis_rows = []

    for year in years:
        year_games = early_g5_p4[early_g5_p4["year"] == year]
        if len(year_games) == 0:
            continue

        # Get SP+ priors (from prior year)
        sp_ratings = fetch_sp_ratings(client, year)

        # Calculate own-priors
        own_gen = OwnPriorGenerator(historical_ratings, client=client)
        try:
            own_priors = own_gen.calculate_preseason_ratings(year)
        except Exception as e:
            logger.warning(f"Could not calculate own-priors for {year}: {e}")
            own_priors = {}

        # Get end-of-season ratings for this year
        end_of_season = historical_ratings.get(str(year), {})

        # Get conference mapping for this year
        conf_map = conf_by_year.get(year, {})

        for _, row in year_games.iterrows():
            g5_team = row["g5_team"]
            p4_team = row["p4_team"]
            g5_conf = row["g5_conf"]

            # Get various ratings for G5 team
            sp_prior = sp_ratings.get(g5_team)
            own_prior_obj = own_priors.get(g5_team)
            own_prior = own_prior_obj.combined_rating if own_prior_obj else None
            eos_rating = end_of_season.get(g5_team, {}).get("overall")

            # Get ratings for P4 team too
            sp_prior_p4 = sp_ratings.get(p4_team)
            own_prior_obj_p4 = own_priors.get(p4_team)
            own_prior_p4 = own_prior_obj_p4.combined_rating if own_prior_obj_p4 else None
            eos_rating_p4 = end_of_season.get(p4_team, {}).get("overall")

            # Calculate spread differentials (home - away perspective)
            if row["g5_is_home"]:
                # G5 is home, so spread diff = G5 - P4 + HFA
                sp_diff = (sp_prior - sp_prior_p4) if (sp_prior and sp_prior_p4) else None
                own_diff = (own_prior - own_prior_p4) if (own_prior and own_prior_p4) else None
                eos_diff = (eos_rating - eos_rating_p4) if (eos_rating and eos_rating_p4) else None
            else:
                # P4 is home, so spread diff = P4 - G5 + HFA
                sp_diff = (sp_prior_p4 - sp_prior) if (sp_prior and sp_prior_p4) else None
                own_diff = (own_prior_p4 - own_prior) if (own_prior and own_prior_p4) else None
                eos_diff = (eos_rating_p4 - eos_rating) if (eos_rating and eos_rating_p4) else None

            analysis_rows.append({
                "year": year,
                "week": row["week"],
                "game_id": row["game_id"],
                "g5_team": g5_team,
                "p4_team": p4_team,
                "g5_conf": g5_conf,
                "g5_is_home": row["g5_is_home"],
                "sp_prior_g5": sp_prior,
                "own_prior_g5": own_prior,
                "eos_rating_g5": eos_rating,
                "sp_prior_p4": sp_prior_p4,
                "own_prior_p4": own_prior_p4,
                "eos_rating_p4": eos_rating_p4,
                "sp_diff": sp_diff,
                "own_diff": own_diff,
                "eos_diff": eos_diff,
                "predicted_spread": row["predicted_spread"],
                "actual_margin": row["actual_margin"],
                "error": row["error"],  # predicted - actual
            })

    analysis_df = pd.DataFrame(analysis_rows)
    logger.info(f"Built analysis DataFrame with {len(analysis_df)} rows")

    # Calculate summary statistics
    results = {
        "n_games": len(analysis_df),
        "years": years,
    }

    # 1. Compare SP+ vs Own-Prior for G5 teams
    # The question: does own-prior rate G5 teams lower (closer to reality)?
    valid_both = analysis_df[(analysis_df["sp_prior_g5"].notna()) & (analysis_df["own_prior_g5"].notna())]

    if len(valid_both) > 0:
        # Calculate mean G5 rating under each prior system
        mean_sp_g5 = valid_both["sp_prior_g5"].mean()
        mean_own_g5 = valid_both["own_prior_g5"].mean()
        mean_eos_g5 = valid_both["eos_rating_g5"].mean()

        # Calculate implied spread error for each prior system
        # Error = (implied spread from prior) - (actual margin)
        # For G5 vs P4, if we overrate G5, we predict G5 does better than actual

        results["g5_rating_comparison"] = {
            "n": len(valid_both),
            "mean_sp_prior_g5": round(mean_sp_g5, 2),
            "mean_own_prior_g5": round(mean_own_g5, 2),
            "mean_eos_rating_g5": round(mean_eos_g5, 2),
            "sp_overrates_g5_by": round(mean_sp_g5 - mean_eos_g5, 2),
            "own_overrates_g5_by": round(mean_own_g5 - mean_eos_g5, 2),
        }

        # Calculate spread-level prediction errors
        # For each game, what would the spread error be under SP+ vs Own?
        valid_spreads = valid_both[(valid_both["sp_diff"].notna()) & (valid_both["own_diff"].notna())]
        if len(valid_spreads) > 0:
            # Error when using pure SP+ prior
            # (We don't have the full spread, but we can compare the rating differential)
            # Since actual_margin incorporates HFA etc, we compare diff trends

            # What we really want: does own-prior better predict end-of-season?
            # sp_error = SP+ prior - EOS rating (for G5 team)
            # own_error = Own prior - EOS rating (for G5 team)
            valid_spreads = valid_spreads.copy()
            valid_spreads["sp_error_g5"] = valid_spreads["sp_prior_g5"] - valid_spreads["eos_rating_g5"]
            valid_spreads["own_error_g5"] = valid_spreads["own_prior_g5"] - valid_spreads["eos_rating_g5"]

            results["g5_prior_error"] = {
                "n": len(valid_spreads),
                "sp_mean_error": round(valid_spreads["sp_error_g5"].mean(), 2),
                "own_mean_error": round(valid_spreads["own_error_g5"].mean(), 2),
                "sp_mae": round(valid_spreads["sp_error_g5"].abs().mean(), 2),
                "own_mae": round(valid_spreads["own_error_g5"].abs().mean(), 2),
            }

    # 2. Sun Belt team-level breakdown
    sun_belt = analysis_df[analysis_df["g5_conf"] == "Sun Belt"]
    if len(sun_belt) > 0:
        # Group by team, calculate mean error
        sun_belt_teams = sun_belt.groupby("g5_team").agg({
            "error": ["mean", "count"],
            "sp_prior_g5": "mean",
            "own_prior_g5": "mean",
            "eos_rating_g5": "mean",
        }).reset_index()
        sun_belt_teams.columns = ["team", "mean_error", "n_games", "sp_prior", "own_prior", "eos_rating"]

        # Sort by mean error (most negative = most overrated)
        sun_belt_teams = sun_belt_teams.sort_values("mean_error", ascending=True)

        results["sun_belt_breakdown"] = {
            "n_games": len(sun_belt),
            "n_teams": len(sun_belt_teams),
            "overall_me": round(sun_belt["error"].mean(), 2),
            "top_5_overrated": sun_belt_teams.head(5).to_dict("records"),
            "all_teams": sun_belt_teams.to_dict("records"),
        }

    # 3. By-conference breakdown for all G5
    conf_breakdown = []
    for conf in G5_CONFERENCES:
        conf_games = analysis_df[analysis_df["g5_conf"] == conf]
        if len(conf_games) >= 10:
            valid = conf_games[(conf_games["sp_prior_g5"].notna()) & (conf_games["eos_rating_g5"].notna())]
            if len(valid) >= 10:
                sp_error = (valid["sp_prior_g5"] - valid["eos_rating_g5"]).mean()
                own_error = None
                if valid["own_prior_g5"].notna().any():
                    own_valid = valid[valid["own_prior_g5"].notna()]
                    own_error = (own_valid["own_prior_g5"] - own_valid["eos_rating_g5"]).mean()

                conf_breakdown.append({
                    "conference": conf,
                    "n_games": len(conf_games),
                    "mean_error": round(conf_games["error"].mean(), 2),
                    "sp_overrates_by": round(sp_error, 2),
                    "own_overrates_by": round(own_error, 2) if own_error else None,
                })

    results["conference_prior_breakdown"] = conf_breakdown

    # Save raw data for further analysis
    output_dir = Path("data/outputs/g5_diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_df.to_csv(output_dir / "g5_prior_analysis_raw.csv", index=False)

    return results, analysis_df


def generate_report(results: dict, analysis_df: pd.DataFrame) -> str:
    """Generate markdown report."""
    lines = [
        "# G5 Prior Analysis: Connecting Bias to Priors System",
        "",
        f"**Generated:** 2026-02-11",
        f"**Data:** {results.get('years', [])} ({results.get('n_games', 0)} Weeks 1-3 G5 vs P4 games)",
        "",
        "## Executive Summary",
        "",
        "This analysis investigates whether the -3.16 mean error in Weeks 1-3 G5 vs P4 games",
        "is explained by prior system differences (SP+ vs own-prior).",
        "",
    ]

    # G5 Rating Comparison
    if "g5_rating_comparison" in results:
        c = results["g5_rating_comparison"]
        lines.extend([
            "---",
            "",
            "## 1. SP+ vs Own-Prior: How They Rate G5 Teams",
            "",
            f"Analyzing {c['n']} G5 teams in early-season cross-tier games:",
            "",
            "| Prior System | Mean G5 Rating | Overrates G5 By |",
            "|--------------|----------------|-----------------|",
            f"| **SP+ (prior year)** | {c['mean_sp_prior_g5']:+.2f} | {c['sp_overrates_g5_by']:+.2f} |",
            f"| **Own-Prior (JP+)** | {c['mean_own_prior_g5']:+.2f} | {c['own_overrates_g5_by']:+.2f} |",
            f"| **End-of-Season (truth)** | {c['mean_eos_rating_g5']:+.2f} | — |",
            "",
        ])

        # Interpretation
        sp_over = c['sp_overrates_g5_by']
        own_over = c['own_overrates_g5_by']
        if abs(sp_over) > abs(own_over):
            lines.append(f"**Finding:** SP+ overrates G5 teams by {sp_over:+.2f} pts vs end-of-season, "
                         f"while own-prior overrates by only {own_over:+.2f} pts.")
            lines.append("**Implication:** Own-prior is more accurate for G5 teams. Weighting own-prior more heavily in early weeks could reduce bias.")
        else:
            lines.append(f"**Finding:** Both prior systems show similar G5 overrating: SP+ by {sp_over:+.2f}, own-prior by {own_over:+.2f}.")
            lines.append("**Implication:** The bias may not be fixable by switching between prior systems.")
        lines.append("")

    # G5 Prior Error Comparison
    if "g5_prior_error" in results:
        e = results["g5_prior_error"]
        lines.extend([
            "### Prior Prediction Error (vs End-of-Season Rating)",
            "",
            f"For {e['n']} G5 teams with both SP+ and own-prior available:",
            "",
            "| Metric | SP+ Prior | Own-Prior | Better |",
            "|--------|-----------|-----------|--------|",
            f"| Mean Error (vs EOS) | {e['sp_mean_error']:+.2f} | {e['own_mean_error']:+.2f} | {'Own' if abs(e['own_mean_error']) < abs(e['sp_mean_error']) else 'SP+'} |",
            f"| MAE (vs EOS) | {e['sp_mae']:.2f} | {e['own_mae']:.2f} | {'Own' if e['own_mae'] < e['sp_mae'] else 'SP+'} |",
            "",
        ])

    # Sun Belt Breakdown
    if "sun_belt_breakdown" in results:
        sb = results["sun_belt_breakdown"]
        lines.extend([
            "---",
            "",
            "## 2. Sun Belt Team-Level Breakdown",
            "",
            f"Sun Belt has the worst performance: -3.80 ME, 47.1% ATS in G5 vs P4 games.",
            "",
            f"**{sb['n_games']} games across {sb['n_teams']} teams, Overall ME: {sb['overall_me']:+.2f}**",
            "",
            "### Top 5 Most Overrated Sun Belt Teams (vs P4)",
            "",
            "| Team | N | Mean Error | SP+ Prior | Own-Prior | End-of-Season |",
            "|------|---|------------|-----------|-----------|---------------|",
        ])

        for t in sb["top_5_overrated"]:
            sp = f"{t['sp_prior']:.1f}" if pd.notna(t.get('sp_prior')) else "-"
            own = f"{t['own_prior']:.1f}" if pd.notna(t.get('own_prior')) else "-"
            eos = f"{t['eos_rating']:.1f}" if pd.notna(t.get('eos_rating')) else "-"
            lines.append(f"| {t['team']} | {t['n_games']} | {t['mean_error']:+.2f} | {sp} | {own} | {eos} |")

        lines.extend([
            "",
            "**Interpretation:**",
        ])

        # Check if it's a few teams or conference-wide
        all_teams = sb["all_teams"]
        overrated_teams = [t for t in all_teams if t["mean_error"] < -2.0]
        if len(overrated_teams) >= len(all_teams) / 2:
            lines.append(f"- **Conference-wide issue:** {len(overrated_teams)}/{len(all_teams)} teams are overrated by >2 pts")
        else:
            lines.append(f"- **Concentrated in few teams:** Only {len(overrated_teams)}/{len(all_teams)} teams are overrated by >2 pts")
            lines.append(f"- Top offenders: {', '.join([t['team'] for t in overrated_teams[:3]])}")
        lines.append("")

    # Conference Prior Breakdown
    if "conference_prior_breakdown" in results:
        lines.extend([
            "---",
            "",
            "## 3. By-Conference Prior Analysis",
            "",
            "How much does each prior system overrate G5 teams (vs end-of-season)?",
            "",
            "| Conference | N | Spread ME | SP+ Overrates By | Own Overrates By |",
            "|------------|---|-----------|------------------|------------------|",
        ])

        for c in results["conference_prior_breakdown"]:
            own = f"{c['own_overrates_by']:+.2f}" if c.get('own_overrates_by') else "-"
            lines.append(f"| {c['conference']} | {c['n_games']} | {c['mean_error']:+.2f} | {c['sp_overrates_by']:+.2f} | {own} |")

        lines.append("")

    # Recommendations
    lines.extend([
        "---",
        "",
        "## Conclusions & Recommendations",
        "",
    ])

    if "g5_rating_comparison" in results:
        sp_over = results["g5_rating_comparison"]["sp_overrates_g5_by"]
        own_over = results["g5_rating_comparison"]["own_overrates_g5_by"]

        if sp_over > own_over + 1.0:  # SP+ overrates more by at least 1 pt
            lines.extend([
                "### Evidence Supports G5-Specific Prior Weighting",
                "",
                f"1. **SP+ overrates G5 teams by {sp_over:+.2f} pts** vs end-of-season",
                f"2. **Own-prior overrates by only {own_over:+.2f} pts** — more accurate",
                f"3. **Implication:** For G5 teams in Weeks 1-3, weight own-prior more heavily",
                "",
                "### Suggested Blend Schedule for G5 Teams",
                "",
                "| Week | Current SP+ Weight | Proposed G5 SP+ Weight |",
                "|------|-------------------|------------------------|",
                "| 1 | 65% | 40% |",
                "| 2 | 55% | 35% |",
                "| 3 | 45% | 30% |",
                "",
                "This would shift ~25% more weight to own-prior for G5 teams specifically.",
            ])
        else:
            lines.extend([
                "### Limited Evidence for G5-Specific Prior Weighting",
                "",
                f"Both prior systems show similar G5 overrating:",
                f"- SP+ overrates by {sp_over:+.2f}",
                f"- Own-prior overrates by {own_over:+.2f}",
                "",
                "The bias may be inherent to preseason estimation for G5 teams,",
                "not fixable by choosing between prior systems.",
                "",
                "**Alternative approaches:**",
                "- Apply a G5 preseason discount (e.g., -1.5 pts in Weeks 1-3)",
                "- Weight G5 priors toward conference mean more aggressively",
            ])

    lines.append("")

    return "\n".join(lines)


def main():
    years = [2022, 2023, 2024, 2025]

    logger.info("Starting G5 prior analysis...")
    results, analysis_df = analyze_g5_priors(years)

    if not results:
        logger.error("Analysis failed!")
        return

    # Generate report
    report = generate_report(results, analysis_df)

    # Save report
    output_path = Path("data/outputs/g5_prior_analysis.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    logger.info(f"Saved report to {output_path}")

    # Print report
    print("\n" + "=" * 70)
    print(report)
    print("=" * 70 + "\n")

    # Also save JSON results
    json_path = Path("data/outputs/g5_diagnostics/g5_prior_analysis.json")

    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj) if not np.isnan(obj) else None
        return obj

    with open(json_path, "w") as f:
        json.dump(convert_types(results), f, indent=2)
    logger.info(f"Saved JSON to {json_path}")


if __name__ == "__main__":
    main()
