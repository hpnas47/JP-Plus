#!/usr/bin/env python3
"""
G5 Conference Island Diagnostics

Phase A: Diagnose whether "conference island" / low-connectivity inflation exists
in G5 conferences and whether it harms ATS/CLV performance.

Produces:
1. G5 vs P4 cohort analysis (MAE, mean error, ATS, CLV)
2. Conference connectivity metrics through each week
3. Bias identification by conference

Usage:
    python3 scripts/diagnose_g5_islands.py [--years 2022,2023,2024,2025] [--output-dir data/outputs]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import polars as pl

from scripts.backtest import (
    fetch_all_season_data,
    run_backtest,
    calculate_ats_results,
)
from src.api.cfbd_client import CFBDClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Conference tier classification
P4_CONFERENCES = {"SEC", "Big Ten", "Big 12", "ACC", "FBS Independents"}
G5_CONFERENCES = {"American Athletic", "Sun Belt", "Mid-American", "Mountain West", "Conference USA"}

# Notable independents that should be treated as P4
P4_INDEPENDENTS = {"Notre Dame", "BYU"}  # BYU was independent 2011-2022


def get_team_tier(team: str, conf: str) -> str:
    """Classify a team as P4, G5, or Independent."""
    if team in P4_INDEPENDENTS:
        return "P4"
    if conf in P4_CONFERENCES:
        return "P4"
    if conf in G5_CONFERENCES:
        return "G5"
    if conf == "FBS Independents":
        return "IND"
    return "UNKNOWN"


def fetch_conference_mappings(years: list[int]) -> dict[int, dict[str, str]]:
    """Fetch year-accurate conference mappings for each season."""
    client = CFBDClient()
    conf_by_year = {}

    for year in years:
        try:
            teams = client.get_fbs_teams(year=year)
            conf_map = {t.school: t.conference for t in teams if t.school and t.conference}
            conf_by_year[year] = conf_map
            logger.info(f"Fetched {year} conference data: {len(conf_map)} teams")
        except Exception as e:
            logger.warning(f"Could not fetch conference data for {year}: {e}")
            conf_by_year[year] = {}

    return conf_by_year


def add_tier_info(
    predictions_df: pd.DataFrame,
    conf_by_year: dict[int, dict[str, str]],
) -> pd.DataFrame:
    """Add conference and tier info to predictions."""
    df = predictions_df.copy()

    # Add conference info
    def get_conf(row, team_col):
        year = row["year"]
        team = row[team_col]
        return conf_by_year.get(year, {}).get(team, "UNKNOWN")

    df["home_conf"] = df.apply(lambda r: get_conf(r, "home_team"), axis=1)
    df["away_conf"] = df.apply(lambda r: get_conf(r, "away_team"), axis=1)

    # Add tier info
    df["home_tier"] = df.apply(lambda r: get_team_tier(r["home_team"], r["home_conf"]), axis=1)
    df["away_tier"] = df.apply(lambda r: get_team_tier(r["away_team"], r["away_conf"]), axis=1)

    # Classify game type
    def game_type(row):
        tiers = {row["home_tier"], row["away_tier"]}
        if tiers == {"P4"}:
            return "P4_vs_P4"
        elif tiers == {"G5"}:
            return "G5_vs_G5"
        elif "P4" in tiers and "G5" in tiers:
            return "G5_vs_P4"
        else:
            return "OTHER"

    df["game_type"] = df.apply(game_type, axis=1)

    return df


def calculate_connectivity(
    games_df: pl.DataFrame,
    conf_by_year: dict[int, dict[str, str]],
    year: int,
    max_week: int = 15,
) -> pd.DataFrame:
    """Calculate conference connectivity through each week.

    Connectivity = OOC FBS games played through week < W
    """
    conf_map = conf_by_year.get(year, {})
    if not conf_map:
        return pd.DataFrame()

    # Filter to this year's games
    year_games = games_df.filter(pl.col("year") == year).to_pandas()

    # Add conference info
    year_games["home_conf"] = year_games["home_team"].map(conf_map)
    year_games["away_conf"] = year_games["away_team"].map(conf_map)

    # OOC game = different conferences
    year_games["is_ooc"] = year_games["home_conf"] != year_games["away_conf"]

    # FBS game = both teams have known conferences
    year_games["is_fbs"] = (
        year_games["home_conf"].notna() &
        year_games["away_conf"].notna()
    )

    # Build connectivity table
    rows = []
    for week in range(1, max_week + 1):
        # Games played before this week
        prior_games = year_games[year_games["week"] < week]

        # Count OOC FBS games by conference (each game counts for both conferences)
        for conf in set(conf_map.values()):
            if conf in ["FCS", None]:
                continue

            # Games where this conference played OOC
            conf_ooc = prior_games[
                prior_games["is_ooc"] &
                prior_games["is_fbs"] &
                ((prior_games["home_conf"] == conf) | (prior_games["away_conf"] == conf))
            ]

            # Count teams in conference
            teams_in_conf = sum(1 for t, c in conf_map.items() if c == conf)

            rows.append({
                "year": year,
                "week": week,
                "conference": conf,
                "ooc_fbs_games": len(conf_ooc),
                "teams_in_conf": teams_in_conf,
                "ooc_games_per_team": len(conf_ooc) / teams_in_conf if teams_in_conf > 0 else 0,
            })

    return pd.DataFrame(rows)


def calculate_cohort_metrics(
    df: pd.DataFrame,
    label: str,
) -> dict:
    """Calculate metrics for a cohort of games."""
    n = len(df)
    if n == 0:
        return {"label": label, "n": 0}

    # Basic error metrics
    mae = df["abs_error"].mean()
    me = df["error"].mean()  # Positive = model over-predicted home margin
    rmse = (df["error"] ** 2).mean() ** 0.5

    # ATS metrics (need spread columns)
    ats_close = None
    ats_open = None
    ats_close_edge3 = None
    ats_close_edge5 = None
    ats_open_edge3 = None
    ats_open_edge5 = None
    clv_mean = None

    if "spread_close" in df.columns and df["spread_close"].notna().any():
        # Filter to games with closing spread
        ats_df = df[df["spread_close"].notna()].copy()

        # Calculate edge (model disagreement with Vegas)
        # Positive edge = model likes away more than Vegas
        ats_df["edge_close"] = ats_df["predicted_spread"] - ats_df["spread_close"]

        # ATS result: cover if (predicted - vegas) * sign(actual - vegas_implied) > 0
        # Simpler: compare model pick vs actual outcome relative to spread
        # model_pick = sign(predicted_spread - spread_close)
        #   positive = model thinks home is better than vegas implies -> pick home
        # actual = actual_margin - (-spread_close) = actual_margin + spread_close
        #   positive = home covered
        ats_df["home_covered"] = (ats_df["actual_margin"] + ats_df["spread_close"]) > 0
        ats_df["model_pick_home"] = ats_df["edge_close"] < 0  # Model likes home more than Vegas
        ats_df["ats_correct"] = ats_df["model_pick_home"] == ats_df["home_covered"]

        # Remove pushes
        ats_df_no_push = ats_df[(ats_df["actual_margin"] + ats_df["spread_close"]) != 0]

        if len(ats_df_no_push) > 0:
            ats_close = ats_df_no_push["ats_correct"].mean()

            # Edge buckets
            edge3 = ats_df_no_push[ats_df_no_push["edge_close"].abs() >= 3]
            edge5 = ats_df_no_push[ats_df_no_push["edge_close"].abs() >= 5]

            if len(edge3) > 0:
                ats_close_edge3 = edge3["ats_correct"].mean()
            if len(edge5) > 0:
                ats_close_edge5 = edge5["ats_correct"].mean()

    if "spread_open" in df.columns and df["spread_open"].notna().any():
        ats_df = df[df["spread_open"].notna()].copy()
        ats_df["edge_open"] = ats_df["predicted_spread"] - ats_df["spread_open"]
        ats_df["home_covered_open"] = (ats_df["actual_margin"] + ats_df["spread_open"]) > 0
        ats_df["model_pick_home_open"] = ats_df["edge_open"] < 0
        ats_df["ats_correct_open"] = ats_df["model_pick_home_open"] == ats_df["home_covered_open"]

        ats_df_no_push = ats_df[(ats_df["actual_margin"] + ats_df["spread_open"]) != 0]

        if len(ats_df_no_push) > 0:
            ats_open = ats_df_no_push["ats_correct_open"].mean()

            edge3 = ats_df_no_push[ats_df_no_push["edge_open"].abs() >= 3]
            edge5 = ats_df_no_push[ats_df_no_push["edge_open"].abs() >= 5]

            if len(edge3) > 0:
                ats_open_edge3 = edge3["ats_correct_open"].mean()
            if len(edge5) > 0:
                ats_open_edge5 = edge5["ats_correct_open"].mean()

    # CLV (Closing Line Value)
    if "spread_open" in df.columns and "spread_close" in df.columns:
        clv_df = df[(df["spread_open"].notna()) & (df["spread_close"].notna())].copy()
        if len(clv_df) > 0:
            # CLV = movement in our favor
            # If we pick home (edge_open < 0), CLV = spread_open - spread_close (line moved toward us)
            # If we pick away (edge_open > 0), CLV = spread_close - spread_open
            clv_df["edge_open"] = clv_df["predicted_spread"] - clv_df["spread_open"]
            clv_df["model_pick_home"] = clv_df["edge_open"] < 0
            clv_df["clv"] = clv_df.apply(
                lambda r: (r["spread_open"] - r["spread_close"]) if r["model_pick_home"]
                else (r["spread_close"] - r["spread_open"]),
                axis=1
            )
            clv_mean = clv_df["clv"].mean()

    return {
        "label": label,
        "n": n,
        "mae": round(mae, 2),
        "me": round(me, 2),
        "rmse": round(rmse, 2),
        "ats_close": round(ats_close * 100, 1) if ats_close else None,
        "ats_open": round(ats_open * 100, 1) if ats_open else None,
        "ats_close_3+": round(ats_close_edge3 * 100, 1) if ats_close_edge3 else None,
        "ats_close_5+": round(ats_close_edge5 * 100, 1) if ats_close_edge5 else None,
        "ats_open_3+": round(ats_open_edge3 * 100, 1) if ats_open_edge3 else None,
        "ats_open_5+": round(ats_open_edge5 * 100, 1) if ats_open_edge5 else None,
        "clv_mean": round(clv_mean, 3) if clv_mean else None,
    }


def analyze_g5_vs_p4(
    predictions_df: pd.DataFrame,
    conf_by_year: dict[int, dict[str, str]],
) -> dict:
    """Analyze G5 vs P4 matchups specifically."""
    df = add_tier_info(predictions_df, conf_by_year)

    # Filter to G5 vs P4 games
    g5_p4 = df[df["game_type"] == "G5_vs_P4"].copy()

    if len(g5_p4) == 0:
        logger.warning("No G5 vs P4 games found!")
        return {}

    # Identify which team is G5 and which is P4
    g5_p4["g5_team"] = g5_p4.apply(
        lambda r: r["home_team"] if r["home_tier"] == "G5" else r["away_team"], axis=1
    )
    g5_p4["p4_team"] = g5_p4.apply(
        lambda r: r["home_team"] if r["home_tier"] == "P4" else r["away_team"], axis=1
    )
    g5_p4["g5_is_home"] = g5_p4["home_tier"] == "G5"
    g5_p4["g5_conf"] = g5_p4.apply(
        lambda r: r["home_conf"] if r["g5_is_home"] else r["away_conf"], axis=1
    )

    # Determine favorite (using closing spread if available, else predicted)
    if "spread_close" in g5_p4.columns:
        # Vegas spread: negative = home favored
        g5_p4["g5_favored"] = g5_p4.apply(
            lambda r: (r["spread_close"] < 0) if r["g5_is_home"] else (r["spread_close"] > 0)
            if pd.notna(r["spread_close"]) else None,
            axis=1
        )
    else:
        g5_p4["g5_favored"] = None

    results = {
        "total_games": len(g5_p4),
        "cohorts": {},
    }

    # Overall G5 vs P4
    results["cohorts"]["all"] = calculate_cohort_metrics(g5_p4, "G5 vs P4 (All)")

    # By week slice
    early = g5_p4[g5_p4["week"] <= 3]
    core = g5_p4[(g5_p4["week"] >= 4) & (g5_p4["week"] <= 15)]
    results["cohorts"]["weeks_1-3"] = calculate_cohort_metrics(early, "G5 vs P4 (Weeks 1-3)")
    results["cohorts"]["weeks_4-15"] = calculate_cohort_metrics(core, "G5 vs P4 (Weeks 4-15)")

    # By home/away
    g5_home = g5_p4[g5_p4["g5_is_home"]]
    g5_away = g5_p4[~g5_p4["g5_is_home"]]
    results["cohorts"]["g5_home"] = calculate_cohort_metrics(g5_home, "G5 Home vs P4")
    results["cohorts"]["g5_away"] = calculate_cohort_metrics(g5_away, "G5 @ P4")

    # By favorite/dog (where available)
    g5_fav = g5_p4[g5_p4["g5_favored"] == True]
    g5_dog = g5_p4[g5_p4["g5_favored"] == False]
    results["cohorts"]["g5_favorite"] = calculate_cohort_metrics(g5_fav, "G5 Favored vs P4")
    results["cohorts"]["g5_underdog"] = calculate_cohort_metrics(g5_dog, "G5 Underdog vs P4")

    # By G5 conference
    for conf in G5_CONFERENCES:
        conf_games = g5_p4[g5_p4["g5_conf"] == conf]
        if len(conf_games) > 0:
            results["cohorts"][f"conf_{conf}"] = calculate_cohort_metrics(
                conf_games, f"{conf} vs P4"
            )

    # Game type breakdown for context
    all_games = df
    results["game_type_counts"] = {
        "P4_vs_P4": len(all_games[all_games["game_type"] == "P4_vs_P4"]),
        "G5_vs_G5": len(all_games[all_games["game_type"] == "G5_vs_G5"]),
        "G5_vs_P4": len(all_games[all_games["game_type"] == "G5_vs_P4"]),
        "OTHER": len(all_games[all_games["game_type"] == "OTHER"]),
    }

    return results


def identify_bias_conferences(
    predictions_df: pd.DataFrame,
    connectivity_df: pd.DataFrame,
    conf_by_year: dict[int, dict[str, str]],
    me_threshold: float = 2.0,
    ats_threshold: float = 48.0,
    connectivity_threshold: float = 2.0,  # OOC games per team at time of game
) -> list[dict]:
    """Identify conferences with persistent bias in G5 vs P4 games."""
    df = add_tier_info(predictions_df, conf_by_year)
    g5_p4 = df[df["game_type"] == "G5_vs_P4"].copy()

    if len(g5_p4) == 0:
        return []

    # Add G5 conference
    g5_p4["g5_conf"] = g5_p4.apply(
        lambda r: r["home_conf"] if r["home_tier"] == "G5" else r["away_conf"], axis=1
    )

    # Merge with connectivity data
    # Get connectivity at time of each game
    g5_p4 = g5_p4.merge(
        connectivity_df[["year", "week", "conference", "ooc_games_per_team"]],
        left_on=["year", "week", "g5_conf"],
        right_on=["year", "week", "conference"],
        how="left",
    )

    flagged = []

    for conf in G5_CONFERENCES:
        conf_games = g5_p4[g5_p4["g5_conf"] == conf]
        if len(conf_games) < 10:  # Need minimum sample
            continue

        metrics = calculate_cohort_metrics(conf_games, conf)
        avg_connectivity = conf_games["ooc_games_per_team"].mean() if "ooc_games_per_team" in conf_games else None

        # Check bias conditions
        is_biased = False
        bias_reasons = []

        if metrics.get("me") and abs(metrics["me"]) > me_threshold:
            is_biased = True
            bias_reasons.append(f"Mean Error {metrics['me']:+.2f}")

        if metrics.get("ats_close") and metrics["ats_close"] < ats_threshold:
            is_biased = True
            bias_reasons.append(f"ATS {metrics['ats_close']:.1f}%")

        has_low_connectivity = (
            avg_connectivity is not None and avg_connectivity < connectivity_threshold
        )

        flagged.append({
            "conference": conf,
            "n_games": metrics["n"],
            "mae": metrics.get("mae"),
            "me": metrics.get("me"),
            "ats_close": metrics.get("ats_close"),
            "ats_close_5+": metrics.get("ats_close_5+"),
            "avg_connectivity": round(avg_connectivity, 2) if avg_connectivity else None,
            "is_biased": is_biased,
            "has_low_connectivity": has_low_connectivity,
            "bias_reasons": bias_reasons,
        })

    return flagged


def run_diagnostics(
    years: list[int],
    output_dir: Path,
    use_cache: bool = True,
) -> dict:
    """Run full G5 island diagnostics."""
    logger.info(f"Running G5 island diagnostics for years: {years}")

    # Fetch conference mappings
    conf_by_year = fetch_conference_mappings(years)

    # Fetch season data
    logger.info("Fetching season data...")
    season_data = fetch_all_season_data(
        years=years,
        use_priors=True,
        use_portal=True,
        use_cache=use_cache,
    )

    # Run backtest to get predictions
    logger.info("Running backtest...")
    results = run_backtest(
        years=years,
        start_week=1,
        season_data=season_data,
        use_priors=True,
        # Use default production settings
    )

    predictions = results.get("predictions", [])
    if predictions is None or (isinstance(predictions, list) and len(predictions) == 0):
        logger.error("No predictions generated!")
        return {}

    # Handle both list and DataFrame formats
    if isinstance(predictions, pd.DataFrame):
        pred_df = predictions
    else:
        pred_df = pd.DataFrame(predictions)

    if pred_df.empty:
        logger.error("No predictions generated!")
        return {}
    logger.info(f"Generated {len(pred_df)} predictions")

    # Merge with betting data for ATS/CLV
    all_betting = []
    for year in years:
        if year in season_data and season_data[year].betting_df is not None:
            all_betting.append(season_data[year].betting_df.to_pandas())

    if all_betting:
        betting_df = pd.concat(all_betting, ignore_index=True)
        pred_df = pred_df.merge(
            betting_df[["game_id", "spread_open", "spread_close"]],
            on="game_id",
            how="left",
        )
        logger.info(f"Merged betting data: {pred_df['spread_close'].notna().sum()} games with closing lines")

    # Calculate connectivity for all years
    all_connectivity = []
    for year in years:
        if year in season_data:
            conn = calculate_connectivity(
                season_data[year].games_df,
                conf_by_year,
                year,
            )
            if not conn.empty:
                all_connectivity.append(conn)

    connectivity_df = pd.concat(all_connectivity, ignore_index=True) if all_connectivity else pd.DataFrame()
    logger.info(f"Calculated connectivity: {len(connectivity_df)} conference-week records")

    # Run analyses
    logger.info("Analyzing G5 vs P4 matchups...")
    g5_analysis = analyze_g5_vs_p4(pred_df, conf_by_year)

    logger.info("Identifying bias conferences...")
    bias_flags = identify_bias_conferences(pred_df, connectivity_df, conf_by_year)

    # Compile results
    diagnostics = {
        "years": years,
        "total_predictions": len(pred_df),
        "g5_vs_p4_analysis": g5_analysis,
        "conference_bias_flags": bias_flags,
    }

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save connectivity table
    if not connectivity_df.empty:
        conn_path = output_dir / "conference_connectivity.csv"
        connectivity_df.to_csv(conn_path, index=False)
        logger.info(f"Saved connectivity data to {conn_path}")

    # Save diagnostics JSON (convert numpy types to native Python)
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
            return float(obj)
        return obj

    diag_path = output_dir / "g5_island_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(convert_types(diagnostics), f, indent=2)
    logger.info(f"Saved diagnostics to {diag_path}")

    # Print summary report
    print_summary_report(diagnostics, connectivity_df, bias_flags)

    return diagnostics


def print_summary_report(
    diagnostics: dict,
    connectivity_df: pd.DataFrame,
    bias_flags: list[dict],
):
    """Print human-readable summary report."""
    print("\n" + "=" * 70)
    print("G5 CONFERENCE ISLAND DIAGNOSTICS")
    print("=" * 70)

    # Game type breakdown
    if "g5_vs_p4_analysis" in diagnostics and "game_type_counts" in diagnostics["g5_vs_p4_analysis"]:
        counts = diagnostics["g5_vs_p4_analysis"]["game_type_counts"]
        print("\n📊 GAME TYPE BREAKDOWN")
        print("-" * 40)
        for gtype, count in counts.items():
            print(f"  {gtype:12}: {count:5} games")

    # G5 vs P4 cohort analysis
    if "g5_vs_p4_analysis" in diagnostics and "cohorts" in diagnostics["g5_vs_p4_analysis"]:
        cohorts = diagnostics["g5_vs_p4_analysis"]["cohorts"]

        print("\n📈 G5 vs P4 COHORT ANALYSIS")
        print("-" * 70)
        print(f"{'Cohort':<25} {'N':>5} {'MAE':>6} {'ME':>7} {'ATS%':>6} {'5+ATS':>6} {'CLV':>7}")
        print("-" * 70)

        for key in ["all", "weeks_1-3", "weeks_4-15", "g5_home", "g5_away",
                    "g5_favorite", "g5_underdog"]:
            if key in cohorts:
                c = cohorts[key]
                ats = f"{c['ats_close']:.1f}" if c.get('ats_close') else "-"
                ats5 = f"{c['ats_close_5+']:.1f}" if c.get('ats_close_5+') else "-"
                clv = f"{c['clv_mean']:+.3f}" if c.get('clv_mean') else "-"
                me = f"{c['me']:+.2f}" if c.get('me') else "-"
                print(f"  {c['label']:<23} {c['n']:>5} {c.get('mae', '-'):>6} {me:>7} {ats:>6} {ats5:>6} {clv:>7}")

        # Conference breakdown
        print("\n  BY G5 CONFERENCE:")
        for key, c in cohorts.items():
            if key.startswith("conf_"):
                ats = f"{c['ats_close']:.1f}" if c.get('ats_close') else "-"
                ats5 = f"{c['ats_close_5+']:.1f}" if c.get('ats_close_5+') else "-"
                clv = f"{c['clv_mean']:+.3f}" if c.get('clv_mean') else "-"
                me = f"{c['me']:+.2f}" if c.get('me') else "-"
                print(f"  {c['label']:<23} {c['n']:>5} {c.get('mae', '-'):>6} {me:>7} {ats:>6} {ats5:>6} {clv:>7}")

    # Conference connectivity summary
    if not connectivity_df.empty:
        print("\n🔗 CONFERENCE CONNECTIVITY (OOC FBS games per team at Week 4)")
        print("-" * 50)

        # Show week 4 connectivity (when core predictions start)
        week4 = connectivity_df[connectivity_df["week"] == 4]
        if not week4.empty:
            conf_avg = week4.groupby("conference")["ooc_games_per_team"].mean().sort_values(ascending=False)
            for conf, avg in conf_avg.items():
                tier = "P4" if conf in P4_CONFERENCES else "G5" if conf in G5_CONFERENCES else "?"
                print(f"  {conf:<25} [{tier}]: {avg:.2f} OOC games/team")

    # Bias flags
    print("\n⚠️  BIAS FLAGS (ME > ±2.0 or ATS < 48%)")
    print("-" * 70)
    if bias_flags:
        flagged_confs = [f for f in bias_flags if f["is_biased"]]
        if flagged_confs:
            for f in flagged_confs:
                low_conn = "⚡ LOW CONN" if f["has_low_connectivity"] else ""
                print(f"  {f['conference']:<20} N={f['n_games']:>3}  "
                      f"ME={f['me']:+.2f}  ATS={f['ats_close']:.1f}%  "
                      f"Conn={f['avg_connectivity']:.1f}  {low_conn}")
                if f["bias_reasons"]:
                    print(f"     Reasons: {', '.join(f['bias_reasons'])}")
        else:
            print("  No conferences flagged for bias.")
    else:
        print("  No bias analysis available.")

    print("\n" + "=" * 70)
    print("PHASE A DIAGNOSTICS COMPLETE")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="G5 Conference Island Diagnostics")
    parser.add_argument(
        "--years",
        type=str,
        default="2022,2023,2024,2025",
        help="Comma-separated list of years to analyze",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/outputs/g5_diagnostics",
        help="Directory for output files",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable season data cache",
    )

    args = parser.parse_args()

    years = [int(y.strip()) for y in args.years.split(",")]
    output_dir = Path(args.output_dir)

    run_diagnostics(
        years=years,
        output_dir=output_dir,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()
