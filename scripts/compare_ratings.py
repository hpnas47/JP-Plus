#!/usr/bin/env python3
"""Compare JP+ ratings to FPI and SP+ for validation and divergence analysis."""

import sys
sys.path.insert(0, "/Users/jason/Documents/CFB Power Ratings Model")

import pandas as pd
from src.api.cfbd_client import CFBDClient
from src.models.efficiency_foundation_model import EfficiencyFoundationModel


def fetch_jp_ratings(client: CFBDClient, year: int) -> pd.DataFrame:
    """Generate JP+ ratings for comparison."""
    print(f"Generating {year} JP+ ratings...")

    # Get FBS teams first
    fbs_teams = set(t.school for t in client.get_fbs_teams(year))
    print(f"  Got {len(fbs_teams)} FBS teams")

    # Fetch plays for all weeks
    all_plays = []
    for week in range(1, 17):
        try:
            plays = client.get_plays(year, week, season_type="regular")
            all_plays.extend(plays)
        except:
            pass
    # Also postseason
    for week in range(1, 6):
        try:
            plays = client.get_plays(year, week, season_type="postseason")
            all_plays.extend(plays)
        except:
            pass

    print(f"  Fetched {len(all_plays)} total plays")

    plays_df = pd.DataFrame([{
        "game_id": p.game_id,
        "offense": p.offense,
        "defense": p.defense,
        "ppa": p.ppa,
        "yards_gained": p.yards_gained,
        "down": p.down,
        "distance": p.distance,
        "play_type": p.play_type,
        "period": p.period,
        "clock_minutes": p.clock.minutes if p.clock else None,
        "offense_score": p.offense_score,
        "defense_score": p.defense_score,
    } for p in all_plays if p.ppa is not None])

    # Filter to FBS vs FBS
    plays_df = plays_df[
        plays_df["offense"].isin(fbs_teams) &
        plays_df["defense"].isin(fbs_teams)
    ]
    print(f"  Filtered to {len(plays_df)} FBS-only plays")

    # Run EFM
    efm = EfficiencyFoundationModel(
        ridge_alpha=50.0,
        efficiency_weight=0.45,
        explosiveness_weight=0.45,
        turnover_weight=0.10,
        asymmetric_garbage=True,
    )
    efm.calculate_ratings(plays_df)

    # Get ratings
    ratings_df = efm.get_ratings_df()

    # Normalize to similar scale as SP+/FPI (mean ~0, std ~12)
    mean_rating = ratings_df["overall"].mean()
    std_rating = ratings_df["overall"].std()
    ratings_df["jp_overall"] = (ratings_df["overall"] - mean_rating) / std_rating * 12
    ratings_df["jp_off"] = ratings_df["offense"]
    ratings_df["jp_def"] = ratings_df["defense"]

    print(f"  Generated {len(ratings_df)} JP+ ratings")

    return ratings_df[["team", "jp_overall", "jp_off", "jp_def"]]


def fetch_all_ratings(client: CFBDClient, year: int) -> pd.DataFrame:
    """Fetch JP+, FPI, and SP+ ratings for comparison."""

    # Generate JP+ ratings
    jp_df = fetch_jp_ratings(client, year)

    # Fetch FPI
    print(f"Fetching {year} FPI ratings...")
    fpi_data = client.get_fpi_ratings(year)
    fpi_df = pd.DataFrame([{
        "team": r.team,
        "fpi_overall": r.fpi,
        "fpi_off": r.efficiencies.offense if r.efficiencies else None,
        "fpi_def": r.efficiencies.defense if r.efficiencies else None,
        "fpi_st": r.efficiencies.special_teams if r.efficiencies else None,
    } for r in fpi_data])
    print(f"  Got {len(fpi_df)} FPI ratings")

    # Fetch SP+
    print(f"Fetching {year} SP+ ratings...")
    sp_data = client.get_sp_ratings(year)
    sp_df = pd.DataFrame([{
        "team": r.team,
        "sp_overall": r.rating,
        "sp_off": r.offense.rating if r.offense else None,
        "sp_def": r.defense.rating if r.defense else None,
    } for r in sp_data])
    print(f"  Got {len(sp_df)} SP+ ratings")

    # Merge all three
    merged = jp_df.merge(fpi_df, on="team", how="outer")
    merged = merged.merge(sp_df, on="team", how="outer")

    return merged


def add_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Add rank columns for each rating system."""
    df = df.copy()

    # JP+ rank (higher = better)
    df["jp_rank"] = df["jp_overall"].rank(ascending=False, method="min")

    # FPI rank (higher = better)
    df["fpi_rank"] = df["fpi_overall"].rank(ascending=False, method="min")

    # SP+ rank (higher = better)
    df["sp_rank"] = df["sp_overall"].rank(ascending=False, method="min")

    return df


def find_divergences(df: pd.DataFrame, min_diff: int = 10) -> pd.DataFrame:
    """Find teams where FPI and SP+ rankings diverge significantly."""
    df = df.copy()
    df["rank_diff"] = (df["fpi_rank"] - df["sp_rank"]).abs()
    divergent = df[df["rank_diff"] >= min_diff].sort_values("rank_diff", ascending=False)
    return divergent


def main():
    client = CFBDClient()
    year = 2025

    print(f"\n{'='*60}")
    print(f"RATINGS COMPARISON: {year}")
    print(f"{'='*60}\n")

    # Fetch ratings
    ratings = fetch_all_ratings(client, year)
    ratings = add_rankings(ratings)

    # Top 25 comparison
    print(f"\n{'='*70}")
    print("TOP 25 COMPARISON (JP+ vs FPI vs SP+)")
    print(f"{'='*70}")

    # Get top 25 by each system
    top_jp = set(ratings.nsmallest(25, "jp_rank")["team"])
    top_fpi = set(ratings.nsmallest(25, "fpi_rank")["team"])
    top_sp = set(ratings.nsmallest(25, "sp_rank")["team"])

    # Find consensus (in all 3)
    consensus = top_jp & top_fpi & top_sp
    print(f"\nConsensus Top 25 (all 3 systems): {len(consensus)} teams")

    print(f"\nIn JP+ Top 25 but NOT FPI or SP+ Top 25:")
    jp_only = top_jp - top_fpi - top_sp
    for team in sorted(jp_only):
        row = ratings[ratings["team"] == team].iloc[0]
        print(f"  {team}: JP+ #{int(row['jp_rank'])} ({row['jp_overall']:.1f}), FPI #{int(row['fpi_rank'])}, SP+ #{int(row['sp_rank'])}")

    print(f"\nIn FPI/SP+ Top 25 but NOT JP+ Top 25:")
    others_not_jp = (top_fpi | top_sp) - top_jp
    for team in sorted(others_not_jp):
        row = ratings[ratings["team"] == team].iloc[0]
        jp_r = int(row['jp_rank']) if pd.notna(row['jp_rank']) else 'N/A'
        fpi_r = int(row['fpi_rank']) if pd.notna(row['fpi_rank']) else 'N/A'
        sp_r = int(row['sp_rank']) if pd.notna(row['sp_rank']) else 'N/A'
        in_fpi = "FPI" if team in top_fpi else ""
        in_sp = "SP+" if team in top_sp else ""
        print(f"  {team}: JP+ #{jp_r}, FPI #{fpi_r}, SP+ #{sp_r} (in: {in_fpi} {in_sp})")

    # Biggest divergences
    print(f"\n{'='*60}")
    print("BIGGEST RANK DIVERGENCES (FPI vs SP+, diff >= 15)")
    print(f"{'='*60}")

    divergent = find_divergences(ratings, min_diff=15)
    for _, row in divergent.head(15).iterrows():
        direction = "FPI higher" if row["fpi_rank"] < row["sp_rank"] else "SP+ higher"
        print(f"  {row['team']}: FPI #{int(row['fpi_rank'])} vs SP+ #{int(row['sp_rank'])} ({direction}, diff={int(row['rank_diff'])})")

    # Top 10 side by side
    print(f"\n{'='*90}")
    print("TOP 10 SIDE BY SIDE")
    print(f"{'='*90}")
    print(f"{'Rank':<5} {'JP+':<18} {'Rtg':>6}  {'FPI':<18} {'Rtg':>6}  {'SP+':<18} {'Rtg':>6}")
    print("-" * 90)

    jp_top10 = ratings.nsmallest(10, "jp_rank").sort_values("jp_rank")
    fpi_top10 = ratings.nsmallest(10, "fpi_rank").sort_values("fpi_rank")
    sp_top10 = ratings.nsmallest(10, "sp_rank").sort_values("sp_rank")

    for i in range(10):
        jp_row = jp_top10.iloc[i]
        fpi_row = fpi_top10.iloc[i]
        sp_row = sp_top10.iloc[i]
        print(f"{i+1:<5} {jp_row['team']:<18} {jp_row['jp_overall']:>6.1f}  {fpi_row['team']:<18} {fpi_row['fpi_overall']:>6.1f}  {sp_row['team']:<18} {sp_row['sp_overall']:>6.1f}")

    # Correlation
    print(f"\n{'='*70}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*70}")

    valid = ratings.dropna(subset=["jp_overall", "fpi_overall", "sp_overall"])

    jp_fpi_corr = valid["jp_overall"].corr(valid["fpi_overall"])
    jp_sp_corr = valid["jp_overall"].corr(valid["sp_overall"])
    fpi_sp_corr = valid["fpi_overall"].corr(valid["sp_overall"])

    jp_fpi_rank = valid["jp_rank"].corr(valid["fpi_rank"])
    jp_sp_rank = valid["jp_rank"].corr(valid["sp_rank"])
    fpi_sp_rank = valid["fpi_rank"].corr(valid["sp_rank"])

    print(f"\n  Rating Correlations:")
    print(f"    JP+ vs FPI:  r = {jp_fpi_corr:.3f}")
    print(f"    JP+ vs SP+:  r = {jp_sp_corr:.3f}")
    print(f"    FPI vs SP+:  r = {fpi_sp_corr:.3f}")

    print(f"\n  Rank Correlations:")
    print(f"    JP+ vs FPI:  r = {jp_fpi_rank:.3f}")
    print(f"    JP+ vs SP+:  r = {jp_sp_rank:.3f}")
    print(f"    FPI vs SP+:  r = {fpi_sp_rank:.3f}")

    # Save full comparison
    output_file = f"/Users/jason/Documents/CFB Power Ratings Model/ratings_comparison_{year}.csv"
    ratings.to_csv(output_file, index=False)
    print(f"\n  Full comparison saved to: {output_file}")


if __name__ == "__main__":
    main()
