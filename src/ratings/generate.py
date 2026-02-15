"""Canonical rating generation — single source of truth.

Every consumer (generate_docs.py, show_ratings.py, Discord bot) must call
generate_ratings() from this module to ensure consistency.
"""

import logging
from pathlib import Path

import polars as pl

from scripts.backtest import fetch_all_season_data
from src.models.efficiency_foundation_model import EfficiencyFoundationModel
from src.models.special_teams import SpecialTeamsModel

logger = logging.getLogger(__name__)


def generate_ratings(year: int, week: int | None = None) -> list[dict]:
    """Generate JP+ power ratings for a given year.

    This is the SINGLE SOURCE OF TRUTH for rating generation.
    Uses the full production pipeline: fetch_all_season_data + EFM
    (with conference anchors, RZ leverage, asymmetric garbage time,
    turnover shrinkage) + Special Teams.

    Args:
        year: Season year
        week: Optional week cutoff. None = full season (including postseason).

    Returns:
        List of dicts with keys: team, overall, offense, defense, st,
        rank, off_rank, def_rank, st_rank. Sorted by overall descending.
        All FBS teams included.
    """
    season_data = fetch_all_season_data([year], use_priors=False, use_portal=False)
    sd = season_data[year]

    plays_pd = sd.efficiency_plays_df.to_pandas()
    games_pd = sd.games_df.to_pandas()
    st_plays_pd = sd.st_plays_df.to_pandas()

    # Apply week filter if specified
    if week is not None:
        plays_pd = sd.efficiency_plays_df.filter(pl.col("week") <= week).to_pandas()
        games_pd = sd.games_df.filter(pl.col("week") <= week).to_pandas()
        st_plays_pd = sd.st_plays_df.filter(pl.col("week") <= week).to_pandas()

    # EFM ratings — full production pipeline
    efm = EfficiencyFoundationModel(ridge_alpha=50.0)
    efm.calculate_ratings(
        plays_pd, games_pd,
        fbs_teams=sd.fbs_teams,
        team_conferences=sd.team_conferences,
    )
    efm_df = efm.get_ratings_df()
    efm_df = efm_df[efm_df["team"].isin(sd.fbs_teams)].copy()

    # ST ratings
    st_model = SpecialTeamsModel()
    games_played = games_pd.groupby("home_team").size().to_dict()
    away_games = games_pd.groupby("away_team").size().to_dict()
    for team, count in away_games.items():
        games_played[team] = games_played.get(team, 0) + count
    st_model.calculate_all_st_ratings_from_plays(st_plays_pd, games_played)

    # Combine EFM + ST
    ratings = []
    for _, row in efm_df.iterrows():
        team = row["team"]
        st_rating = st_model.get_rating(team)
        st_val = st_rating.overall_rating if st_rating else 0.0
        ratings.append({
            "team": team,
            "overall": row["overall"] + st_val,
            "offense": row["offense"],
            "defense": row["defense"],
            "st": st_val,
        })

    # Sort and rank
    ratings.sort(key=lambda x: -x["overall"])
    for i, r in enumerate(ratings):
        r["rank"] = i + 1

    off_sorted = sorted(ratings, key=lambda x: -x["offense"])
    def_sorted = sorted(ratings, key=lambda x: -x["defense"])
    st_sorted = sorted(ratings, key=lambda x: -x["st"])

    off_ranks = {r["team"]: i + 1 for i, r in enumerate(off_sorted)}
    def_ranks = {r["team"]: i + 1 for i, r in enumerate(def_sorted)}
    st_ranks = {r["team"]: i + 1 for i, r in enumerate(st_sorted)}

    for r in ratings:
        r["off_rank"] = off_ranks[r["team"]]
        r["def_rank"] = def_ranks[r["team"]]
        r["st_rank"] = st_ranks[r["team"]]

    # Add metadata
    n_regular = len(games_pd[games_pd["week"] <= 15])
    n_postseason = len(games_pd[games_pd["week"] >= 16])
    n_fbs = len(sd.fbs_teams)

    # Attach as list attribute (not per-item) for consumers that need it
    ratings_meta = {
        "n_regular": n_regular,
        "n_postseason": n_postseason,
        "n_fbs": n_fbs,
    }

    logger.info(f"Generated ratings for {len(ratings)} FBS teams")
    return ratings, ratings_meta
