"""Strength of Schedule adjustment for power ratings."""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def calculate_sos(
    games_df: pd.DataFrame,
    team_ratings: dict[str, float],
) -> dict[str, float]:
    """Calculate strength of schedule for each team.

    SOS = average rating of opponents faced.

    Args:
        games_df: DataFrame with home_team, away_team columns
        team_ratings: Dict mapping team name to rating

    Returns:
        Dict mapping team name to SOS (average opponent rating)
    """
    team_opponents: dict[str, list[float]] = {}

    for _, game in games_df.iterrows():
        home = game["home_team"]
        away = game["away_team"]

        # Home team faced away team
        if home not in team_opponents:
            team_opponents[home] = []
        team_opponents[home].append(team_ratings.get(away, 0.0))

        # Away team faced home team
        if away not in team_opponents:
            team_opponents[away] = []
        team_opponents[away].append(team_ratings.get(home, 0.0))

    # Calculate average opponent rating for each team
    sos = {}
    for team, opp_ratings in team_opponents.items():
        if opp_ratings:
            sos[team] = sum(opp_ratings) / len(opp_ratings)
        else:
            sos[team] = 0.0

    return sos


def apply_sos_adjustment(
    team_ratings: dict[str, float],
    games_df: pd.DataFrame,
    adjustment_factor: float = 0.4,
    iterations: int = 3,
) -> dict[str, float]:
    """Apply strength of schedule adjustment to ratings.

    Teams with harder schedules (higher avg opponent rating) get a boost.
    Teams with easier schedules get a penalty.

    Uses iterative refinement: after adjusting ratings, recalculate SOS
    with new ratings and adjust again. This helps ratings converge.

    Args:
        team_ratings: Dict mapping team name to initial rating
        games_df: DataFrame with game data
        adjustment_factor: How much to weight SOS difference (0-1).
                          0.4 means 40% of SOS difference is added to rating.
        iterations: Number of refinement iterations

    Returns:
        Dict mapping team name to SOS-adjusted rating
    """
    current_ratings = team_ratings.copy()

    for i in range(iterations):
        # Calculate SOS based on current ratings
        sos = calculate_sos(games_df, current_ratings)

        # Calculate league average SOS
        if sos:
            league_avg_sos = sum(sos.values()) / len(sos)
        else:
            league_avg_sos = 0.0

        # Apply adjustment
        adjusted = {}
        for team, rating in current_ratings.items():
            team_sos = sos.get(team, league_avg_sos)
            sos_diff = team_sos - league_avg_sos
            adjustment = adjustment_factor * sos_diff
            adjusted[team] = rating + adjustment

        current_ratings = adjusted

        logger.debug(f"SOS iteration {i+1}: avg adjustment magnitude = "
                    f"{sum(abs(v) for v in sos.values()) / len(sos):.2f}")

    # Log some examples
    if sos:
        sorted_sos = sorted(sos.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"Hardest schedules: {sorted_sos[:3]}")
        logger.info(f"Easiest schedules: {sorted_sos[-3:]}")

    return current_ratings


def get_sos_summary(
    games_df: pd.DataFrame,
    team_ratings: dict[str, float],
) -> pd.DataFrame:
    """Get a summary DataFrame of SOS for all teams.

    Args:
        games_df: DataFrame with game data
        team_ratings: Dict mapping team name to rating

    Returns:
        DataFrame with team, sos, and games_played columns
    """
    sos = calculate_sos(games_df, team_ratings)

    # Count games per team
    games_played = {}
    for _, game in games_df.iterrows():
        home = game["home_team"]
        away = game["away_team"]
        games_played[home] = games_played.get(home, 0) + 1
        games_played[away] = games_played.get(away, 0) + 1

    data = [
        {
            "team": team,
            "sos": sos_val,
            "games_played": games_played.get(team, 0),
        }
        for team, sos_val in sos.items()
    ]

    df = pd.DataFrame(data)
    return df.sort_values("sos", ascending=False).reset_index(drop=True)
