"""Special teams model for field goals, punts, and kickoffs."""

import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SpecialTeamsRating:
    """Container for team special teams ratings.

    All values are PBTA (Points Better Than Average) - the marginal point
    contribution per game compared to a league-average unit.

    Positive = gains points vs average, Negative = costs points vs average.
    """

    team: str
    field_goal_rating: float  # PAAE (Points Added Above Expected) per game
    punt_rating: float  # Punting value in points per game
    kickoff_rating: float  # Kickoff coverage + return value in points per game
    overall_rating: float  # Total ST marginal contribution (sum of components)
    is_complete: bool = False  # P2.3: True only when all 3 components are populated


class SpecialTeamsModel:
    """
    Model for evaluating special teams performance.

    All ratings are expressed as PBTA (Points Better Than Average) - the marginal
    point contribution compared to a league-average unit.

    Components:
    - Field Goals: Points Added Above Expected (PAAE) based on make rates by distance
    - Punts: Field position value vs expected, converted to points
    - Kickoffs: Coverage/return efficiency, converted to points

    Sign Convention:
    - Positive (+): Better than average, GAINS points for the team
    - Negative (-): Worse than average, COSTS the team points
    """

    # Expected FG make rates by distance (yards)
    EXPECTED_FG_RATES = {
        (0, 30): 0.92,
        (30, 40): 0.83,
        (40, 50): 0.72,
        (50, 60): 0.55,
        (60, 100): 0.30,
    }

    # Expected punt net yards
    EXPECTED_PUNT_NET = 40.0

    # Expected kickoff touchback rate
    EXPECTED_TOUCHBACK_RATE = 0.60

    # Yards-to-points conversion factor (from expected points models)
    # ~0.04 points per yard of field position change
    YARDS_TO_POINTS = 0.04

    def __init__(self) -> None:
        """Initialize the special teams model."""
        self.team_ratings: dict[str, SpecialTeamsRating] = {}

    def calculate_from_game_stats(
        self,
        team: str,
        games_df: pd.DataFrame,
    ) -> SpecialTeamsRating:
        """Calculate special teams rating from aggregated game statistics.

        P0.3: This is a FALLBACK pathway for when play-by-play data is unavailable
        (e.g., run_weekly.py). It produces approximate PBTA pts/game but lacks
        kickoff data entirely. Prefer calculate_all_st_ratings_from_plays() when possible.

        All outputs are in PBTA points/game (FG + punt). Kickoff defaults to 0.

        Args:
            team: Team name
            games_df: DataFrame with game-level stats

        Returns:
            SpecialTeamsRating for the team
        """
        logger.debug(f"ST fallback: using game-stats path for {team} (no play-by-play)")
        # Filter to team's games
        home_games = games_df[games_df["home_team"] == team]
        away_games = games_df[games_df["away_team"] == team]

        # Aggregate stats if available (VECTORIZED)
        # P3.3: Replaced iterrows with column-level .sum() for ~10x speedup
        fg_made = 0
        fg_attempts = 0
        punt_yards = 0
        punt_count = 0

        if "home_fg_made" in games_df.columns:
            fg_made += home_games["home_fg_made"].fillna(0).sum()
            fg_attempts += home_games["home_fg_attempts"].fillna(0).sum()

        if "away_fg_made" in games_df.columns:
            fg_made += away_games["away_fg_made"].fillna(0).sum()
            fg_attempts += away_games["away_fg_attempts"].fillna(0).sum()

        if "home_punt_yards" in games_df.columns:
            punt_yards += home_games["home_punt_yards"].fillna(0).sum()
            punt_count += home_games["home_punts"].fillna(0).sum()

        if "away_punt_yards" in games_df.columns:
            punt_yards += away_games["away_punt_yards"].fillna(0).sum()
            punt_count += away_games["away_punts"].fillna(0).sum()

        # P0.1: Calculate simplified ratings - ALL in PBTA points per game
        fg_rating = 0.0
        if fg_attempts > 0:
            actual_rate = fg_made / fg_attempts
            # Compare to expected rate for average distance (~35 yards)
            expected_rate = 0.78
            # Total points above expected across all attempts
            total_paae = (actual_rate - expected_rate) * fg_attempts * 3.0
            # Convert to per-game rating (typical team has ~2.5 FG attempts per game)
            n_games = len(home_games) + len(away_games)
            fg_rating = total_paae / max(1, n_games)

        punt_rating = 0.0
        if punt_count > 0:
            avg_gross = punt_yards / punt_count
            expected_gross = 42.0
            # P0.1: Convert yards difference to POINTS (was missing!)
            yards_diff = avg_gross - expected_gross
            punt_rating_per_punt = yards_diff * self.YARDS_TO_POINTS
            # Convert to per-game rating (typical team has ~5 punts per game)
            n_games = len(home_games) + len(away_games)
            punt_rating = punt_rating_per_punt * (punt_count / max(1, n_games))

        # Kickoff rating defaulted without detailed data
        kick_rating = 0.0

        # P0.1: Overall is SUM of per-game components (all in PBTA points/game)
        overall = fg_rating + punt_rating + kick_rating

        rating = SpecialTeamsRating(
            team=team,
            field_goal_rating=fg_rating,
            punt_rating=punt_rating,
            kickoff_rating=kick_rating,
            overall_rating=overall,
        )

        self.team_ratings[team] = rating
        return rating

    def get_rating(self, team: str) -> Optional[SpecialTeamsRating]:
        """Get special teams rating for a team.

        Args:
            team: Team name

        Returns:
            SpecialTeamsRating or None if not calculated
        """
        return self.team_ratings.get(team)

    def get_matchup_differential(
        self, team_a: str, team_b: str
    ) -> float:
        """Get special teams point differential in a matchup.

        Args:
            team_a: First team
            team_b: Second team

        Returns:
            Expected point advantage for team_a from special teams
        """
        rating_a = self.team_ratings.get(team_a)
        rating_b = self.team_ratings.get(team_b)

        # P2.3: Warn if using incomplete (FG-only) ratings in a matchup differential
        if rating_a and not rating_a.is_complete:
            logger.debug(f"ST rating for {team_a} is FG-only (incomplete)")
        if rating_b and not rating_b.is_complete:
            logger.debug(f"ST rating for {team_b} is FG-only (incomplete)")

        overall_a = rating_a.overall_rating if rating_a else 0.0
        overall_b = rating_b.overall_rating if rating_b else 0.0

        return overall_a - overall_b

    def get_summary_df(self) -> pd.DataFrame:
        """Get summary of all team ratings as a DataFrame.

        Returns:
            DataFrame with special teams ratings
        """
        if not self.team_ratings:
            return pd.DataFrame()

        data = [
            {
                "team": r.team,
                "fg_rating": r.field_goal_rating,
                "punt_rating": r.punt_rating,
                "kickoff_rating": r.kickoff_rating,
                "overall": r.overall_rating,
            }
            for r in self.team_ratings.values()
        ]

        df = pd.DataFrame(data)
        return df.sort_values("overall", ascending=False).reset_index(drop=True)

    def calculate_fg_ratings_from_plays(
        self,
        plays_df: pd.DataFrame,
        games_played: Optional[dict[str, int]] = None,
    ) -> None:
        """Calculate FG ratings for all teams from play-by-play data.

        Parses field goal plays from the play-by-play data and calculates
        Points Added Above Expected (PAAE) for each team.

        Args:
            plays_df: DataFrame with play-by-play data (must have play_type, play_text, offense columns)
            games_played: Optional dict of team -> games played for per-game normalization
        """
        if plays_df.empty:
            logger.warning("Empty plays dataframe, skipping FG rating calculation")
            return

        # Filter to field goal plays
        # P3.5: Select only needed columns and copy once (drop unused columns early)
        needed_cols = ["offense", "play_type", "play_text"]
        fg_mask = plays_df["play_type"].str.contains("Field Goal", case=False, na=False)
        fg_plays = plays_df.loc[fg_mask, needed_cols].copy()

        if fg_plays.empty:
            logger.warning("No field goal plays found")
            return

        # Parse distance from play_text
        # PERFORMANCE: Vectorized regex extraction (replaces .apply() for 10-100x speedup)
        # Extract yardage from text like "42 Yd Field Goal" or "35 yard FG Good"
        extracted = fg_plays["play_text"].str.extract(r'(\d+)\s*(?:Yd|yard)', flags=re.IGNORECASE, expand=False)
        fg_plays["distance"] = pd.to_numeric(extracted, errors='coerce').astype('Int64')

        # P2.1: Parse coverage diagnostics for FG distance
        n_total_fg = len(fg_plays)
        n_parsed_fg = fg_plays["distance"].notna().sum()
        fg_parse_pct = n_parsed_fg / n_total_fg * 100 if n_total_fg > 0 else 0
        if fg_parse_pct < 80:
            logger.warning(f"FG distance parse coverage low: {fg_parse_pct:.0f}% ({n_parsed_fg}/{n_total_fg})")
        else:
            logger.debug(f"FG distance parse coverage: {fg_parse_pct:.0f}% ({n_parsed_fg}/{n_total_fg})")

        # Determine if made
        fg_plays["made"] = fg_plays["play_type"].str.contains("Good", case=False, na=False)

        # Filter out plays without distance (can't evaluate)
        # P3.5: No second .copy() - already working with a copy, just filter in place
        fg_plays = fg_plays[fg_plays["distance"].notna()]

        if fg_plays.empty:
            logger.warning("No field goal plays with parseable distance")
            return

        # Calculate PAAE for each attempt (VECTORIZED)
        # P3.3: Replaced apply(axis=1) with np.select for ~10x speedup
        distance = fg_plays["distance"].values
        made = fg_plays["made"].values

        # Vectorized expected rate lookup using np.select
        # Dynamically build conditions/choices from EXPECTED_FG_RATES
        sorted_ranges = sorted(self.EXPECTED_FG_RATES.keys(), key=lambda x: x[0])
        conditions = []
        choices = []
        for (low, high), rate in [(rng, self.EXPECTED_FG_RATES[rng]) for rng in sorted_ranges]:
            if low == 0:
                conditions.append(distance < high)
            else:
                conditions.append((distance >= low) & (distance < high))
            choices.append(rate)

        # Default is the longest range rate (60-100 yards)
        default_rate = self.EXPECTED_FG_RATES[(60, 100)]
        expected_rate = np.select(conditions[:-1], choices[:-1], default=default_rate)

        # Vectorized PAAE: actual_points - expected_points
        actual_points = np.where(made, 3.0, 0.0)
        expected_points = 3.0 * expected_rate
        fg_plays["paae"] = actual_points - expected_points

        # Aggregate by team
        team_fg = fg_plays.groupby("offense").agg(
            total_paae=("paae", "sum"),
            attempts=("paae", "count"),
            makes=("made", "sum"),
        )

        # PERFORMANCE: Vectorized per-game calculation (replaces iterrows for 10-100x speedup)
        # Estimate games from attempts (avg ~2.5 FG attempts per game)
        team_fg["estimated_games"] = np.maximum(1, team_fg["attempts"] / 2.5)

        # Override with actual games_played if available
        if games_played:
            team_fg["estimated_games"] = team_fg.index.map(
                lambda t: games_played.get(t, team_fg.loc[t, "estimated_games"])
            )

        # Vectorized per-game rating calculation
        team_fg["per_game_rating"] = team_fg["total_paae"] / team_fg["estimated_games"]

        # Build team_ratings dict (dictionary comprehension ~10x faster than iterrows)
        # P2.3: is_complete=False — callers should use calculate_all_st_ratings_from_plays()
        for team in team_fg.index:
            per_game_rating = team_fg.loc[team, "per_game_rating"]
            self.team_ratings[team] = SpecialTeamsRating(
                team=team,
                field_goal_rating=per_game_rating,
                punt_rating=0.0,
                kickoff_rating=0.0,
                overall_rating=per_game_rating,  # FG-only
                is_complete=False,
            )

        # P3.9: Debug level for per-week logging
        logger.debug(
            f"Calculated FG ratings for {len(self.team_ratings)} teams "
            f"from {len(fg_plays)} FG attempts"
        )

    def get_fg_differential(self, home_team: str, away_team: str) -> float:
        """Get field goal efficiency differential for a matchup.

        Args:
            home_team: Home team name
            away_team: Away team name

        Returns:
            Expected point advantage for home team from FG efficiency
        """
        home_rating = self.team_ratings.get(home_team)
        away_rating = self.team_ratings.get(away_team)

        home_fg = home_rating.field_goal_rating if home_rating else 0.0
        away_fg = away_rating.field_goal_rating if away_rating else 0.0

        return home_fg - away_fg

    def calculate_punt_ratings_from_plays(
        self,
        plays_df: pd.DataFrame,
        games_played: Optional[dict[str, int]] = None,
    ) -> dict[str, float]:
        """Calculate punt ratings for all teams from play-by-play data.

        Parses punt plays and calculates marginal point contribution (PBTA).
        Punting team is identified from the 'offense' column (team punting).

        Components (all converted to POINTS):
        - Net yards vs expected (40 yards): ~0.04 pts/yard
        - Inside-20 bonus: +0.5 pts (better field position for defense)
        - Touchback penalty: -0.3 pts (opponent starts at 25 instead of worse)

        Args:
            plays_df: DataFrame with play-by-play data
            games_played: Optional dict of team -> games played

        Returns:
            Dict mapping team to per-game punt rating (POINTS better than average)
        """
        if plays_df.empty:
            logger.debug("Empty plays dataframe, skipping punt rating calculation")
            return {}

        # Filter to punt plays
        # P3.5: Select only needed columns and copy once (drop unused columns early)
        needed_cols = ["offense", "play_type", "play_text"]
        punt_mask = plays_df["play_type"].str.contains("Punt", case=False, na=False)
        punt_plays = plays_df.loc[punt_mask, needed_cols].copy()

        if punt_plays.empty:
            logger.debug("No punt plays found")
            return {}

        # PERFORMANCE: Vectorized text parsing (replaces .apply() for 10-100x speedup)
        # Parse punt distance from play_text
        # Examples: "John Smith punt for 45 yards", "punt for 52 Yds"
        gross_extracted = punt_plays["play_text"].str.extract(
            r'punt\s+(?:for\s+)?(\d+)\s*(?:Yd|yard)', flags=re.IGNORECASE, expand=False
        )
        punt_plays["gross_yards"] = pd.to_numeric(gross_extracted, errors='coerce').astype('Int64')

        # Detect touchbacks (vectorized string contains)
        punt_plays["is_touchback"] = punt_plays["play_text"].str.lower().str.contains(
            'touchback', na=False, regex=False
        )

        # Detect inside-20 indicators (vectorized multi-condition check)
        text_lower = punt_plays["play_text"].str.lower().fillna('')
        punt_plays["is_inside_20"] = (
            (text_lower.str.contains('inside', regex=False) & text_lower.str.contains('20', regex=False)) |
            text_lower.str.contains('downed', regex=False) |
            (text_lower.str.contains('fair catch', regex=False) & text_lower.str.contains('to the', regex=False))
        )

        # Extract return yards from "returned by X for 12 yards"
        return_extracted = punt_plays["play_text"].str.extract(
            r'return(?:ed)?.*?(\d+)\s*(?:Yd|yard)', flags=re.IGNORECASE, expand=False
        )
        punt_plays["return_yards"] = pd.to_numeric(return_extracted, errors='coerce').fillna(0).astype('int64')

        # P2.1: Parse coverage diagnostics for punt gross yards
        n_total_punts = len(punt_plays)
        n_parsed_punts = punt_plays["gross_yards"].notna().sum()
        punt_parse_pct = n_parsed_punts / n_total_punts * 100 if n_total_punts > 0 else 0
        if punt_parse_pct < 80:
            logger.warning(f"Punt gross yards parse coverage low: {punt_parse_pct:.0f}% ({n_parsed_punts}/{n_total_punts})")
        else:
            logger.debug(f"Punt gross yards parse coverage: {punt_parse_pct:.0f}% ({n_parsed_punts}/{n_total_punts})")

        # Filter out plays without parseable gross yards
        # P3.5: No second .copy() - already working with a copy, just filter in place
        punt_plays = punt_plays[punt_plays["gross_yards"].notna()]

        if punt_plays.empty:
            logger.debug("No punt plays with parseable distance")
            return {}

        # Calculate net yards
        # Calculate net yards (VECTORIZED)
        # P3.3: Replaced apply(axis=1) with np.where for ~10x speedup
        # Touchback: ball placed at 25, so net is limited
        # Normal: net = gross - return_yards
        gross = punt_plays["gross_yards"].values
        return_yards = punt_plays["return_yards"].values
        is_touchback = punt_plays["is_touchback"].values

        # Touchback: opponent gets ball at their 25-yard line, so net field
        # position gain = gross_yards - 25 (the 25 yards are "given back").
        # E.g., 50-yard touchback punt = 25 net yards, not 50.
        # Normal: net = gross - return_yards
        punt_plays["net_yards"] = np.where(
            is_touchback,
            gross - 25,
            gross - return_yards
        )

        # Calculate punt value (VECTORIZED)
        # P3.3: Replaced apply(axis=1) with vectorized arithmetic
        net_yards = punt_plays["net_yards"].values
        is_inside_20 = punt_plays["is_inside_20"].values

        # Components: yards value + inside-20 bonus + touchback penalty
        yards_value = (net_yards - self.EXPECTED_PUNT_NET) * self.YARDS_TO_POINTS
        inside_20_bonus = np.where(is_inside_20, 0.5, 0.0)
        touchback_penalty = np.where(is_touchback, -0.3, 0.0)

        punt_plays["punt_value"] = yards_value + inside_20_bonus + touchback_penalty

        # Aggregate by punting team (offense column = team that punted)
        team_punts = punt_plays.groupby("offense").agg(
            total_value=("punt_value", "sum"),
            punt_count=("punt_value", "count"),
            avg_gross=("gross_yards", "mean"),
            avg_net=("net_yards", "mean"),
        )

        # PERFORMANCE: Vectorized per-game calculation (replaces iterrows for 10-100x speedup)
        # Estimate games from punt count (avg ~5 punts per game)
        team_punts["estimated_games"] = np.maximum(1, team_punts["punt_count"] / 5.0)

        # Override with actual games_played if available
        if games_played:
            team_punts["estimated_games"] = team_punts.index.map(
                lambda t: games_played.get(t, team_punts.loc[t, "estimated_games"])
            )

        # Vectorized per-game rating calculation
        team_punts["per_game_rating"] = team_punts["total_value"] / team_punts["estimated_games"]

        # Convert to dict (dictionary comprehension ~10x faster than iterrows)
        punt_ratings = team_punts["per_game_rating"].to_dict()

        # P3.9: Debug level for per-week logging
        logger.debug(
            f"Calculated punt ratings for {len(punt_ratings)} teams "
            f"from {len(punt_plays)} punts"
        )

        return punt_ratings

    def calculate_kickoff_ratings_from_plays(
        self,
        plays_df: pd.DataFrame,
        games_played: Optional[dict[str, int]] = None,
    ) -> dict[str, float]:
        """Calculate kickoff ratings for all teams from play-by-play data.

        Two components (all converted to POINTS):
        1. Coverage rating: How well team covers kickoffs
           - Touchback rate vs expected (60%): ~1 pt per 10% above expected
           - Return yards allowed vs expected (23 yds): ~0.04 pts/yard saved
        2. Return rating: How well team returns kickoffs
           - Return yards vs expected (23 yds): ~0.04 pts/yard gained

        Args:
            plays_df: DataFrame with play-by-play data
            games_played: Optional dict of team -> games played

        Returns:
            Dict mapping team to per-game kickoff rating (POINTS better than average)
        """
        if plays_df.empty:
            logger.debug("Empty plays dataframe, skipping kickoff rating calculation")
            return {}

        # Filter to kickoff plays (exclude touchdowns which are in play_type)
        # P3.5: Select only needed columns and copy once (drop unused columns early)
        needed_cols = ["offense", "defense", "play_type", "play_text"]
        play_type_lower = plays_df["play_type"].str.lower()
        kickoff_mask = (
            play_type_lower.str.contains("kickoff", na=False) &
            ~play_type_lower.str.contains("touchdown", na=False)
        )
        kickoff_plays = plays_df.loc[kickoff_mask, needed_cols].copy()

        if kickoff_plays.empty:
            logger.debug("No kickoff plays found")
            return {}

        # P1.3: Vectorized touchback detection (replaces row-wise apply)
        kickoff_plays["is_touchback"] = kickoff_plays["play_text"].str.contains(
            "touchback", case=False, na=False
        )

        # P1.3: Vectorized return yards extraction
        return_match = kickoff_plays["play_text"].str.extract(
            r'return(?:ed)?.*?(\d+)\s*(?:Yd|yard)', flags=re.IGNORECASE
        )
        kickoff_plays["return_yards"] = pd.to_numeric(return_match[0], errors="coerce")

        # P2.1: Parse coverage diagnostics for kickoff return yards
        non_tb = kickoff_plays[~kickoff_plays["is_touchback"]]
        if len(non_tb) > 0:
            n_parsed_ko = non_tb["return_yards"].notna().sum()
            ko_parse_pct = n_parsed_ko / len(non_tb) * 100
            if ko_parse_pct < 80:
                logger.warning(f"Kickoff return yards parse coverage low: {ko_parse_pct:.0f}% ({n_parsed_ko}/{len(non_tb)} non-TB)")
            else:
                logger.debug(f"Kickoff return yards parse coverage: {ko_parse_pct:.0f}% ({n_parsed_ko}/{len(non_tb)} non-TB)")

        # For non-touchbacks without parsed return yards, use average
        kickoff_plays.loc[
            ~kickoff_plays["is_touchback"] & kickoff_plays["return_yards"].isna(),
            "return_yards"
        ] = 23  # Default to expected

        # Kicking team = offense (the team kicking off)
        # Returning team = defense (the team receiving)

        # Coverage rating (for kicking teams)
        coverage_ratings = {}
        kicker_groups = kickoff_plays.groupby("offense")
        for team, group in kicker_groups:
            touchbacks = group["is_touchback"].sum()
            total_kicks = len(group)
            tb_rate = touchbacks / total_kicks if total_kicks > 0 else 0.6

            # Returns allowed (non-touchback kicks)
            returns = group[~group["is_touchback"]]
            if len(returns) > 0:
                avg_return_allowed = returns["return_yards"].mean()
            else:
                avg_return_allowed = 23  # Expected

            # Coverage value in POINTS:
            # Touchback bonus: opponent starts at 25 vs avg return to ~27, ~2 yards = ~0.08 pts
            # Per 10% touchback rate above expected = ~0.1 pts per game
            tb_bonus = (tb_rate - self.EXPECTED_TOUCHBACK_RATE) * 1.0
            # Return yards saved: convert to points (~0.04 pts/yard)
            return_saved = (23.0 - avg_return_allowed) * self.YARDS_TO_POINTS

            # Coverage rating (in points per kick, not per game)
            # tb_bonus and return_saved are per-kick averages
            # When games_played is provided, scale to per-game; otherwise keep per-kick
            if games_played and team in games_played:
                kicks_per_game = total_kicks / games_played[team]
                coverage_ratings[team] = (tb_bonus + return_saved) * kicks_per_game
            else:
                coverage_ratings[team] = tb_bonus + return_saved

        # Return rating (for returning teams) in POINTS
        return_ratings = {}
        returner_groups = kickoff_plays[~kickoff_plays["is_touchback"]].groupby("defense")
        for team, group in returner_groups:
            if len(group) == 0:
                continue
            avg_return = group["return_yards"].mean()
            # Return yards above expected, converted to points (~0.04 pts/yard)
            return_value = (avg_return - 23.0) * self.YARDS_TO_POINTS

            # Return rating (in points per return, not per game)
            # return_value is per-return average
            # When games_played is provided, scale to per-game; otherwise keep per-return
            if games_played and team in games_played:
                returns_per_game = len(group) / games_played[team]
                return_ratings[team] = return_value * returns_per_game
            else:
                return_ratings[team] = return_value

        # Combine coverage and return ratings
        # DETERMINISM: Sort for consistent iteration order
        all_teams = sorted(set(coverage_ratings.keys()) | set(return_ratings.keys()))
        kickoff_ratings = {}
        for team in all_teams:
            coverage = coverage_ratings.get(team, 0.0)
            returns = return_ratings.get(team, 0.0)
            kickoff_ratings[team] = coverage + returns

        # P3.9: Debug level for per-week logging
        logger.debug(
            f"Calculated kickoff ratings for {len(kickoff_ratings)} teams "
            f"from {len(kickoff_plays)} kickoffs"
        )

        return kickoff_ratings

    def calculate_all_st_ratings_from_plays(
        self,
        plays_df: pd.DataFrame,
        games_played: Optional[dict[str, int]] = None,
        max_week: int | None = None,
    ) -> None:
        """Calculate complete special teams ratings (FG + punt + kickoff) from plays.

        This is the main entry point for populating ST ratings from play-by-play data.
        All components are expressed as PBTA (Points Better Than Average) - the marginal
        point contribution per game compared to a league-average unit.

        Overall rating = FG rating + Punt rating + Kickoff rating (all in points)

        Args:
            plays_df: DataFrame with play-by-play data (needs play_type, play_text, offense, defense)
            games_played: Optional dict of team -> games played for normalization
            max_week: Maximum week allowed in training data (for data leakage prevention).
                      If provided, asserts that no plays exceed this week.
        """
        if plays_df.empty:
            logger.warning("Empty plays dataframe, skipping all ST rating calculations")
            return

        # DATA LEAKAGE GUARD: Verify no future weeks in training data
        if max_week is not None and "week" in plays_df.columns:
            actual_max = plays_df["week"].max()
            assert actual_max <= max_week, (
                f"DATA LEAKAGE in SpecialTeams: plays include week {actual_max} "
                f"but max_week={max_week}. Filter plays before calling."
            )

        # Calculate each component
        # FG ratings (stored directly in self.team_ratings)
        self.calculate_fg_ratings_from_plays(plays_df, games_played)

        # Get existing FG ratings to merge with
        fg_ratings = {team: r.field_goal_rating for team, r in self.team_ratings.items()}

        # Punt ratings
        punt_ratings = self.calculate_punt_ratings_from_plays(plays_df, games_played)

        # Kickoff ratings
        kickoff_ratings = self.calculate_kickoff_ratings_from_plays(plays_df, games_played)

        # Merge all ratings
        # DETERMINISM: Sort for consistent iteration order
        all_teams = sorted(set(fg_ratings.keys()) | set(punt_ratings.keys()) | set(kickoff_ratings.keys()))

        for team in all_teams:
            fg_rating = fg_ratings.get(team, 0.0)
            punt_rating = punt_ratings.get(team, 0.0)
            kick_rating = kickoff_ratings.get(team, 0.0)

            # Overall: sum of all components (all already in POINTS per game)
            # Each component is a marginal point contribution vs average
            overall = fg_rating + punt_rating + kick_rating

            self.team_ratings[team] = SpecialTeamsRating(
                team=team,
                field_goal_rating=fg_rating,
                punt_rating=punt_rating,
                kickoff_rating=kick_rating,
                overall_rating=overall,
                is_complete=True,  # P2.3: All 3 components populated
            )

        # P3.9: Debug level for per-week logging
        logger.debug(
            f"Calculated complete ST ratings for {len(self.team_ratings)} teams "
            f"(FG: {len(fg_ratings)}, Punt: {len(punt_ratings)}, Kickoff: {len(kickoff_ratings)})"
        )
