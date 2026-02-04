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

    # Points value of a field goal attempt by distance
    FG_POINT_VALUES = {
        (0, 30): 2.8,  # ~3 pts * make rate
        (30, 40): 2.5,
        (40, 50): 2.2,
        (50, 60): 1.7,
        (60, 100): 1.0,
    }

    # Expected punt net yards
    EXPECTED_PUNT_NET = 40.0

    # Expected kickoff touchback rate
    EXPECTED_TOUCHBACK_RATE = 0.60

    # Yards-to-points conversion factor (from expected points models)
    # ~0.04 points per yard of field position change
    YARDS_TO_POINTS = 0.04

    def __init__(self):
        """Initialize the special teams model."""
        self.team_ratings: dict[str, SpecialTeamsRating] = {}

    def _get_fg_expected_rate(self, distance: int) -> float:
        """Get expected FG make rate for a given distance."""
        for (low, high), rate in self.EXPECTED_FG_RATES.items():
            if low <= distance < high:
                return rate
        return 0.30  # Default for very long attempts

    def _get_fg_point_value(self, distance: int) -> float:
        """Get expected point value for a FG attempt at given distance."""
        for (low, high), value in self.FG_POINT_VALUES.items():
            if low <= distance < high:
                return value
        return 1.0

    def calculate_fg_paar(
        self,
        attempts: list[dict],  # List of {distance: int, made: bool}
    ) -> float:
        """Calculate Field Goal Points Added Above Replacement.

        Args:
            attempts: List of FG attempts with distance and result

        Returns:
            PAAR value (positive = above replacement, negative = below)
        """
        if not attempts:
            return 0.0

        total_paar = 0.0

        for attempt in attempts:
            distance = attempt.get("distance", 35)
            made = attempt.get("made", False)

            expected_rate = self._get_fg_expected_rate(distance)
            point_value = self._get_fg_point_value(distance)

            # PAAR = actual points - expected points
            actual_points = 3.0 if made else 0.0
            expected_points = 3.0 * expected_rate

            total_paar += actual_points - expected_points

        return total_paar

    def calculate_punt_rating(
        self,
        punts: list[dict],  # List of {gross: int, net: int, inside_20: bool, touchback: bool}
    ) -> float:
        """Calculate punt efficiency rating.

        Args:
            punts: List of punt data

        Returns:
            Punt rating (yards above expected per punt)
        """
        if not punts:
            return 0.0

        total_net = sum(p.get("net", 40) for p in punts)
        total_punts = len(punts)
        inside_20_count = sum(1 for p in punts if p.get("inside_20", False))
        touchback_count = sum(1 for p in punts if p.get("touchback", False))

        avg_net = total_net / total_punts
        inside_20_rate = inside_20_count / total_punts
        touchback_rate = touchback_count / total_punts

        # Base rating: net yards vs expected
        net_rating = avg_net - self.EXPECTED_PUNT_NET

        # Bonus for inside 20s, penalty for touchbacks
        positional_rating = (inside_20_rate * 5.0) - (touchback_rate * 3.0)

        return net_rating + positional_rating

    def calculate_kickoff_rating(
        self,
        kickoffs: list[dict],  # List of {touchback: bool, return_yards: int}
        kickoff_returns: list[dict],  # List of {return_yards: int} for returns allowed
    ) -> float:
        """Calculate kickoff efficiency rating.

        Args:
            kickoffs: Kickoffs by the team
            kickoff_returns: Returns against the team's kickoffs

        Returns:
            Kickoff rating (combined coverage and return efficiency)
        """
        coverage_rating = 0.0
        return_rating = 0.0

        # Coverage rating (for kickoffs by team)
        if kickoffs:
            touchbacks = sum(1 for k in kickoffs if k.get("touchback", False))
            touchback_rate = touchbacks / len(kickoffs)

            # Returns allowed
            returns = [k for k in kickoffs if not k.get("touchback", False)]
            if returns:
                avg_return = np.mean(
                    [k.get("return_yards", 20) for k in returns]
                )
                # Expected return is ~23 yards
                coverage_rating = (23.0 - avg_return) / 5.0

            # Bonus for touchback rate above expected
            coverage_rating += (touchback_rate - self.EXPECTED_TOUCHBACK_RATE) * 3.0

        # Return rating (for team's own returns)
        if kickoff_returns:
            avg_return = np.mean(
                [r.get("return_yards", 20) for r in kickoff_returns]
            )
            return_rating = (avg_return - 23.0) / 5.0

        return coverage_rating + return_rating

    def calculate_team_rating(
        self,
        team: str,
        fg_attempts: Optional[list[dict]] = None,
        punts: Optional[list[dict]] = None,
        kickoffs: Optional[list[dict]] = None,
        kickoff_returns: Optional[list[dict]] = None,
    ) -> SpecialTeamsRating:
        """Calculate comprehensive special teams rating for a team.

        Args:
            team: Team name
            fg_attempts: Field goal attempt data
            punts: Punt data
            kickoffs: Kickoff data
            kickoff_returns: Kickoff return data

        Returns:
            SpecialTeamsRating for the team
        """
        fg_rating = self.calculate_fg_paar(fg_attempts or [])
        punt_rating = self.calculate_punt_rating(punts or [])
        kick_rating = self.calculate_kickoff_rating(
            kickoffs or [], kickoff_returns or []
        )

        # Overall is sum, but normalized to per-game impact
        # Typical game has ~3 FG attempts, ~5 punts, ~5 kickoffs
        overall = (fg_rating / 3.0) + (punt_rating / 5.0) + (kick_rating / 5.0)

        rating = SpecialTeamsRating(
            team=team,
            field_goal_rating=fg_rating,
            punt_rating=punt_rating,
            kickoff_rating=kick_rating,
            overall_rating=overall,
        )

        self.team_ratings[team] = rating
        return rating

    def calculate_from_game_stats(
        self,
        team: str,
        games_df: pd.DataFrame,
    ) -> SpecialTeamsRating:
        """Calculate special teams rating from aggregated game statistics.

        This is a simplified calculation when detailed play-by-play isn't available.

        Args:
            team: Team name
            games_df: DataFrame with game-level stats

        Returns:
            SpecialTeamsRating for the team
        """
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

        # Calculate simplified ratings
        fg_rating = 0.0
        if fg_attempts > 0:
            actual_rate = fg_made / fg_attempts
            # Compare to expected rate for average distance (~35 yards)
            expected_rate = 0.78
            fg_rating = (actual_rate - expected_rate) * fg_attempts * 3.0

        punt_rating = 0.0
        if punt_count > 0:
            avg_gross = punt_yards / punt_count
            expected_gross = 42.0
            punt_rating = avg_gross - expected_gross

        # Kickoff rating defaulted without detailed data
        kick_rating = 0.0

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
        def extract_distance(text):
            if pd.isna(text):
                return None
            match = re.search(r'(\d+)\s*(?:Yd|yard)', str(text), re.IGNORECASE)
            return int(match.group(1)) if match else None

        fg_plays["distance"] = fg_plays["play_text"].apply(extract_distance)

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
        conditions = [
            distance < 30,
            (distance >= 30) & (distance < 40),
            (distance >= 40) & (distance < 50),
            (distance >= 50) & (distance < 60),
        ]
        choices = [0.92, 0.83, 0.72, 0.55]
        expected_rate = np.select(conditions, choices, default=0.30)

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

        # Convert to per-game rating
        # Typical team has ~2.5 FG attempts per game
        # We want a per-game adjustment value
        for team, row in team_fg.iterrows():
            total_paae = row["total_paae"]
            attempts = row["attempts"]

            # Estimate games from attempts (avg ~2.5 per game)
            estimated_games = max(1, attempts / 2.5)

            if games_played and team in games_played:
                estimated_games = games_played[team]

            # Per-game FG rating
            per_game_rating = total_paae / estimated_games

            # Create rating (FG-only for now, others zeroed)
            rating = SpecialTeamsRating(
                team=team,
                field_goal_rating=per_game_rating,
                punt_rating=0.0,
                kickoff_rating=0.0,
                overall_rating=per_game_rating,  # FG-only
            )
            self.team_ratings[team] = rating

        logger.info(
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

        # Parse punt distance from play_text
        # Examples: "John Smith punt for 45 yards", "punt for 52 Yds"
        def extract_punt_yards(text):
            if pd.isna(text):
                return None
            match = re.search(r'punt\s+(?:for\s+)?(\d+)\s*(?:Yd|yard)', str(text), re.IGNORECASE)
            return int(match.group(1)) if match else None

        # Detect touchbacks (ball goes into end zone)
        def is_touchback(text):
            if pd.isna(text):
                return False
            return 'touchback' in str(text).lower()

        # Detect inside-20 (fair catch or downed inside 20)
        def is_inside_20(text):
            if pd.isna(text):
                return False
            text_lower = str(text).lower()
            # Look for indicators of inside-20
            return ('inside' in text_lower and '20' in text_lower) or \
                   ('downed' in text_lower) or \
                   ('fair catch' in text_lower and 'to the' in text_lower)

        # Parse return yards
        def extract_return_yards(text):
            if pd.isna(text):
                return 0
            # "returned by X for Y yards" or "return for Y yards"
            match = re.search(r'return(?:ed)?.*?(\d+)\s*(?:Yd|yard)', str(text), re.IGNORECASE)
            return int(match.group(1)) if match else 0

        punt_plays["gross_yards"] = punt_plays["play_text"].apply(extract_punt_yards)
        punt_plays["is_touchback"] = punt_plays["play_text"].apply(is_touchback)
        punt_plays["is_inside_20"] = punt_plays["play_text"].apply(is_inside_20)
        punt_plays["return_yards"] = punt_plays["play_text"].apply(extract_return_yards)

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

        # Touchback: net = min(gross, 55) since touchback from far means wasted yards
        # Normal: net = gross - return_yards
        punt_plays["net_yards"] = np.where(
            is_touchback,
            np.minimum(gross, 55),
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

        # Convert to per-game rating
        punt_ratings = {}
        for team, row in team_punts.iterrows():
            punt_count = row["punt_count"]
            # Estimate games from punt count (avg ~5 punts per game)
            estimated_games = max(1, punt_count / 5.0)
            if games_played and team in games_played:
                estimated_games = games_played[team]

            per_game_rating = row["total_value"] / estimated_games
            punt_ratings[team] = per_game_rating

        logger.info(
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

        # Detect touchbacks
        def is_touchback(text, play_type):
            if pd.isna(text):
                return False
            text_lower = str(text).lower()
            # Touchback if explicitly stated or "for a touchback"
            return 'touchback' in text_lower

        # Parse return yards
        def extract_return_yards(text):
            if pd.isna(text):
                return None
            # "returned by X for Y yards" or "return for Y yards"
            match = re.search(r'return(?:ed)?.*?(\d+)\s*(?:Yd|yard)', str(text), re.IGNORECASE)
            return int(match.group(1)) if match else None

        kickoff_plays["is_touchback"] = kickoff_plays.apply(
            lambda r: is_touchback(r["play_text"], r["play_type"]), axis=1
        )
        kickoff_plays["return_yards"] = kickoff_plays["play_text"].apply(extract_return_yards)

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

            # Estimate games
            estimated_games = max(1, total_kicks / 5.0)
            if games_played and team in games_played:
                estimated_games = games_played[team]

            # Per-game coverage rating (in points)
            kicks_per_game = total_kicks / estimated_games
            coverage_ratings[team] = (tb_bonus + return_saved) * kicks_per_game / 5.0

        # Return rating (for returning teams) in POINTS
        return_ratings = {}
        returner_groups = kickoff_plays[~kickoff_plays["is_touchback"]].groupby("defense")
        for team, group in returner_groups:
            if len(group) == 0:
                continue
            avg_return = group["return_yards"].mean()
            # Return yards above expected, converted to points (~0.04 pts/yard)
            return_value = (avg_return - 23.0) * self.YARDS_TO_POINTS

            # Estimate games
            estimated_games = max(1, len(group) / 3.0)  # ~3 non-TB kickoffs per game to return
            if games_played and team in games_played:
                estimated_games = games_played[team]

            # Per-game return rating (in points)
            returns_per_game = len(group) / estimated_games
            return_ratings[team] = return_value * returns_per_game / 3.0

        # Combine coverage and return ratings
        # DETERMINISM: Sort for consistent iteration order
        all_teams = sorted(set(coverage_ratings.keys()) | set(return_ratings.keys()))
        kickoff_ratings = {}
        for team in all_teams:
            coverage = coverage_ratings.get(team, 0.0)
            returns = return_ratings.get(team, 0.0)
            kickoff_ratings[team] = coverage + returns

        logger.info(
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
            )

        logger.info(
            f"Calculated complete ST ratings for {len(self.team_ratings)} teams "
            f"(FG: {len(fg_ratings)}, Punt: {len(punt_ratings)}, Kickoff: {len(kickoff_ratings)})"
        )
