"""Home field advantage calculations."""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Conference-level HFA defaults
# Used when team doesn't have a specific adjustment
CONFERENCE_HFA_DEFAULTS = {
    # Power 4
    "SEC": 2.75,
    "Big Ten": 2.75,
    "Big 12": 2.50,
    "ACC": 2.50,
    # Group of 5
    "American Athletic": 2.25,
    "Mountain West": 2.25,
    "Sun Belt": 2.25,
    "MAC": 2.00,
    "Conference USA": 2.00,
    # Independent
    "FBS Independents": 2.50,
    # Default
    "default": 2.25,
}

# Team-specific HFA values (curated based on stadium environment)
# These are TOTAL HFA values, not adjustments
TEAM_HFA_VALUES = {
    # Elite environments (3.5 - 4.0)
    "LSU": 4.0,            # Death Valley night games
    "Alabama": 3.75,       # Bryant-Denny
    "Ohio State": 3.75,    # The Horseshoe, 105k
    "Penn State": 3.75,    # Whiteout, 107k
    "Texas A&M": 3.5,      # Kyle Field, 12th Man
    "Clemson": 3.5,        # Memorial Stadium
    "Michigan": 3.5,       # Big House

    # Strong environments (3.0 - 3.25)
    "Nebraska": 3.25,      # Memorial Stadium sellout streak
    "Wisconsin": 3.25,     # Camp Randall, Jump Around
    "Auburn": 3.0,         # Jordan-Hare
    "Florida": 3.0,        # The Swamp
    "Tennessee": 3.0,      # Neyland
    "Oklahoma": 3.0,
    "Oregon": 3.0,         # Autzen
    "Iowa": 3.0,           # Kinnick
    "Georgia": 3.0,        # Sanford
    "Notre Dame": 3.0,
    "Boise State": 3.0,    # Elite G5, blue turf
    "Missouri": 3.0,

    # Above average (2.75)
    "Texas": 2.75,
    "USC": 2.75,
    "Florida State": 2.75,
    "Miami": 2.75,
    "Virginia Tech": 2.75, # Enter Sandman
    "South Carolina": 2.75,
    "Oklahoma State": 2.75,
    "Washington": 2.75,
    "Louisville": 2.75,
    "Memphis": 2.75,
    "Minnesota": 2.75,
    "NC State": 2.75,
    "James Madison": 2.75, # Strong G5

    # Below average (2.0 - 2.25)
    "Louisiana": 2.25,
    "Louisiana Tech": 2.25,
    "Marshall": 2.25,
    "Miami (OH)": 2.25,
    "Middle Tennessee": 2.25,
    "Maryland": 2.0,
    "Rutgers": 2.0,
    "Vanderbilt": 2.0,
    "Duke": 2.0,

    # Weak environments (1.5 - 1.75)
    "Kent State": 1.75,
    "Akron": 1.75,
    "Bowling Green": 1.75,
    "UConn": 1.75,
    "Temple": 1.75,        # NFL stadium, sparse
    "UMass": 1.5,
}

# Trajectory modifier settings
# Programs on the rise get HFA boost, declining programs get penalty
TRAJECTORY_MAX_MODIFIER = 0.5  # Maximum adjustment (±0.5 points)
TRAJECTORY_BASELINE_YEARS = 3  # Years to establish baseline (older)
TRAJECTORY_RECENT_YEARS = 1    # Recent years to compare (newer)


class HomeFieldAdvantage:
    """
    Calculate home field advantage (HFA) from historical data.

    HFA is calculated as the average home team margin advantage
    after accounting for team strength differences.
    """

    def __init__(
        self,
        base_hfa: Optional[float] = None,
        data_path: Optional[Path] = None,
    ):
        """Initialize HFA calculator.

        Args:
            base_hfa: Default HFA value. If None, uses settings.
            data_path: Path to historical HFA data file.
        """
        settings = get_settings()
        self.base_hfa = base_hfa if base_hfa is not None else settings.base_hfa
        self.data_path = data_path or (settings.historical_dir / "hfa_by_team.json")

        self.team_hfa: dict[str, float] = {}
        self.trajectory_modifiers: dict[str, float] = {}  # Team trajectory adjustments
        self._load_historical_data()

    def _load_historical_data(self) -> None:
        """Load historical team-specific HFA data if available."""
        if self.data_path.exists():
            try:
                with open(self.data_path, "r") as f:
                    self.team_hfa = json.load(f)
                logger.info(f"Loaded HFA data for {len(self.team_hfa)} teams")
            except Exception as e:
                logger.warning(f"Could not load HFA data: {e}")

    def save_historical_data(self) -> None:
        """Save team-specific HFA data to file."""
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_path, "w") as f:
            json.dump(self.team_hfa, f, indent=2)
        logger.info(f"Saved HFA data for {len(self.team_hfa)} teams")

    def calculate_league_hfa(
        self,
        games_df: pd.DataFrame,
        team_ratings: Optional[dict[str, float]] = None,
    ) -> float:
        """Calculate overall league HFA from game data.

        Args:
            games_df: DataFrame with game results
            team_ratings: Optional dict of team overall ratings for adjustment

        Returns:
            League-wide HFA in points
        """
        if games_df.empty:
            return self.base_hfa

        # Filter to games with scores
        completed = games_df[
            games_df["home_points"].notna() & games_df["away_points"].notna()
        ].copy()

        if completed.empty:
            return self.base_hfa

        # Calculate home margins
        completed["home_margin"] = completed["home_points"] - completed["away_points"]

        # If we have team ratings, adjust for team strength
        if team_ratings:
            adjusted_margins = []
            for _, game in completed.iterrows():
                home_rating = team_ratings.get(game["home_team"], 0)
                away_rating = team_ratings.get(game["away_team"], 0)
                expected_margin = home_rating - away_rating
                adjusted_margin = game["home_margin"] - expected_margin
                adjusted_margins.append(adjusted_margin)

            return np.mean(adjusted_margins)

        # Simple average home margin (biased by schedule imbalances)
        return completed["home_margin"].mean()

    def calculate_team_hfa(
        self,
        team: str,
        games_df: pd.DataFrame,
        team_ratings: Optional[dict[str, float]] = None,
        min_home_games: int = 5,
    ) -> float:
        """Calculate team-specific HFA.

        Args:
            team: Team name
            games_df: DataFrame with game results
            team_ratings: Optional dict of team ratings
            min_home_games: Minimum home games required for team-specific HFA

        Returns:
            Team's HFA in points
        """
        # Get team's home games
        home_games = games_df[
            (games_df["home_team"] == team)
            & games_df["home_points"].notna()
        ].copy()

        if len(home_games) < min_home_games:
            return self.base_hfa

        # Calculate home margin
        home_games["home_margin"] = home_games["home_points"] - home_games["away_points"]

        # Adjust for opponent strength if ratings available
        if team_ratings:
            adjusted_margins = []
            team_rating = team_ratings.get(team, 0)

            for _, game in home_games.iterrows():
                away_rating = team_ratings.get(game["away_team"], 0)
                expected_margin = team_rating - away_rating
                adjusted_margin = game["home_margin"] - expected_margin
                adjusted_margins.append(adjusted_margin)

            team_hfa = np.mean(adjusted_margins)
        else:
            team_hfa = home_games["home_margin"].mean()

        # Regress toward league average (avoid extreme values from small samples)
        regression_weight = min(len(home_games) / 20, 1.0)
        final_hfa = (regression_weight * team_hfa) + ((1 - regression_weight) * self.base_hfa)

        return final_hfa

    def calculate_all_team_hfa(
        self,
        games_df: pd.DataFrame,
        team_ratings: Optional[dict[str, float]] = None,
        years: Optional[tuple] = None,
    ) -> dict[str, float]:
        """Calculate HFA for all teams.

        Args:
            games_df: DataFrame with game results
            team_ratings: Optional dict of team ratings
            years: Tuple of years to include (filters by season column)

        Returns:
            Dictionary mapping team names to HFA values
        """
        # Filter by years if specified
        if years and "season" in games_df.columns:
            games_df = games_df[games_df["season"].isin(years)]

        # Get all teams
        all_teams = set(games_df["home_team"]) | set(games_df["away_team"])

        # Calculate league HFA first
        self.base_hfa = self.calculate_league_hfa(games_df, team_ratings)
        logger.info(f"League HFA: {self.base_hfa:.2f} points")

        # Calculate team-specific HFA
        for team in all_teams:
            hfa = self.calculate_team_hfa(team, games_df, team_ratings)
            self.team_hfa[team] = hfa

        return self.team_hfa

    def calculate_team_hfa_with_regression(
        self,
        games_df: pd.DataFrame,
        team_ratings: dict[str, float],
        team_conferences: dict[str, str],
        min_games: int = 10,
        regress_games: int = 25,
    ) -> dict[str, float]:
        """Calculate opponent-adjusted HFA for all teams with conference-based regression.

        This method:
        1. Calculates raw HFA residuals (actual margin - expected margin based on ratings)
        2. Regresses toward conference-appropriate prior based on sample size
        3. Returns team-specific HFA values

        Args:
            games_df: DataFrame with game results (home_team, away_team, home_points, away_points)
            team_ratings: Dict mapping team name to power rating (e.g., from SP+ or JP+)
            team_conferences: Dict mapping team name to conference name
            min_games: Minimum home games to calculate team-specific HFA
            regress_games: Number of games for full regression (fewer = more regression)

        Returns:
            Dictionary mapping team names to calculated HFA values
        """
        # Filter to completed, non-neutral games
        completed = games_df[
            games_df["home_points"].notna() &
            games_df["away_points"].notna() &
            (games_df.get("neutral_site", False) == False)
        ].copy()

        if completed.empty:
            logger.warning("No completed home games for HFA calculation")
            return {}

        completed["home_margin"] = completed["home_points"] - completed["away_points"]

        # Calculate residuals for each game
        residuals_by_team = {}
        for _, game in completed.iterrows():
            home = game["home_team"]
            away = game["away_team"]

            home_rating = team_ratings.get(home, 0)
            away_rating = team_ratings.get(away, 0)

            # Expected margin without HFA
            expected = home_rating - away_rating
            actual = game["home_margin"]

            # Residual is the "extra" margin at home beyond talent difference
            residual = actual - expected

            if home not in residuals_by_team:
                residuals_by_team[home] = []
            residuals_by_team[home].append(residual)

        # Calculate regressed HFA for each team
        calculated_hfa = {}
        for team, residuals in residuals_by_team.items():
            n_games = len(residuals)

            if n_games < min_games:
                # Not enough data, use conference prior
                conf = team_conferences.get(team, "default")
                calculated_hfa[team] = CONFERENCE_HFA_PRIORS.get(conf, CONFERENCE_HFA_PRIORS["default"])
                continue

            # Raw average residual
            raw_hfa = np.mean(residuals)

            # Get conference prior for regression
            conf = team_conferences.get(team, "default")
            prior = CONFERENCE_HFA_PRIORS.get(conf, CONFERENCE_HFA_PRIORS["default"])

            # Bayesian regression: more games = trust data more
            # At regress_games, we weight data ~70%, prior ~30%
            data_weight = min(n_games / regress_games, 0.85)
            prior_weight = 1 - data_weight

            regressed_hfa = (raw_hfa * data_weight) + (prior * prior_weight)

            # Clamp to reasonable range (1.0 to 5.0)
            regressed_hfa = max(1.0, min(5.0, regressed_hfa))

            calculated_hfa[team] = regressed_hfa

        logger.info(f"Calculated HFA for {len(calculated_hfa)} teams")

        # Store in instance
        self.team_hfa = calculated_hfa
        return calculated_hfa

    def calculate_trajectory_modifiers(
        self,
        team_records: dict[str, dict[int, tuple[int, int]]],
        current_year: int,
    ) -> dict[str, float]:
        """Calculate HFA trajectory modifiers based on program improvement/decline.

        Compares recent performance to baseline to identify rising/falling programs.
        Rising programs get HFA boost (better crowds, more energy).
        Declining programs get HFA penalty.

        IMPORTANT: This uses only data from years STRICTLY PRIOR to current_year.
        For predicting 2024 games, we use:
          - Baseline: 2020, 2021, 2022 (3 years before recent)
          - Recent: 2023 (1 year before current_year)
        This prevents leakage of current-season results into predictions.

        Args:
            team_records: Dict mapping team -> {year: (wins, losses)}
                Example: {"Vanderbilt": {2022: (5, 7), 2023: (2, 10), 2024: (7, 5)}}
            current_year: The season being predicted (NOT included in calculation)

        Returns:
            Dictionary mapping team name to trajectory modifier (-0.5 to +0.5)
        """
        modifiers = {}

        # Calculate year ranges (strictly prior to current_year)
        # Recent: the TRAJECTORY_RECENT_YEARS years immediately before current_year
        # Baseline: the TRAJECTORY_BASELINE_YEARS years before the recent period
        recent_start = current_year - TRAJECTORY_RECENT_YEARS
        recent_end = current_year  # exclusive, so this is [recent_start, current_year)
        baseline_start = recent_start - TRAJECTORY_BASELINE_YEARS
        baseline_end = recent_start  # exclusive

        # Log the year ranges to prove no current-year data is used
        logger.info(
            f"Trajectory modifiers for {current_year}: "
            f"baseline years [{baseline_start}-{baseline_end - 1}], "
            f"recent years [{recent_start}-{recent_end - 1}] "
            f"(current year {current_year} NOT included)"
        )

        for team, records in team_records.items():
            # Get baseline win pct (older years)
            baseline_wins = 0
            baseline_games = 0
            for year in range(baseline_start, baseline_end):
                if year in records:
                    w, l = records[year]
                    baseline_wins += w
                    baseline_games += w + l

            # Get recent win pct (years immediately before current_year)
            # CRITICAL: Does NOT include current_year to prevent leakage
            recent_wins = 0
            recent_games = 0
            for year in range(recent_start, recent_end):
                if year in records:
                    w, l = records[year]
                    recent_wins += w
                    recent_games += w + l

            # Need enough data
            if baseline_games < 6 or recent_games < 6:
                continue

            baseline_pct = baseline_wins / baseline_games
            recent_pct = recent_wins / recent_games

            # Calculate improvement (positive = rising, negative = declining)
            improvement = recent_pct - baseline_pct

            # Scale to modifier range
            # +0.3 win pct improvement -> +0.5 HFA modifier
            # -0.3 win pct decline -> -0.5 HFA modifier
            modifier = (improvement / 0.3) * TRAJECTORY_MAX_MODIFIER
            modifier = max(-TRAJECTORY_MAX_MODIFIER, min(TRAJECTORY_MAX_MODIFIER, modifier))

            # Only apply meaningful modifiers
            if abs(modifier) >= 0.1:
                modifiers[team] = round(modifier, 2)

        self.trajectory_modifiers = modifiers
        logger.info(f"Calculated trajectory modifiers for {len(modifiers)} teams")

        return modifiers

    def set_trajectory_modifier(self, team: str, modifier: float) -> None:
        """Manually set a trajectory modifier for a team.

        Args:
            team: Team name
            modifier: HFA adjustment (-0.5 to +0.5)
        """
        modifier = max(-TRAJECTORY_MAX_MODIFIER, min(TRAJECTORY_MAX_MODIFIER, modifier))
        self.trajectory_modifiers[team] = modifier

    def get_hfa(
        self,
        home_team: str,
        neutral_site: bool = False,
        conference: Optional[str] = None,
        apply_trajectory: bool = True,
    ) -> float:
        """Get HFA for a specific matchup.

        Priority for base HFA:
        1. If neutral site, return 0
        2. If team has curated HFA in TEAM_HFA_VALUES, use it
        3. If team has dynamically calculated HFA, use it
        4. Fall back to conference default if known
        5. Fall back to base HFA (2.5)

        Then applies trajectory modifier if available and enabled.

        Args:
            home_team: Home team name
            neutral_site: If True, returns 0 (no HFA)
            conference: Team's conference (for fallback to conference default)
            apply_trajectory: If True, applies trajectory modifier for rising/declining programs

        Returns:
            HFA in points favoring home team
        """
        if neutral_site:
            return 0.0

        # Get base HFA
        if home_team in TEAM_HFA_VALUES:
            base = TEAM_HFA_VALUES[home_team]
        elif home_team in self.team_hfa:
            base = self.team_hfa[home_team]
        elif conference:
            base = CONFERENCE_HFA_DEFAULTS.get(conference, self.base_hfa)
        else:
            base = self.base_hfa

        # Apply trajectory modifier for rising/declining programs
        if apply_trajectory and home_team in self.trajectory_modifiers:
            modifier = self.trajectory_modifiers[home_team]
            base = base + modifier
            # Keep within reasonable bounds
            base = max(1.0, min(5.0, base))

        return base

    def get_summary_df(self) -> pd.DataFrame:
        """Get summary of team HFA values.

        Returns:
            DataFrame with team HFA values sorted by magnitude
        """
        if not self.team_hfa:
            return pd.DataFrame()

        data = [
            {"team": team, "hfa": hfa}
            for team, hfa in self.team_hfa.items()
        ]

        df = pd.DataFrame(data)
        df["hfa_vs_avg"] = df["hfa"] - self.base_hfa
        return df.sort_values("hfa", ascending=False).reset_index(drop=True)
