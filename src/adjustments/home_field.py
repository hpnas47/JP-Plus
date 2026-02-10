"""Home field advantage calculations."""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import get_settings
from config.teams import normalize_team_name

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
        global_offset: float = 0.0,
    ):
        """Initialize HFA calculator.

        Args:
            base_hfa: Default HFA value. If None, uses settings.
            data_path: Path to historical HFA data file.
            global_offset: Points subtracted from ALL HFA values (for sweep testing).
                           Positive offset = reduce HFA. Default 0.0 (no change).
        """
        settings = get_settings()
        self.base_hfa = base_hfa if base_hfa is not None else settings.base_hfa
        self.data_path = data_path or (settings.historical_dir / "hfa_by_team.json")
        self.global_offset = global_offset

        self.team_hfa: dict[str, float] = {}
        self.trajectory_modifiers: dict[str, float] = {}  # Team trajectory adjustments
        self._load_historical_data()

    def _load_historical_data(self) -> None:
        """Load historical team-specific HFA data if available."""
        if self.data_path.exists():
            try:
                with open(self.data_path, "r") as f:
                    raw_data = json.load(f)
                # Normalize keys on load to ensure consistent key space
                self.team_hfa = {
                    normalize_team_name(team): hfa
                    for team, hfa in raw_data.items()
                }
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
        # P3.5: Avoid .copy() by computing derived values as Series directly
        score_mask = games_df["home_points"].notna() & games_df["away_points"].notna()
        completed = games_df[score_mask]

        if completed.empty:
            return self.base_hfa

        # Calculate home margins as Series (no DataFrame modification)
        home_margin = completed["home_points"] - completed["away_points"]

        # If we have team ratings, adjust for team strength (VECTORIZED)
        # P3.3: Replaced iterrows with .map() for ~10x speedup
        if team_ratings:
            # Map teams through normalization for lookup in team_ratings
            home_rating = completed["home_team"].apply(
                lambda t: team_ratings.get(normalize_team_name(t), team_ratings.get(t, 0))
            )
            away_rating = completed["away_team"].apply(
                lambda t: team_ratings.get(normalize_team_name(t), team_ratings.get(t, 0))
            )
            expected_margin = home_rating - away_rating
            adjusted_margin = home_margin - expected_margin
            return adjusted_margin.mean()

        # Simple average home margin (biased by schedule imbalances)
        return home_margin.mean()

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
        # P3.5: Avoid .copy() by computing derived values as Series directly
        home_mask = (games_df["home_team"] == team) & games_df["home_points"].notna()
        home_games = games_df[home_mask]

        if len(home_games) < min_home_games:
            return self.base_hfa

        # Calculate home margin as Series (no DataFrame modification)
        home_margin = home_games["home_points"] - home_games["away_points"]

        # Adjust for opponent strength if ratings available (VECTORIZED)
        # P3.3: Replaced iterrows with .map() for ~10x speedup
        if team_ratings:
            # Normalize team name for lookup in team_ratings
            normalized_team = normalize_team_name(team)
            team_rating = team_ratings.get(normalized_team, team_ratings.get(team, 0))
            # Map away teams through normalization for lookup
            away_rating = home_games["away_team"].apply(
                lambda t: team_ratings.get(normalize_team_name(t), team_ratings.get(t, 0))
            )
            expected_margin = team_rating - away_rating
            adjusted_margin = home_margin - expected_margin
            team_hfa = adjusted_margin.mean()
        else:
            team_hfa = home_margin.mean()

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
        # Clear stale entries from prior calls with different data/years
        self.team_hfa.clear()

        # Filter by years if specified
        if years and "season" in games_df.columns:
            games_df = games_df[games_df["season"].isin(years)]

        # Get all teams - DETERMINISM: Sort for consistent iteration order
        all_teams = sorted(set(games_df["home_team"]) | set(games_df["away_team"]))

        # Calculate league HFA for logging only (local variable)
        # Does NOT overwrite self.base_hfa to preserve caller's fallback behavior
        league_hfa = self.calculate_league_hfa(games_df, team_ratings)
        logger.info(f"League HFA: {league_hfa:.2f} points")

        # Calculate team-specific HFA (normalize keys for consistent storage)
        # Note: calculate_team_hfa uses self.base_hfa for regression target,
        # which is the configured default, not this dataset's league average
        for team in all_teams:
            hfa = self.calculate_team_hfa(team, games_df, team_ratings)
            normalized_team = normalize_team_name(team)
            self.team_hfa[normalized_team] = hfa

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
        # Clear stale entries from prior calls with different data/years
        self.team_hfa.clear()

        # Filter to completed, non-neutral games
        # P3.5: Avoid .copy() by computing derived values as Series directly
        neutral_col = games_df.get("neutral_site", pd.Series(False, index=games_df.index))
        completed_mask = (
            games_df["home_points"].notna() &
            games_df["away_points"].notna() &
            (neutral_col == False)
        )
        completed = games_df[completed_mask]

        if completed.empty:
            logger.warning("No completed home games for HFA calculation")
            return {}

        # Calculate all derived Series without modifying DataFrame
        # Map teams through normalization for lookup in team_ratings
        home_margin = completed["home_points"] - completed["away_points"]
        home_rating = completed["home_team"].apply(
            lambda t: team_ratings.get(normalize_team_name(t), team_ratings.get(t, 0))
        )
        away_rating = completed["away_team"].apply(
            lambda t: team_ratings.get(normalize_team_name(t), team_ratings.get(t, 0))
        )
        expected_margin = home_rating - away_rating
        residual = home_margin - expected_margin

        # Aggregate residuals by home team using groupby on Series
        # Create a temporary DataFrame for aggregation (minimal columns)
        residual_df = pd.DataFrame({
            "home_team": completed["home_team"].values,
            "residual": residual.values,
        })
        residual_stats = residual_df.groupby("home_team").agg(
            raw_hfa=("residual", "mean"),
            n_games=("residual", "count"),
        )

        # Calculate regressed HFA for each team
        calculated_hfa = {}
        for team in residual_stats.index:
            n_games = residual_stats.loc[team, "n_games"]
            raw_hfa = residual_stats.loc[team, "raw_hfa"]

            # Normalize team name for consistent storage and conference lookup
            normalized_team = normalize_team_name(team)

            # Look up conference using both normalized and raw names
            conf = team_conferences.get(normalized_team, team_conferences.get(team, "default"))

            if n_games < min_games:
                # Not enough data, use conference prior
                calculated_hfa[normalized_team] = CONFERENCE_HFA_DEFAULTS.get(conf, CONFERENCE_HFA_DEFAULTS["default"])
                continue

            # Get conference prior for regression
            prior = CONFERENCE_HFA_DEFAULTS.get(conf, CONFERENCE_HFA_DEFAULTS["default"])

            # Bayesian regression: more games = trust data more
            # At regress_games, we weight data ~70%, prior ~30%
            data_weight = min(n_games / regress_games, 0.85)
            prior_weight = 1 - data_weight

            regressed_hfa = (raw_hfa * data_weight) + (prior * prior_weight)

            # Clamp to reasonable range (1.0 to 5.0)
            regressed_hfa = max(1.0, min(5.0, regressed_hfa))

            calculated_hfa[normalized_team] = regressed_hfa

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
        # P3.9: Debug level for per-week logging
        logger.debug(
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

            # Only apply meaningful modifiers (normalize key for consistent storage)
            if abs(modifier) >= 0.1:
                normalized_team = normalize_team_name(team)
                modifiers[normalized_team] = round(modifier, 2)

        self.trajectory_modifiers = modifiers
        # P3.9: Debug level for per-week logging
        logger.debug(f"Calculated trajectory modifiers for {len(modifiers)} teams")

        return modifiers

    def set_trajectory_modifier(self, team: str, modifier: float) -> None:
        """Manually set a trajectory modifier for a team.

        Args:
            team: Team name
            modifier: HFA adjustment (-0.5 to +0.5)
        """
        modifier = max(-TRAJECTORY_MAX_MODIFIER, min(TRAJECTORY_MAX_MODIFIER, modifier))
        normalized_team = normalize_team_name(team)
        self.trajectory_modifiers[normalized_team] = modifier

    def get_hfa(
        self,
        home_team: str,
        neutral_site: bool = False,
        conference: Optional[str] = None,
        apply_trajectory: bool = True,
    ) -> tuple[float, str]:
        """Get HFA for a specific matchup with source tracking.

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
            Tuple of (HFA in points, source string describing where HFA came from)
        """
        if neutral_site:
            return 0.0, "neutral"

        # P2.13: Normalize team name for lookups
        normalized_team = normalize_team_name(home_team)

        # Get base HFA with source tracking
        if normalized_team in TEAM_HFA_VALUES:
            base = TEAM_HFA_VALUES[normalized_team]
            source = "curated"
        elif normalized_team in self.team_hfa:
            base = self.team_hfa[normalized_team]
            source = "dynamic"
        elif conference:
            base = CONFERENCE_HFA_DEFAULTS.get(conference, self.base_hfa)
            source = f"conf:{conference}" if conference in CONFERENCE_HFA_DEFAULTS else "fallback"
        else:
            base = self.base_hfa
            source = "fallback"

        # Apply trajectory modifier for rising/declining programs
        trajectory_applied = False
        if apply_trajectory and normalized_team in self.trajectory_modifiers:
            modifier = self.trajectory_modifiers[normalized_team]
            base = base + modifier
            # Keep within reasonable bounds
            base = max(1.0, min(5.0, base))
            trajectory_applied = True
            source += f"+traj({modifier:+.2f})"

        # Apply global offset (for HFA sweep testing)
        if self.global_offset != 0.0:
            base = max(0.5, base - self.global_offset)
            source += f"+offset({-self.global_offset:+.2f})"

        return base, source

    def get_hfa_value(
        self,
        home_team: str,
        neutral_site: bool = False,
        conference: Optional[str] = None,
        apply_trajectory: bool = True,
    ) -> float:
        """Get HFA value only (without source tracking).

        Convenience method for backward compatibility.

        Args:
            home_team: Home team name
            neutral_site: If True, returns 0 (no HFA)
            conference: Team's conference (for fallback)
            apply_trajectory: If True, applies trajectory modifier

        Returns:
            HFA in points favoring home team
        """
        hfa, _ = self.get_hfa(home_team, neutral_site, conference, apply_trajectory)
        return hfa

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

    def log_hfa_summary(self, teams: list[str]) -> None:
        """Log a summary of HFA sources for a set of teams.

        Args:
            teams: List of team names to summarize
        """
        source_counts = {"curated": 0, "dynamic": 0, "conference": 0, "fallback": 0}
        trajectory_count = 0

        for team in teams:
            hfa, source = self.get_hfa(team)

            # Count source type
            if source.startswith("curated"):
                source_counts["curated"] += 1
            elif source.startswith("dynamic"):
                source_counts["dynamic"] += 1
            elif source.startswith("conf:"):
                source_counts["conference"] += 1
            elif source.startswith("fallback"):
                source_counts["fallback"] += 1

            # Count trajectory modifiers
            if "+traj" in source:
                trajectory_count += 1

        logger.info(
            f"HFA sources for {len(teams)} teams: "
            f"curated={source_counts['curated']}, "
            f"dynamic={source_counts['dynamic']}, "
            f"conference={source_counts['conference']}, "
            f"fallback={source_counts['fallback']}, "
            f"with trajectory={trajectory_count}"
        )

    def get_hfa_breakdown(self, teams: list[str]) -> dict[str, dict]:
        """Get detailed HFA breakdown for each team.

        Args:
            teams: List of team names

        Returns:
            Dict mapping team name to {hfa: float, source: str}
        """
        breakdown = {}
        for team in teams:
            hfa, source = self.get_hfa(team)
            breakdown[team] = {"hfa": hfa, "source": source}
        return breakdown
