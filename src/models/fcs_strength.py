"""Dynamic FCS Strength Estimator.

Walk-forward safe estimator that calculates FCS team strength from prior FBS-vs-FCS game
margins. Uses Bayesian shrinkage toward a baseline when data is sparse, and maps strength
to a continuous penalty function.

Replaces the static ELITE_FCS_TEAMS frozenset with data-driven, year-adaptive estimates.

Key features:
- Walk-forward safe: Only uses games from weeks < current week
- HFA neutralization: Removes home field advantage bias from margin calculations
- Bayesian shrinkage: Heavy shrinkage early season, less as games accumulate
- Continuous penalty: Smooth mapping from margin to penalty (no discrete tiers)
- Fallback: Unknown FCS teams get baseline-derived penalty
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class FCSTeamStrength:
    """Strength estimate for a single FCS team."""

    team: str
    raw_margin: float  # Mean(FCS_pts - FBS_pts), HFA-neutralized, negative = FCS loses
    shrunk_margin: float  # After Bayesian shrinkage toward baseline
    n_games: int  # Number of FBS games observed
    penalty: float  # Final penalty in points (positive = FBS advantage)


@dataclass
class FCSStrengthEstimator:
    """Data-driven FCS penalty estimator using Bayesian shrinkage.

    For each FCS team, calculates expected margin against FBS from prior game results.
    Uses Empirical Bayes shrinkage to handle sparse data (heavily shrinks toward baseline
    when few games observed, trusts data more as games accumulate).

    HFA Neutralization:
        When Home is FBS: margin = fcs_pts - (fbs_pts - hfa_value)
        When Home is FCS: margin = (fcs_pts - hfa_value) - fbs_pts
        This removes venue bias from margin calculations.

    Shrinkage formula:
        shrink_factor = n_games / (n_games + k_fcs)
        shrunk_margin = baseline + (raw_margin - baseline) * shrink_factor

    With k_fcs=8:
        - 0 games: 100% baseline (unknown team gets default)
        - 4 games: 33% data, 67% baseline
        - 8 games: 50% data, 50% baseline
        - 16 games: 67% data, 33% baseline

    Penalty mapping (continuous):
        avg_loss = -shrunk_margin  # Negative margin = loss → positive avg_loss
        penalty = clamp(intercept + slope * avg_loss, min_pen, max_pen)

    With defaults (intercept=10, slope=0.8):
        - margin=+5 (elite FCS beats FBS by 5): avg_loss=-5 → penalty=10 pts (floor)
        - margin=-10 (good FCS loses by 10): avg_loss=10 → penalty=18 pts
        - margin=-28 (average FCS, baseline): avg_loss=28 → penalty=32.4 pts
        - margin=-40 (weak FCS): avg_loss=40 → penalty=42 pts
        - margin=-50 (very weak FCS): avg_loss=50 → penalty=45 pts (capped)
    """

    # HFA neutralization parameter (disabled by default - hurts Phase 1 more than helps Core)
    hfa_value: float = 0.0  # Set >0 to strip HFA from margins (tested: no benefit)

    # Shrinkage parameters
    k_fcs: float = 8.0  # Games needed for 50% trust in data
    baseline_margin: float = -28.0  # Prior for unknown FCS teams (FCS - FBS)

    # Penalty mapping parameters (calibrated to match static baseline)
    min_penalty: float = 10.0  # Floor for elite FCS teams
    max_penalty: float = 45.0  # Ceiling for weak FCS teams
    slope: float = 0.8  # Penalty increase per point of avg loss to FBS
    intercept: float = 10.0  # Base penalty (elite FCS with 0 avg loss)

    # Internal state
    _team_strengths: dict[str, FCSTeamStrength] = field(default_factory=dict)
    _team_margins: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize internal state."""
        self._team_strengths = {}
        self._team_margins = {}

    def reset(self) -> None:
        """Clear all state for a new season."""
        self._team_strengths.clear()
        self._team_margins.clear()

    def update_from_games(
        self,
        games_df: pl.DataFrame,
        fbs_teams: set[str],
        max_week: int,
    ) -> None:
        """Update FCS strength estimates from game results.

        Walk-forward safe: only uses games where week <= max_week.
        Uses vectorized Polars operations for performance.

        Args:
            games_df: Polars DataFrame with columns: week, home_team, away_team,
                      home_points, away_points
            fbs_teams: Set of FBS team names
            max_week: Maximum week to include (for walk-forward safety)
        """
        # Clear existing state and rebuild from scratch
        self._team_margins.clear()
        self._team_strengths.clear()

        # Filter to completed games up to max_week
        filtered = games_df.filter(
            (pl.col("week") <= max_week)
            & pl.col("home_points").is_not_null()
            & pl.col("away_points").is_not_null()
        )

        if filtered.height == 0:
            return

        # Convert fbs_teams set to a Series for vectorized membership check
        fbs_list = list(fbs_teams)

        # Vectorized pipeline: identify FBS-vs-FCS games and calculate HFA-neutralized margins
        fcs_games = (
            filtered.with_columns(
                [
                    pl.col("home_team").is_in(fbs_list).alias("home_is_fbs"),
                    pl.col("away_team").is_in(fbs_list).alias("away_is_fbs"),
                ]
            )
            # Filter to FBS-vs-FCS games only (exactly one team is FBS)
            .filter(pl.col("home_is_fbs") != pl.col("away_is_fbs"))
            .with_columns(
                [
                    # Identify FCS team
                    pl.when(pl.col("home_is_fbs"))
                    .then(pl.col("away_team"))
                    .otherwise(pl.col("home_team"))
                    .alias("fcs_team"),
                    # Calculate HFA-neutralized margin from FCS perspective
                    # When Home is FBS: margin = fcs_pts - (fbs_pts - hfa)
                    # When Home is FCS: margin = (fcs_pts - hfa) - fbs_pts
                    pl.when(pl.col("home_is_fbs"))
                    .then(
                        pl.col("away_points")
                        - (pl.col("home_points") - pl.lit(self.hfa_value))
                    )
                    .otherwise(
                        (pl.col("home_points") - pl.lit(self.hfa_value))
                        - pl.col("away_points")
                    )
                    .alias("hfa_adj_margin"),
                ]
            )
        )

        if fcs_games.height == 0:
            return

        # Aggregate per FCS team: n_games and mean margin
        team_stats = fcs_games.group_by("fcs_team").agg(
            [
                pl.col("hfa_adj_margin").count().alias("n_games"),
                pl.col("hfa_adj_margin").mean().alias("mean_margin"),
                pl.col("hfa_adj_margin").alias("margins"),  # Keep list for internal state
            ]
        )

        # Store margins in internal state and calculate strengths
        for row in team_stats.iter_rows(named=True):
            team = row["fcs_team"]
            self._team_margins[team] = row["margins"]

        # Recalculate shrunk margins and penalties
        self._recalculate_strengths()

        # Log summary
        n_teams = len(self._team_strengths)
        n_games = sum(s.n_games for s in self._team_strengths.values())
        if n_teams > 0:
            avg_margin = (
                sum(s.raw_margin for s in self._team_strengths.values()) / n_teams
            )
            logger.debug(
                f"FCS estimator updated: {n_teams} teams, {n_games} games, "
                f"avg margin={avg_margin:.1f}, hfa_adj={self.hfa_value} (through week {max_week})"
            )

    def _recalculate_strengths(self) -> None:
        """Recalculate shrunk margins and penalties from accumulated game margins."""
        for team, margins in self._team_margins.items():
            n_games = len(margins)
            raw_margin = sum(margins) / n_games if n_games > 0 else self.baseline_margin

            # Bayesian shrinkage toward baseline
            shrink_factor = n_games / (n_games + self.k_fcs)
            shrunk_margin = (
                self.baseline_margin
                + (raw_margin - self.baseline_margin) * shrink_factor
            )

            # Map to penalty
            penalty = self._margin_to_penalty(shrunk_margin)

            self._team_strengths[team] = FCSTeamStrength(
                team=team,
                raw_margin=raw_margin,
                shrunk_margin=shrunk_margin,
                n_games=n_games,
                penalty=penalty,
            )

    def _margin_to_penalty(self, margin: float) -> float:
        """Convert shrunk margin to penalty points.

        Linear mapping with clipping at min/max bounds.
        Uses negative of margin so that:
        - Positive margin (FCS win) → negative avg_loss → lower penalty
        - Negative margin (FCS loss) → positive avg_loss → higher penalty

        Args:
            margin: Expected FCS - FBS margin (negative = FCS loses, positive = FCS wins)

        Returns:
            Penalty points to apply in favor of FBS team
        """
        avg_loss = -margin  # FCS wins (positive margin) → lower penalty
        raw_penalty = self.intercept + self.slope * avg_loss
        return max(self.min_penalty, min(self.max_penalty, raw_penalty))

    def get_penalty(self, fcs_team: str) -> float:
        """Get penalty for an FCS team.

        Returns data-driven penalty if team has been observed, otherwise
        falls back to baseline-derived penalty.

        Args:
            fcs_team: FCS team name

        Returns:
            Penalty in points (positive = FBS advantage)
        """
        if fcs_team in self._team_strengths:
            return self._team_strengths[fcs_team].penalty

        # Fallback: use baseline (no shrinkage, since n=0)
        return self._margin_to_penalty(self.baseline_margin)

    def get_strength(self, fcs_team: str) -> Optional[FCSTeamStrength]:
        """Get full strength estimate for an FCS team.

        Returns None if team hasn't been observed.
        """
        return self._team_strengths.get(fcs_team)

    def get_all_strengths(self) -> dict[str, FCSTeamStrength]:
        """Get all FCS team strength estimates."""
        return self._team_strengths.copy()

    @property
    def baseline_penalty(self) -> float:
        """Get penalty for unknown FCS teams (baseline-derived)."""
        return self._margin_to_penalty(self.baseline_margin)

    def get_summary_stats(self) -> dict:
        """Get summary statistics for diagnostics."""
        if not self._team_strengths:
            return {
                "n_teams": 0,
                "n_games": 0,
                "avg_raw_margin": None,
                "avg_shrunk_margin": None,
                "avg_penalty": None,
                "baseline_margin": self.baseline_margin,
                "baseline_penalty": self.baseline_penalty,
                "hfa_value": self.hfa_value,
            }

        strengths = list(self._team_strengths.values())
        n_teams = len(strengths)
        n_games = sum(s.n_games for s in strengths)

        return {
            "n_teams": n_teams,
            "n_games": n_games,
            "avg_raw_margin": sum(s.raw_margin for s in strengths) / n_teams,
            "avg_shrunk_margin": sum(s.shrunk_margin for s in strengths) / n_teams,
            "avg_penalty": sum(s.penalty for s in strengths) / n_teams,
            "baseline_margin": self.baseline_margin,
            "baseline_penalty": self.baseline_penalty,
            "hfa_value": self.hfa_value,
            "min_penalty_team": min(strengths, key=lambda s: s.penalty).team
            if strengths
            else None,
            "max_penalty_team": max(strengths, key=lambda s: s.penalty).team
            if strengths
            else None,
        }
