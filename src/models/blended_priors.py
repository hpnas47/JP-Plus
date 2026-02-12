"""Blended Prior System.

Combines SP+ and JP+ own-priors with week-dependent weights to leverage the
strengths of both:
- SP+ is better for very early-season predictions (Weeks 1-2)
- Own-prior is better at predicting end-of-season ratings
- Blending shifts weight from SP+ to own-prior as season progresses

The blend schedule is tunable and can be optimized via tune_blended_priors.py.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BlendSchedule:
    """Week-dependent blending weights for SP+ vs Own-Prior.

    Attributes:
        week_weights: Dict mapping week number to (w_sp, w_own) tuple
        default_sp_weight: SP+ weight to use for weeks not in week_weights
    """

    week_weights: dict[int, tuple[float, float]] = field(default_factory=dict)
    default_sp_weight: float = 0.35  # Used for weeks >= 4

    def __post_init__(self):
        # Initialize default schedule if empty
        if not self.week_weights:
            self.week_weights = {
                1: (0.65, 0.35),  # Week 1: 65% SP+, 35% Own
                2: (0.55, 0.45),  # Week 2: 55% SP+, 45% Own
                3: (0.45, 0.55),  # Week 3: 45% SP+, 55% Own
                # Week 4+: use default_sp_weight
            }

    def get_weights(self, week: int) -> tuple[float, float]:
        """Get (w_sp, w_own) for a given week.

        Args:
            week: Week number (1-indexed)

        Returns:
            Tuple of (sp_weight, own_weight) that sum to 1.0
        """
        if week in self.week_weights:
            return self.week_weights[week]

        # For weeks not explicitly set, use default
        w_sp = self.default_sp_weight
        w_own = 1.0 - w_sp
        return (w_sp, w_own)

    @classmethod
    def from_dict(cls, config: dict) -> "BlendSchedule":
        """Create BlendSchedule from config dict.

        Args:
            config: Dict with keys like "week_1", "week_2", etc. mapping to SP+ weights
                    and optional "default" for weeks >= 4

        Returns:
            Configured BlendSchedule instance
        """
        week_weights = {}
        default = 0.35

        for key, value in config.items():
            if key == "default":
                default = value
            elif key.startswith("week_"):
                week = int(key.split("_")[1])
                w_sp = value
                w_own = 1.0 - w_sp
                week_weights[week] = (w_sp, w_own)

        return cls(week_weights=week_weights, default_sp_weight=default)


@dataclass
class BlendedRating:
    """A blended preseason rating combining SP+ and own-prior."""

    team: str
    sp_rating: Optional[float]
    own_rating: Optional[float]
    blended_rating: float
    sp_weight: float
    own_weight: float
    source: str  # "sp_only", "own_only", or "blended"


class BlendedPriorGenerator:
    """Generates blended priors from SP+ and JP+ own-priors.

    The blending weights shift over time, starting SP+-heavy in Week 1
    and transitioning to own-prior-heavy by Week 4+.
    """

    def __init__(
        self,
        sp_generator,
        own_generator,
        schedule: Optional[BlendSchedule] = None,
    ):
        """Initialize with both prior generators.

        Args:
            sp_generator: PreseasonPriors instance (SP+ source)
            own_generator: OwnPriorGenerator instance (JP+ own-prior source)
            schedule: BlendSchedule for week-dependent weights (default schedule if None)
        """
        self.sp_gen = sp_generator
        self.own_gen = own_generator
        self.schedule = schedule or BlendSchedule()

        # Store blended ratings after calculation
        self.blended_ratings: dict[str, BlendedRating] = {}
        self.current_week: int = 1

        logger.debug(f"BlendedPriorGenerator initialized with schedule: {self.schedule.week_weights}")

    def calculate_blended_ratings(
        self,
        year: int,
        week: int = 1,
    ) -> dict[str, BlendedRating]:
        """Calculate blended ratings for a given year and week.

        Args:
            year: Season year
            week: Current week (affects blend weights)

        Returns:
            Dict mapping team name to BlendedRating
        """
        self.current_week = week
        w_sp, w_own = self.schedule.get_weights(week)

        logger.info(f"Calculating blended priors for {year} week {week}: w_sp={w_sp:.2f}, w_own={w_own:.2f}")

        # Get SP+ ratings if not already calculated
        if not self.sp_gen.preseason_ratings:
            self.sp_gen.calculate_preseason_ratings(year)

        # Get own-prior ratings if not already calculated
        if not self.own_gen.preseason_ratings:
            self.own_gen.calculate_preseason_ratings(year)

        # Collect all teams from both sources
        sp_teams = set(self.sp_gen.preseason_ratings.keys()) if self.sp_gen.preseason_ratings else set()
        own_teams = set(self.own_gen.preseason_ratings.keys()) if self.own_gen.preseason_ratings else set()
        all_teams = sp_teams | own_teams

        self.blended_ratings = {}

        for team in sorted(all_teams):
            sp_rating = self.sp_gen.get_preseason_rating(team) if team in sp_teams else None
            own_rating = self.own_gen.get_preseason_rating(team) if team in own_teams else None

            # Handle missing ratings with fallback
            if sp_rating is not None and own_rating is not None:
                # Both available: blend
                blended = w_sp * sp_rating + w_own * own_rating
                source = "blended"
                actual_w_sp = w_sp
                actual_w_own = w_own
            elif sp_rating is not None:
                # Only SP+ available
                blended = sp_rating
                source = "sp_only"
                actual_w_sp = 1.0
                actual_w_own = 0.0
            elif own_rating is not None:
                # Only own-prior available (rare)
                blended = own_rating
                source = "own_only"
                actual_w_sp = 0.0
                actual_w_own = 1.0
            else:
                # Neither available - skip
                continue

            self.blended_ratings[team] = BlendedRating(
                team=team,
                sp_rating=sp_rating,
                own_rating=own_rating,
                blended_rating=blended,
                sp_weight=actual_w_sp,
                own_weight=actual_w_own,
                source=source,
            )

        # Log coverage
        n_blended = sum(1 for r in self.blended_ratings.values() if r.source == "blended")
        n_sp_only = sum(1 for r in self.blended_ratings.values() if r.source == "sp_only")
        n_own_only = sum(1 for r in self.blended_ratings.values() if r.source == "own_only")

        logger.info(
            f"Blended ratings: {n_blended} blended, {n_sp_only} SP+-only, {n_own_only} own-only"
        )

        return self.blended_ratings

    @property
    def preseason_ratings(self) -> dict:
        """Return blended ratings dict for compatibility with PreseasonPriors interface.

        This allows the BlendedPriorGenerator to be used as a drop-in replacement
        for PreseasonPriors in the backtest pipeline.
        """
        return self.blended_ratings

    def get_preseason_rating(self, team: str) -> float:
        """Get the blended preseason rating for a team.

        Args:
            team: Team name

        Returns:
            Blended rating, or 0.0 if not found
        """
        if team in self.blended_ratings:
            return self.blended_ratings[team].blended_rating
        return 0.0

    def get_rating_breakdown(self, team: str) -> Optional[BlendedRating]:
        """Get full rating breakdown for a team.

        Args:
            team: Team name

        Returns:
            BlendedRating object with component details, or None
        """
        return self.blended_ratings.get(team)

    def blend_with_inseason(
        self,
        inseason_ratings: dict[str, float],
        games_played: int,
        games_for_full_weight: int = 9,
        talent_floor_weight: float = 0.08,
    ) -> dict[str, float]:
        """Blend blended preseason ratings with in-season ratings.

        Same interface as PreseasonPriors.blend_with_inseason() for drop-in
        replacement in the pipeline.

        Args:
            inseason_ratings: Current in-season ratings
            games_played: Average games played per team
            games_for_full_weight: Games needed before in-season dominates
            talent_floor_weight: Base talent weight at week 0

        Returns:
            Blended ratings dictionary
        """
        # Recalculate blended SP+/own weights for current week
        current_week = games_played + 1
        if current_week != self.current_week:
            # Update blended ratings for this week's SP+/own balance
            w_sp, w_own = self.schedule.get_weights(current_week)
            for team, rating in self.blended_ratings.items():
                if rating.sp_rating is not None and rating.own_rating is not None:
                    new_blend = w_sp * rating.sp_rating + w_own * rating.own_rating
                    self.blended_ratings[team] = BlendedRating(
                        team=team,
                        sp_rating=rating.sp_rating,
                        own_rating=rating.own_rating,
                        blended_rating=new_blend,
                        sp_weight=w_sp,
                        own_weight=w_own,
                        source="blended",
                    )
            self.current_week = current_week
            logger.debug(f"Recalculated blended weights for week {current_week}: SP+={w_sp:.0%}, own={w_own:.0%}")

        # Calculate prior weights (same formula as PreseasonPriors)
        if games_played <= 0:
            prior_weight = 1.0 - talent_floor_weight
            inseason_weight = 0.0
        elif games_played >= games_for_full_weight:
            prior_weight = 0.05
            inseason_weight = 1.0 - prior_weight - talent_floor_weight
        else:
            t = games_played / games_for_full_weight
            prior_weight = 0.92 * (1.0 - t ** 1.5) ** 1.2
            prior_weight = max(prior_weight, 0.05)
            inseason_weight = 1.0 - prior_weight - talent_floor_weight

        # Decay talent floor weight over time
        effective_talent_weight = self._calculate_decayed_talent_weight(
            games_played, w_base=talent_floor_weight
        )
        if games_played > 0:
            inseason_weight = 1.0 - prior_weight - effective_talent_weight

        logger.debug(
            f"Blending week {games_played}: prior={prior_weight:.1%}, "
            f"inseason={inseason_weight:.1%}, talent={effective_talent_weight:.1%}"
        )

        # Blend ratings
        all_teams = sorted(set(inseason_ratings.keys()) | set(self.blended_ratings.keys()))
        blended = {}

        for team in all_teams:
            preseason = self.get_preseason_rating(team)
            inseason = inseason_ratings.get(team, 0.0)

            # Get normalized talent for floor (from own-prior if available)
            talent_rating = 0.0
            if team in self.own_gen.preseason_ratings:
                talent_rating = self.own_gen.preseason_ratings[team].talent_rating_normalized
            elif team in self.sp_gen.preseason_ratings:
                # Fallback to SP+ talent if available
                sp_rating = self.sp_gen.preseason_ratings[team]
                if hasattr(sp_rating, 'talent_rating_normalized'):
                    talent_rating = sp_rating.talent_rating_normalized

            blended[team] = (
                preseason * prior_weight
                + inseason * inseason_weight
                + talent_rating * effective_talent_weight
            )

        return blended

    def _calculate_decayed_talent_weight(
        self,
        games_played: int,
        w_base: float = 0.08,
        w_min: float = 0.03,
        target_week: int = 10,
    ) -> float:
        """Calculate decayed talent floor weight."""
        if games_played <= 0:
            return w_base
        decay_rate = (w_base - w_min) / target_week
        return max(w_base - games_played * decay_rate, w_min)

    def get_disagreements(self, threshold: float = 3.0) -> dict[str, dict]:
        """Get teams where SP+ and own-prior disagree significantly.

        Args:
            threshold: Minimum absolute difference to flag (default 3.0 pts)

        Returns:
            Dict mapping team to disagreement info
        """
        disagreements = {}

        for team, rating in self.blended_ratings.items():
            if rating.sp_rating is None or rating.own_rating is None:
                continue

            diff = rating.own_rating - rating.sp_rating

            if abs(diff) >= threshold:
                disagreements[team] = {
                    "sp_rating": rating.sp_rating,
                    "own_rating": rating.own_rating,
                    "difference": diff,
                    "direction": "own_higher" if diff > 0 else "sp_higher",
                }

        return disagreements

    def get_ratings_dataframe(self):
        """Get blended ratings as DataFrame."""
        import pandas as pd

        if not self.blended_ratings:
            return pd.DataFrame()

        rows = []
        for team, rating in self.blended_ratings.items():
            rows.append({
                "team": team,
                "blended_rating": rating.blended_rating,
                "sp_rating": rating.sp_rating,
                "own_rating": rating.own_rating,
                "sp_weight": rating.sp_weight,
                "own_weight": rating.own_weight,
                "source": rating.source,
            })

        return pd.DataFrame(rows).sort_values("blended_rating", ascending=False)


def create_blended_generator(
    client,
    historical_ratings: dict,
    schedule: Optional[BlendSchedule] = None,
    own_params: Optional[dict] = None,
) -> BlendedPriorGenerator:
    """Factory function to create a BlendedPriorGenerator.

    Args:
        client: CFBD API client
        historical_ratings: Historical JP+ ratings dict
        schedule: Blend schedule (default if None)
        own_params: Parameters for OwnPriorGenerator (default if None)

    Returns:
        Configured BlendedPriorGenerator instance
    """
    from src.models.preseason_priors import PreseasonPriors
    from src.models.own_priors import OwnPriorGenerator

    sp_gen = PreseasonPriors(client=client)
    own_gen = OwnPriorGenerator(historical_ratings, params=own_params, client=client)

    return BlendedPriorGenerator(sp_gen, own_gen, schedule)
