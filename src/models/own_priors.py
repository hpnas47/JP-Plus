"""JP+ Own-Prior System.

Generate preseason priors from JP+ historical ratings instead of SP+.
Uses fitted Ridge regression model to predict current year ratings from:
- Prior year JP+ rating
- Returning Production (percent PPA returning)
- Talent composite

Walk-forward safe: predictions use only data available before season starts.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default parameters (can be overridden by loading from file)
DEFAULT_PARAMS = {
    "blending": {
        "method": "direct",
        "ridge_coef_prior": 0.5119,
        "ridge_coef_rp": 4.9959,
        "ridge_coef_talent": 0.0209,
        "ridge_intercept": -15.3822,
    },
    "reversion": {
        "beta_all": 0.699,
        "use_tier_specific": False,
    },
}

# P4 conferences (same as PreseasonPriors)
P4_CONFERENCES = {'SEC', 'Big Ten', 'Big 12', 'ACC', 'FBS Independents'}


@dataclass
class OwnPriorRating:
    """Preseason rating generated from JP+ historical ratings."""

    team: str
    prior_jp_rating: float        # Previous year JP+ rating
    prior_offense: float          # Previous year offensive rating
    prior_defense: float          # Previous year defensive rating
    talent_rating: float          # Talent composite (raw)
    talent_rating_normalized: float  # Talent normalized to rating scale
    returning_ppa: float          # RP percentage (0-1)
    combined_rating: float        # Final blended prior
    confidence: float             # 0-1 based on RP + data quality
    tier: str                     # P4 or G5


class OwnPriorGenerator:
    """Generate preseason priors from JP+ historical ratings.

    Uses a fitted Ridge regression model to predict next-year ratings:
        predicted = coef_prior * prior_rating + coef_rp * rp + coef_talent * talent + intercept

    Provides the same interface as PreseasonPriors for drop-in replacement.
    """

    def __init__(
        self,
        historical_ratings: dict,
        params: Optional[dict] = None,
        client=None,
    ):
        """Initialize with historical ratings and model parameters.

        Args:
            historical_ratings: Dict mapping year (str) -> team -> ratings
            params: Fitted parameters dict (or None to use defaults)
            client: CFBD API client for fetching talent/RP data
        """
        self.historical_ratings = historical_ratings
        self.params = params or DEFAULT_PARAMS
        self.client = client
        self.preseason_ratings: dict[str, OwnPriorRating] = {}

        # Extract ridge coefficients
        blending = self.params.get("blending", DEFAULT_PARAMS["blending"])
        self.coef_prior = blending.get("ridge_coef_prior", 0.5119)
        self.coef_rp = blending.get("ridge_coef_rp", 4.9959)
        self.coef_talent = blending.get("ridge_coef_talent", 0.0209)
        self.intercept = blending.get("ridge_intercept", -15.3822)

        logger.debug(
            f"OwnPriorGenerator initialized with coefficients: "
            f"prior={self.coef_prior:.4f}, rp={self.coef_rp:.4f}, "
            f"talent={self.coef_talent:.4f}, intercept={self.intercept:.4f}"
        )

    @classmethod
    def from_files(
        cls,
        ratings_path: str = "data/historical_jp_ratings.json",
        params_path: str = "data/outputs/own_priors/optimal_params.json",
        client=None,
    ) -> "OwnPriorGenerator":
        """Factory method to create from saved files.

        Args:
            ratings_path: Path to historical ratings JSON
            params_path: Path to fitted parameters JSON
            client: CFBD API client

        Returns:
            Configured OwnPriorGenerator instance
        """
        with open(ratings_path) as f:
            ratings = json.load(f)

        params = None
        if Path(params_path).exists():
            with open(params_path) as f:
                params = json.load(f)

        return cls(ratings, params, client)

    def calculate_preseason_ratings(
        self,
        year: int,
        talent_scores: Optional[dict[str, float]] = None,
        returning_production: Optional[dict[str, float]] = None,
        team_conferences: Optional[dict[str, str]] = None,
    ) -> dict[str, OwnPriorRating]:
        """Generate preseason priors for a given year.

        Uses previous year's JP+ ratings + current year's talent and RP.

        Args:
            year: Season year to generate priors for
            talent_scores: Optional pre-fetched talent scores
            returning_production: Optional pre-fetched RP data
            team_conferences: Optional pre-fetched conference mapping

        Returns:
            Dict mapping team name to OwnPriorRating
        """
        prior_year = year - 1
        prior_year_str = str(prior_year)

        if prior_year_str not in self.historical_ratings:
            logger.warning(f"No historical ratings for {prior_year}, cannot generate priors")
            return {}

        prior_ratings = self.historical_ratings[prior_year_str]

        # Fetch supplementary data if not provided
        if self.client is not None:
            if talent_scores is None:
                talent_scores = self._fetch_talent(year)
            if returning_production is None:
                returning_production = self._fetch_returning_production(year)
            if team_conferences is None:
                team_conferences = self._fetch_conferences(year)

        talent_scores = talent_scores or {}
        returning_production = returning_production or {}
        team_conferences = team_conferences or {}

        # Calculate mean talent for normalization
        talent_values = list(talent_scores.values())
        if talent_values:
            mean_talent = np.mean(talent_values)
            std_talent = np.std(talent_values)
        else:
            mean_talent = 700.0  # Approximate mean
            std_talent = 200.0

        self.preseason_ratings = {}

        for team, prior in prior_ratings.items():
            # Get prior year rating
            prior_overall = prior.get("overall", 0.0)
            prior_offense = prior.get("offense", 0.0)
            prior_defense = prior.get("defense", 0.0)

            # Get current year talent and RP
            talent = talent_scores.get(team, mean_talent)
            rp = returning_production.get(team, 0.5)  # Default to 50% if missing

            # Determine tier
            conf = team_conferences.get(team, "Unknown")
            tier = "P4" if conf in P4_CONFERENCES else "G5"

            # Apply Ridge model: predicted = coef * features + intercept
            predicted_rating = (
                self.coef_prior * prior_overall
                + self.coef_rp * rp
                + self.coef_talent * talent
                + self.intercept
            )

            # Normalize talent to rating scale (for blend_with_inseason compatibility)
            # Same approach as PreseasonPriors
            talent_normalized = (talent - mean_talent) / std_talent * 10.0 if std_talent > 0 else 0.0

            # Calculate confidence based on RP and data availability
            confidence = self._calculate_confidence(rp, talent, prior_overall)

            self.preseason_ratings[team] = OwnPriorRating(
                team=team,
                prior_jp_rating=prior_overall,
                prior_offense=prior_offense,
                prior_defense=prior_defense,
                talent_rating=talent,
                talent_rating_normalized=talent_normalized,
                returning_ppa=rp,
                combined_rating=predicted_rating,
                confidence=confidence,
                tier=tier,
            )

        logger.info(
            f"Generated {len(self.preseason_ratings)} own-priors for {year} "
            f"(from {prior_year} JP+ ratings)"
        )

        return self.preseason_ratings

    def get_preseason_rating(self, team: str) -> float:
        """Get the preseason rating for a team.

        Args:
            team: Team name

        Returns:
            Preseason rating, or 0.0 if not found
        """
        if team in self.preseason_ratings:
            return self.preseason_ratings[team].combined_rating
        return 0.0

    def blend_with_inseason(
        self,
        inseason_ratings: dict[str, float],
        games_played: int,
        games_for_full_weight: int = 9,
        talent_floor_weight: float = 0.08,
    ) -> dict[str, float]:
        """Blend preseason ratings with in-season ratings.

        Same interface as PreseasonPriors.blend_with_inseason() for drop-in
        replacement.

        Uses non-linear fade matching SP+ methodology:
        - Weeks 0-3: Priors dominant (~95% -> ~65%)
        - Weeks 4-5: Tipping point (~50%)
        - Weeks 8-9: Priors nearly gone (~5%)

        Args:
            inseason_ratings: Current in-season ratings
            games_played: Average games played per team (or weeks into season)
            games_for_full_weight: Games needed before in-season dominates
            talent_floor_weight: Base talent weight at week 0

        Returns:
            Blended ratings dictionary
        """
        # Calculate weights (same formula as PreseasonPriors)
        if games_played <= 0:
            prior_weight = 1.0 - talent_floor_weight
            inseason_weight = 0.0
        elif games_played >= games_for_full_weight:
            prior_weight = 0.05  # Small residual prior weight
            inseason_weight = 1.0 - prior_weight - talent_floor_weight
        else:
            # Sigmoid-style curve
            t = games_played / games_for_full_weight
            prior_weight = 0.92 * (1.0 - t ** 1.5) ** 1.2
            prior_weight = max(prior_weight, 0.05)
            inseason_weight = 1.0 - prior_weight - talent_floor_weight

        # Decay talent floor weight over time
        effective_talent_weight = self._calculate_decayed_talent_weight(
            games_played, w_base=talent_floor_weight
        )
        # Adjust inseason weight for decayed talent
        if games_played > 0:
            inseason_weight = 1.0 - prior_weight - effective_talent_weight

        logger.debug(
            f"Blending week {games_played}: prior={prior_weight:.1%}, "
            f"inseason={inseason_weight:.1%}, talent={effective_talent_weight:.1%}"
        )

        # Blend ratings
        all_teams = sorted(set(inseason_ratings.keys()) | set(self.preseason_ratings.keys()))
        blended = {}

        for team in all_teams:
            preseason = self.get_preseason_rating(team)
            inseason = inseason_ratings.get(team, 0.0)

            # Get normalized talent for floor
            talent_rating = 0.0
            if team in self.preseason_ratings:
                talent_rating = self.preseason_ratings[team].talent_rating_normalized

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
        """Calculate decayed talent floor weight.

        Same formula as PreseasonPriors.calculate_decayed_talent_weight().
        """
        if games_played <= 0:
            return w_base
        decay_rate = (w_base - w_min) / target_week
        return max(w_base - games_played * decay_rate, w_min)

    def _calculate_confidence(
        self,
        rp: float,
        talent: float,
        prior_rating: float,
    ) -> float:
        """Calculate confidence score for a preseason rating.

        Higher confidence when:
        - High returning production (stable roster)
        - Strong prior year rating (more data)
        - Talent data available

        Args:
            rp: Returning production percentage
            talent: Talent composite score
            prior_rating: Prior year JP+ rating

        Returns:
            Confidence score 0-1
        """
        # RP contributes most (0-0.5)
        rp_conf = min(rp, 1.0) * 0.5

        # Prior rating stability (0-0.3)
        # Teams with extreme ratings are more predictable
        rating_conf = min(abs(prior_rating) / 30.0, 1.0) * 0.3

        # Talent data (0-0.2)
        talent_conf = 0.2 if talent > 0 else 0.0

        return rp_conf + rating_conf + talent_conf

    def _fetch_talent(self, year: int) -> dict[str, float]:
        """Fetch talent composite scores for a year."""
        try:
            talent = self.client.get_team_talent(year=year)
            scores = {t.team: t.talent for t in talent}
            logger.debug(f"Fetched talent for {len(scores)} teams")
            return scores
        except Exception as e:
            logger.warning(f"Could not fetch talent for {year}: {e}")
            return {}

    def _fetch_returning_production(self, year: int) -> dict[str, float]:
        """Fetch returning production for a year."""
        try:
            rp_data = self.client.get_returning_production(year=year)
            rp = {t.team: t.percent_ppa for t in rp_data if t.percent_ppa is not None}
            logger.debug(f"Fetched RP for {len(rp)} teams")
            return rp
        except Exception as e:
            logger.warning(f"Could not fetch RP for {year}: {e}")
            return {}

    def _fetch_conferences(self, year: int) -> dict[str, str]:
        """Fetch team conference mappings for a year."""
        try:
            teams = self.client.get_fbs_teams(year=year)
            return {t.school: t.conference for t in teams}
        except Exception as e:
            logger.warning(f"Could not fetch conferences for {year}: {e}")
            return {}

    def get_ratings_dataframe(self):
        """Get preseason ratings as DataFrame (for compatibility)."""
        import pandas as pd

        if not self.preseason_ratings:
            return pd.DataFrame()

        rows = []
        for team, rating in self.preseason_ratings.items():
            rows.append({
                "team": team,
                "combined_rating": rating.combined_rating,
                "prior_jp_rating": rating.prior_jp_rating,
                "talent_rating": rating.talent_rating,
                "returning_ppa": rating.returning_ppa,
                "confidence": rating.confidence,
                "tier": rating.tier,
            })

        return pd.DataFrame(rows).sort_values("combined_rating", ascending=False)
