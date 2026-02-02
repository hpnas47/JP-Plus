"""Spread generator combining all model components."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.models.special_teams import SpecialTeamsModel
from src.models.finishing_drives import FinishingDrivesModel
# Legacy models
from src.models.legacy.ridge_model import RidgeRatingsModel
from src.models.legacy.luck_regression import LuckRegressor
from src.models.legacy.early_down_model import EarlyDownModel
from src.adjustments.home_field import HomeFieldAdvantage
from src.adjustments.situational import SituationalAdjuster
from src.adjustments.travel import TravelAdjuster
from src.adjustments.altitude import AltitudeAdjuster

logger = logging.getLogger(__name__)

# Elite FCS teams (based on 2022-2024 performance vs FBS)
# These teams average +2 to +15 margin vs FBS (compared to +30 for average FCS)
# Criteria: avg margin < 20 with multiple FBS games, or FCS playoff regulars
# Triple-option / slow-pace teams that require spread compression
# These teams have ~30% worse MAE due to fewer possessions per game
# Spreads should be compressed toward 0 to account for reduced game volume
TRIPLE_OPTION_TEAMS = frozenset({
    "Army",
    "Navy",
    "Air Force",
    "Kennesaw State",
})

ELITE_FCS_TEAMS = frozenset({
    # Top performers vs FBS (data-driven)
    "Sacramento State",
    "Idaho",
    "Incarnate Word",
    "North Dakota State",
    "William & Mary",
    "Southern Illinois",
    "Holy Cross",
    "Weber State",
    "Fordham",
    "Monmouth",
    "South Dakota State",
    "Montana State",
    "Montana",
    # Traditional FCS powers (playoff regulars)
    "James Madison",  # Now FBS but was elite FCS
    "Sam Houston",  # Now FBS but was elite FCS
    "Villanova",
    "UC Davis",
    "Eastern Washington",
    "Northern Iowa",
    "Delaware",
    "Richmond",
    "Furman",  # 2022 was outlier, generally competitive
})


@dataclass
class SpreadComponents:
    """Breakdown of spread components."""

    base_margin: float = 0.0  # From ridge model
    home_field: float = 0.0
    situational: float = 0.0
    travel: float = 0.0
    altitude: float = 0.0
    special_teams: float = 0.0
    finishing_drives: float = 0.0
    early_down: float = 0.0
    luck_adjustment: float = 0.0
    fcs_adjustment: float = 0.0  # Penalty when FBS plays FCS
    pace_adjustment: float = 0.0  # Compression for triple-option teams


@dataclass
class PredictedSpread:
    """Container for a predicted spread with full breakdown."""

    home_team: str
    away_team: str
    spread: float  # Positive = home team favored
    home_win_probability: float
    components: SpreadComponents = field(default_factory=SpreadComponents)
    confidence: str = "Medium"  # Low, Medium, High

    @property
    def favorite(self) -> str:
        """Get the favored team."""
        return self.home_team if self.spread > 0 else self.away_team

    @property
    def underdog(self) -> str:
        """Get the underdog team."""
        return self.away_team if self.spread > 0 else self.home_team

    @property
    def spread_vs_favorite(self) -> float:
        """Get spread from favorite's perspective (always negative)."""
        return -abs(self.spread)

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "spread": self.spread,
            "favorite": self.favorite,
            "spread_vs_favorite": self.spread_vs_favorite,
            "home_win_prob": self.home_win_probability,
            "confidence": self.confidence,
            "base_margin": self.components.base_margin,
            "hfa": self.components.home_field,
            "situational": self.components.situational,
            "travel": self.components.travel,
            "altitude": self.components.altitude,
            "special_teams": self.components.special_teams,
            "finishing_drives": self.components.finishing_drives,
            "early_down": self.components.early_down,
            "luck_adj": self.components.luck_adjustment,
            "fcs_adj": self.components.fcs_adjustment,
            "pace_adj": self.components.pace_adjustment,
        }


class SpreadGenerator:
    """
    Generate predicted spreads by combining all model components.

    Formula:
    spread = base_margin + hfa + situational + travel + altitude +
             special_teams_diff + finishing_drives_diff + luck_adjustment +
             fcs_adjustment
    """

    # Tiered FCS penalties based on backtest analysis (2022-2024)
    # Mean FBS margin vs FCS: 30.3 points
    # Elite FCS teams average +5 to +15 margin, regular FCS average +35+
    DEFAULT_FCS_PENALTY_ELITE = 18.0  # Elite FCS (playoff regulars, consistent vs FBS)
    DEFAULT_FCS_PENALTY_STANDARD = 32.0  # Standard FCS teams

    def __init__(
        self,
        ridge_model: Optional[RidgeRatingsModel] = None,
        luck_regressor: Optional[LuckRegressor] = None,
        special_teams: Optional[SpecialTeamsModel] = None,
        finishing_drives: Optional[FinishingDrivesModel] = None,
        early_down: Optional[EarlyDownModel] = None,
        home_field: Optional[HomeFieldAdvantage] = None,
        situational: Optional[SituationalAdjuster] = None,
        travel: Optional[TravelAdjuster] = None,
        altitude: Optional[AltitudeAdjuster] = None,
        fbs_teams: Optional[set[str]] = None,
        fcs_penalty: Optional[float] = None,
        fcs_penalty_elite: Optional[float] = None,
        fcs_penalty_standard: Optional[float] = None,
        elite_fcs_teams: Optional[set[str]] = None,
    ):
        """Initialize spread generator with model components.

        Args:
            ridge_model: Core ratings model
            luck_regressor: Luck regression model
            special_teams: Special teams model
            finishing_drives: Finishing drives model
            early_down: Early-down success rate model
            home_field: Home field advantage calculator
            situational: Situational adjuster
            travel: Travel adjuster
            altitude: Altitude adjuster
            fbs_teams: Set of FBS team names for FCS detection
            fcs_penalty: DEPRECATED - use fcs_penalty_elite/standard instead
            fcs_penalty_elite: Points for elite FCS teams (default: 18.0)
            fcs_penalty_standard: Points for standard FCS teams (default: 32.0)
            elite_fcs_teams: Set of elite FCS team names (default: ELITE_FCS_TEAMS)
        """
        self.ridge_model = ridge_model or RidgeRatingsModel()
        self.luck_regressor = luck_regressor or LuckRegressor()
        self.special_teams = special_teams or SpecialTeamsModel()
        self.finishing_drives = finishing_drives or FinishingDrivesModel()
        self.early_down = early_down or EarlyDownModel()
        self.home_field = home_field or HomeFieldAdvantage()
        self.situational = situational or SituationalAdjuster()
        self.travel = travel or TravelAdjuster()
        self.altitude = altitude or AltitudeAdjuster()
        self.fbs_teams = fbs_teams or set()
        self.elite_fcs_teams = elite_fcs_teams if elite_fcs_teams is not None else ELITE_FCS_TEAMS

        # Support legacy single fcs_penalty parameter
        if fcs_penalty is not None:
            # Legacy mode: use single penalty for all FCS
            self.fcs_penalty_elite = fcs_penalty
            self.fcs_penalty_standard = fcs_penalty
        else:
            # Tiered mode (default)
            self.fcs_penalty_elite = fcs_penalty_elite if fcs_penalty_elite is not None else self.DEFAULT_FCS_PENALTY_ELITE
            self.fcs_penalty_standard = fcs_penalty_standard if fcs_penalty_standard is not None else self.DEFAULT_FCS_PENALTY_STANDARD

    def _get_pace_adjustment(
        self, home_team: str, away_team: str, spread: float
    ) -> float:
        """Calculate pace adjustment for triple-option teams.

        Triple-option teams run ~30% fewer plays per game, creating more variance
        and consistently worse MAE (16.09 vs 12.36 for standard teams). To account
        for this reduced game volume, we compress spreads toward 0 when a triple-
        option team is involved.

        Args:
            home_team: Home team name
            away_team: Away team name
            spread: Current spread before pace adjustment

        Returns:
            Pace adjustment (added to spread to compress it toward 0)
        """
        # Check if either team runs triple-option
        home_is_triple = home_team in TRIPLE_OPTION_TEAMS
        away_is_triple = away_team in TRIPLE_OPTION_TEAMS

        if not home_is_triple and not away_is_triple:
            return 0.0

        # Compression factor: reduce spread magnitude by 10%
        # This accounts for fewer possessions = more variance
        compression = 0.10

        # If both teams are triple-option, apply double compression
        if home_is_triple and away_is_triple:
            compression = 0.15

        # Adjustment moves spread toward 0
        # If spread is positive (home favored), adjustment is negative
        # If spread is negative (away favored), adjustment is positive
        return -spread * compression

    def _get_fcs_adjustment(self, home_team: str, away_team: str) -> float:
        """Calculate FCS adjustment for the matchup.

        When an FBS team plays an FCS team, apply a tiered penalty in favor of
        the FBS team. Elite FCS teams (playoff regulars, consistent vs FBS)
        get a smaller penalty than standard FCS teams.

        Args:
            home_team: Home team name
            away_team: Away team name

        Returns:
            FCS adjustment (positive = favors home, negative = favors away)
        """
        if not self.fbs_teams:
            return 0.0

        home_is_fbs = home_team in self.fbs_teams
        away_is_fbs = away_team in self.fbs_teams

        if home_is_fbs and not away_is_fbs:
            # Home is FBS, away is FCS - add penalty to home's favor
            penalty = self.fcs_penalty_elite if away_team in self.elite_fcs_teams else self.fcs_penalty_standard
            return penalty
        elif away_is_fbs and not home_is_fbs:
            # Away is FBS, home is FCS - subtract penalty (favor away)
            penalty = self.fcs_penalty_elite if home_team in self.elite_fcs_teams else self.fcs_penalty_standard
            return -penalty
        else:
            # Both FBS or both non-FBS - no adjustment
            return 0.0

    def _spread_to_probability(self, spread: float) -> float:
        """Convert spread to home win probability.

        Uses empirical relationship between spread and win probability.

        Args:
            spread: Point spread (positive = home favored)

        Returns:
            Home team win probability (0-1)
        """
        # Logistic approximation: each point of spread ~ 3% win probability
        # Calibrated to historical data
        import math

        # Spread of 0 = 50% (with small home edge already in spread)
        # k parameter controls steepness (higher = more confident predictions)
        k = 0.15

        prob = 1 / (1 + math.exp(-k * spread))
        return prob

    def _determine_confidence(
        self,
        spread: float,
        components: SpreadComponents,
    ) -> str:
        """Determine confidence level for a prediction.

        Args:
            spread: Predicted spread
            components: Spread components

        Returns:
            Confidence level: "Low", "Medium", or "High"
        """
        # Factors that reduce confidence:
        # - Close game (small spread)
        # - Large situational adjustments
        # - Large luck adjustments

        abs_spread = abs(spread)
        abs_situational = abs(components.situational)
        abs_luck = abs(components.luck_adjustment)

        # Score confidence factors
        score = 0

        if abs_spread >= 14:
            score += 2
        elif abs_spread >= 7:
            score += 1

        if abs_situational <= 1:
            score += 1

        if abs_luck <= 2:
            score += 1

        if score >= 3:
            return "High"
        elif score >= 2:
            return "Medium"
        return "Low"

    def predict_spread(
        self,
        home_team: str,
        away_team: str,
        week: Optional[int] = None,
        schedule_df: Optional[pd.DataFrame] = None,
        rankings: Optional[dict[str, int]] = None,
        neutral_site: bool = False,
    ) -> PredictedSpread:
        """Generate predicted spread for a matchup.

        Args:
            home_team: Home team name
            away_team: Away team name
            week: Current week number (for situational adjustments)
            schedule_df: Schedule DataFrame (for situational adjustments)
            rankings: Team rankings (for situational adjustments)
            neutral_site: Whether game is at neutral site

        Returns:
            PredictedSpread with full component breakdown
        """
        components = SpreadComponents()

        # Base margin from ridge model
        components.base_margin = self.ridge_model.predict_margin(
            home_team, away_team, neutral_site=True  # Get raw margin without HFA
        )

        # Home field advantage
        if not neutral_site:
            components.home_field = self.home_field.get_hfa(home_team)
        else:
            components.home_field = 0.0

        # Situational adjustments
        if week is not None and schedule_df is not None:
            # Determine who's favored for rivalry boost
            prelim_spread = components.base_margin + components.home_field
            home_is_favorite = prelim_spread < 0

            adj, _ = self.situational.get_matchup_adjustment(
                home_team=home_team,
                away_team=away_team,
                current_week=week,
                schedule_df=schedule_df,
                rankings=rankings,
                home_is_favorite=home_is_favorite,
            )
            components.situational = adj

        # Travel adjustment
        travel_adj, _ = self.travel.get_total_travel_adjustment(home_team, away_team)
        components.travel = travel_adj

        # Altitude adjustment
        alt_adj, _ = self.altitude.get_detailed_adjustment(home_team, away_team)
        components.altitude = alt_adj

        # Special teams differential
        components.special_teams = self.special_teams.get_matchup_differential(
            home_team, away_team
        )

        # Finishing drives differential
        components.finishing_drives = self.finishing_drives.get_matchup_differential(
            home_team, away_team
        )

        # Early-down success rate differential
        components.early_down = self.early_down.get_matchup_differential(
            home_team, away_team
        )

        # Luck adjustments
        home_luck = self.luck_regressor.get_luck_adjustment(home_team)
        away_luck = self.luck_regressor.get_luck_adjustment(away_team)
        components.luck_adjustment = home_luck - away_luck

        # FCS adjustment (when FBS plays FCS)
        components.fcs_adjustment = self._get_fcs_adjustment(home_team, away_team)

        # Calculate total spread (positive = home favored, internal convention)
        # Note: Standard Vegas convention is negative = home favored
        # We keep positive = home favored internally and negate when comparing to Vegas
        spread = (
            components.base_margin
            + components.home_field
            + components.situational
            + components.travel
            + components.altitude
            + components.special_teams
            + components.finishing_drives
            + components.early_down
            + components.luck_adjustment
            + components.fcs_adjustment
        )

        # Pace adjustment for triple-option teams (compress toward 0)
        components.pace_adjustment = self._get_pace_adjustment(
            home_team, away_team, spread
        )
        spread += components.pace_adjustment

        # Convert to win probability
        win_prob = self._spread_to_probability(spread)

        # Determine confidence
        confidence = self._determine_confidence(spread, components)

        return PredictedSpread(
            home_team=home_team,
            away_team=away_team,
            spread=round(spread, 1),
            home_win_probability=round(win_prob, 3),
            components=components,
            confidence=confidence,
        )

    def predict_week(
        self,
        games: list[dict],
        week: int,
        schedule_df: Optional[pd.DataFrame] = None,
        rankings: Optional[dict[str, int]] = None,
    ) -> list[PredictedSpread]:
        """Generate predictions for a full week of games.

        Args:
            games: List of game dicts with 'home_team', 'away_team', 'neutral_site'
            week: Week number
            schedule_df: Full season schedule
            rankings: Team rankings

        Returns:
            List of PredictedSpread objects
        """
        predictions = []

        for game in games:
            pred = self.predict_spread(
                home_team=game["home_team"],
                away_team=game["away_team"],
                week=week,
                schedule_df=schedule_df,
                rankings=rankings,
                neutral_site=game.get("neutral_site", False),
            )
            predictions.append(pred)

        return predictions

    def predictions_to_dataframe(
        self, predictions: list[PredictedSpread]
    ) -> pd.DataFrame:
        """Convert predictions to DataFrame.

        Args:
            predictions: List of predictions

        Returns:
            DataFrame with all prediction data
        """
        data = [p.to_dict() for p in predictions]
        df = pd.DataFrame(data)

        # Sort by absolute spread (biggest mismatches first)
        df = df.reindex(
            df["spread"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)

        return df
