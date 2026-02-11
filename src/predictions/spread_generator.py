"""Spread generator combining EFM ratings with adjustment layers."""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from src.models.special_teams import SpecialTeamsModel
from src.models.finishing_drives import FinishingDrivesModel
from src.models.fcs_strength import FCSStrengthEstimator
from src.adjustments.home_field import HomeFieldAdvantage
from src.adjustments.situational import (
    SituationalAdjuster,
    SituationalFactors,
    HistoricalRankings,
)
from src.adjustments.travel import TravelAdjuster
from config.teams import TRIPLE_OPTION_TEAMS
from src.adjustments.altitude import AltitudeAdjuster
from src.adjustments.qb_adjustment import QBInjuryAdjuster
from src.adjustments.qb_continuous import QBContinuousAdjuster
from src.adjustments.aggregator import AdjustmentAggregator, TravelBreakdown
from src.adjustments.diagnostics import (
    AdjustmentStack,
    AdjustmentStackDiagnostics,
    extract_stack_from_prediction,
)

logger = logging.getLogger(__name__)


# TRIPLE_OPTION_TEAMS imported from config.teams (single source of truth)

# Elite FCS teams (based on 2022-2024 performance vs FBS)
# These teams average +2 to +15 margin vs FBS (compared to +30 for average FCS)
# Criteria: avg margin < 20 with multiple FBS games, or FCS playoff regulars
# Note: James Madison (2022) and Sam Houston (2023) graduated to FBS
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
    "Villanova",
    "UC Davis",
    "Eastern Washington",
    "Northern Iowa",
    "Delaware",
    "Richmond",
    "Furman",
})


@dataclass
class SpreadComponents:
    """Breakdown of spread components."""

    base_margin: float = 0.0  # From EFM ratings differential
    home_field: float = 0.0
    situational: float = 0.0
    travel: float = 0.0
    altitude: float = 0.0
    special_teams: float = 0.0
    finishing_drives: float = 0.0
    fcs_adjustment: float = 0.0  # Penalty when FBS plays FCS
    pace_adjustment: float = 0.0  # Compression for triple-option teams
    qb_adjustment: float = 0.0  # Adjustment when starting QB is out
    env_score: float = 0.0  # Post-smoothing environmental stack (from aggregator)
    # Raw values before caps/offsets (for dual-cap mode spread reassembly)
    special_teams_raw: float = 0.0  # Raw ST differential before cap
    home_field_raw: float = 0.0  # Raw HFA before global offset

    @property
    def correlated_stack(self) -> float:
        """Sum of venue-related adjustments (HFA + travel + altitude).

        These are POST-smoothing values (pro-rata allocation from env soft cap).
        They represent what actually hit the spread for these three components.
        """
        return self.home_field + self.travel + self.altitude

    @property
    def full_env_stack(self) -> float:
        """Full environmental stack after soft cap (HFA + travel + altitude + rest + consec).

        This is env_score from the aggregator. Use this for:
        - _determine_confidence() decisions
        - Extreme-stack warnings
        - Any logic that should reflect total environmental impact
        """
        return self.env_score


@dataclass
class PredictedSpread:
    """Container for a predicted spread with full breakdown.

    Note: spread and home_win_probability are stored with full precision internally.
    Use spread_display and win_prob_display for rounded presentation values.
    Use to_dict() for DataFrame output (which rounds automatically).
    """

    home_team: str
    away_team: str
    spread: float  # Positive = home team favored (FULL PRECISION)
    home_win_probability: float  # FULL PRECISION
    components: SpreadComponents = field(default_factory=SpreadComponents)
    confidence: str = "Medium"  # Low, Medium, High
    game_id: Optional[int] = None  # P0.1: For reliable Vegas line matching

    @property
    def spread_display(self) -> float:
        """Get spread rounded to 0.5 for display (standard betting increment)."""
        return round(self.spread * 2) / 2

    @property
    def win_prob_display(self) -> float:
        """Get win probability rounded for display."""
        return round(self.home_win_probability, 3)

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
        """Convert to dictionary for DataFrame creation.

        Note: Spreads are rounded to 0.5 for display/reporting.
        Use self.spread directly for full precision (e.g., MAE calculation).
        """
        return {
            "game_id": self.game_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "spread": round(self.spread * 2) / 2,  # Round to 0.5 for display
            "spread_raw": self.spread,  # Full precision for analysis
            "favorite": self.favorite,
            "spread_vs_favorite": round(self.spread_vs_favorite * 2) / 2,
            "home_win_prob": round(self.home_win_probability, 3),
            "confidence": self.confidence,
            "base_margin": self.components.base_margin,
            "hfa": self.components.home_field,
            "situational": self.components.situational,
            "travel": self.components.travel,
            "altitude": self.components.altitude,
            "correlated_stack": self.components.correlated_stack,  # HFA+travel+altitude (post-smoothing)
            "full_env_stack": self.components.full_env_stack,  # Full env stack (includes rest, consec)
            "special_teams": self.components.special_teams,
            "finishing_drives": self.components.finishing_drives,
            "fcs_adj": self.components.fcs_adjustment,
            "pace_adj": self.components.pace_adjustment,
            "qb_adj": self.components.qb_adjustment,
        }


class SpreadGenerator:
    """
    Generate predicted spreads by combining EFM ratings with adjustment layers.

    Formula:
    spread = base_margin + hfa + situational + travel + altitude +
             special_teams_diff + finishing_drives_diff + fcs_adjustment

    Base margin comes from EFM overall ratings differential (home - away).
    All other components are adjustment layers applied on top.

    Special Teams Integration (P2.7):
    -----------------------------
    Special teams is applied as a SEPARATE ADJUSTMENT LAYER here, not as part
    of the base ratings. This follows SP+ methodology where ST is a game-level
    differential rather than embedded in team ratings.

    - SpecialTeamsModel.get_matchup_differential() provides ST point differential
    - EFM's special_teams_rating is DIAGNOSTIC ONLY (not in EFM overall_rating)
    - This prevents double-counting: ST is added once, here in SpreadGenerator

    If using a ratings model that already includes ST in overall, set
    special_teams=None to disable the adjustment layer here.
    """

    # Tiered FCS penalties based on backtest analysis (2022-2024)
    # Mean FBS margin vs FCS: 30.3 points
    # Elite FCS teams average +5 to +15 margin, regular FCS average +35+
    DEFAULT_FCS_PENALTY_ELITE = 18.0  # Elite FCS (playoff regulars, consistent vs FBS)
    DEFAULT_FCS_PENALTY_STANDARD = 32.0  # Standard FCS teams

    # Logistic curve steepness for spread-to-probability conversion
    # k=0.15 means each point of spread ≈ 3% change in win probability
    # Calibrated to historical FBS game outcomes
    WIN_PROB_K = 0.15

    def __init__(
        self,
        ratings: Optional[dict[str, float]] = None,
        special_teams: Optional[SpecialTeamsModel] = None,
        finishing_drives: Optional[FinishingDrivesModel] = None,
        home_field: Optional[HomeFieldAdvantage] = None,
        situational: Optional[SituationalAdjuster] = None,
        travel: Optional[TravelAdjuster] = None,
        altitude: Optional[AltitudeAdjuster] = None,
        fbs_teams: Optional[set[str]] = None,
        fcs_penalty_elite: Optional[float] = None,
        fcs_penalty_standard: Optional[float] = None,
        elite_fcs_teams: Optional[set[str]] = None,
        qb_adjuster: Optional[QBInjuryAdjuster] = None,
        qb_continuous: Optional[QBContinuousAdjuster] = None,
        aggregator: Optional[AdjustmentAggregator] = None,
        track_diagnostics: bool = False,
        global_cap: float = 7.0,
        st_spread_cap: Optional[float] = 2.5,
        st_early_season_weight: float = 1.0,  # Weight for ST in weeks 1-3 (1.0 = full, 0.5 = half)
        fcs_estimator: Optional[FCSStrengthEstimator] = None,  # Dynamic FCS strength estimator
    ):
        """Initialize spread generator with EFM ratings and adjustment layers.

        Args:
            ratings: Dict mapping team names to EFM overall ratings
            special_teams: Special teams model for FG/punt/kickoff differential
            finishing_drives: Finishing drives model for red zone efficiency
            home_field: Home field advantage calculator
            situational: Situational adjuster (bye weeks, letdown, lookahead, rivalry)
            travel: Travel adjuster (timezone, distance)
            altitude: Altitude adjuster (high elevation venues)
            fbs_teams: Set of FBS team names for FCS detection
            fcs_penalty_elite: Points for elite FCS teams (default: 18.0, used if no fcs_estimator)
            fcs_penalty_standard: Points for standard FCS teams (default: 32.0, used if no fcs_estimator)
            elite_fcs_teams: Set of elite FCS team names (default: ELITE_FCS_TEAMS, used if no fcs_estimator)
            qb_adjuster: QB injury adjuster (optional, for manual QB-out adjustments)
            qb_continuous: Continuous QB rating adjuster (optional, for always-on QB quality)
            aggregator: AdjustmentAggregator for consolidated smoothing (default: creates new)
            track_diagnostics: If True, track adjustment stacks for P2.11 analysis
            global_cap: Maximum total adjustment (default: 7.0)
            st_spread_cap: Cap on ST differential's effect on spread. Default 2.5 (APPROVED). 0/None = no cap.
            st_early_season_weight: Weight for ST impact in weeks 1-3 (1.0 = full). Default 1.0.
            fcs_estimator: Dynamic FCS strength estimator (if provided, overrides static elite list)
        """
        self.ratings = ratings or {}
        # Calculate mean rating for fallback on missing teams
        # If ratings center around +2.0, a missing team should be treated as +2.0, not 0.0
        if self.ratings:
            self._mean_rating = sum(self.ratings.values()) / len(self.ratings)
        else:
            self._mean_rating = 0.0
        self.special_teams = special_teams or SpecialTeamsModel()
        self.finishing_drives = finishing_drives or FinishingDrivesModel()
        self.home_field = home_field or HomeFieldAdvantage()
        self.situational = situational or SituationalAdjuster()
        self.travel = travel or TravelAdjuster()
        self.altitude = altitude or AltitudeAdjuster()
        self.fbs_teams = fbs_teams or set()
        self.elite_fcs_teams = elite_fcs_teams if elite_fcs_teams is not None else ELITE_FCS_TEAMS

        # FCS penalties
        self.fcs_penalty_elite = fcs_penalty_elite if fcs_penalty_elite is not None else self.DEFAULT_FCS_PENALTY_ELITE
        self.fcs_penalty_standard = fcs_penalty_standard if fcs_penalty_standard is not None else self.DEFAULT_FCS_PENALTY_STANDARD

        # QB adjusters (optional)
        # qb_adjuster: Manual injury override (--qb-out flag)
        # qb_continuous: Always-on continuous QB rating system
        # Note: If both are provided, manual override takes precedence for flagged teams
        self.qb_adjuster = qb_adjuster
        self.qb_continuous = qb_continuous

        # Consolidated adjustment aggregator (four-bucket smoothing)
        self.aggregator = aggregator or AdjustmentAggregator(global_cap=global_cap)

        # P2.11: Adjustment stack diagnostics
        self.track_diagnostics = track_diagnostics
        self.diagnostics = AdjustmentStackDiagnostics() if track_diagnostics else None

        # ST spread cap (Approach B: margin-level capping)
        # Caps the effect of ST differential on final spread, not the rating itself
        self.st_spread_cap = st_spread_cap
        self.st_early_season_weight = st_early_season_weight

        # Dynamic FCS strength estimator (overrides static elite list if provided)
        self.fcs_estimator = fcs_estimator

    def _get_base_margin(self, home_team: str, away_team: str) -> float:
        """Get base margin from EFM ratings differential.

        Args:
            home_team: Home team name
            away_team: Away team name

        Returns:
            Expected point margin (home - away) on neutral field
        """
        home_rating = self.ratings.get(home_team)
        away_rating = self.ratings.get(away_team)

        # Use mean rating as fallback for missing teams (not 0.0)
        # If ratings center around +2.0, a missing team should be treated as
        # "average" (+2.0), not "exactly zero". This prevents systematic bias
        # when team names don't match between ratings dict and game data.
        if home_rating is None:
            logger.warning(
                f"Team '{home_team}' not found in ratings "
                f"(defaulting to mean={self._mean_rating:.2f}). "
                "Check for name mismatch or missing data."
            )
            home_rating = self._mean_rating
        if away_rating is None:
            logger.warning(
                f"Team '{away_team}' not found in ratings "
                f"(defaulting to mean={self._mean_rating:.2f}). "
                "Check for name mismatch or missing data."
            )
            away_rating = self._mean_rating

        return home_rating - away_rating

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
        home_is_triple = home_team in TRIPLE_OPTION_TEAMS
        away_is_triple = away_team in TRIPLE_OPTION_TEAMS

        if not home_is_triple and not away_is_triple:
            return 0.0

        # Compression factor: reduce spread magnitude by 10%
        compression = 0.10

        # If both teams are triple-option, apply double compression
        if home_is_triple and away_is_triple:
            compression = 0.15

        # Adjustment moves spread toward 0
        return -spread * compression

    def _get_fcs_adjustment(self, home_team: str, away_team: str) -> float:
        """Calculate FCS adjustment for the matchup.

        When an FBS team plays an FCS team, apply a penalty in favor of the FBS team.
        Uses dynamic FCS estimator if available (data-driven, walk-forward safe),
        otherwise falls back to static elite list with tiered penalties.

        Note: FCS teams may appear in self.ratings if they played FBS opponents,
        but they should still receive the FCS penalty. The ratings dict includes
        all teams in the play data, not just FBS.

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
            fcs_team = away_team
            if self.fcs_estimator is not None:
                # Dynamic penalty from FCS strength estimator
                penalty = self.fcs_estimator.get_penalty(fcs_team)
                strength = self.fcs_estimator.get_strength(fcs_team)
                # avg_loss = -shrunk_margin (positive = FCS loses, negative = FCS wins)
                # Consistent with _margin_to_penalty() in FCS estimator
                avg_loss = -strength.shrunk_margin if strength else -self.fcs_estimator.baseline_margin
                logger.debug(f"[FCS Scale] Team: {fcs_team}, AvgLoss: {avg_loss:.1f}, FinalBonus: {penalty:.1f}")
            else:
                # Fallback to static tiered system
                penalty = self.fcs_penalty_elite if fcs_team in self.elite_fcs_teams else self.fcs_penalty_standard
            return penalty
        elif away_is_fbs and not home_is_fbs:
            # Away is FBS, home is FCS - subtract penalty (favor away)
            fcs_team = home_team
            if self.fcs_estimator is not None:
                # Dynamic penalty from FCS strength estimator
                penalty = self.fcs_estimator.get_penalty(fcs_team)
                strength = self.fcs_estimator.get_strength(fcs_team)
                # avg_loss = -shrunk_margin (positive = FCS loses, negative = FCS wins)
                avg_loss = -strength.shrunk_margin if strength else -self.fcs_estimator.baseline_margin
                logger.debug(f"[FCS Scale] Team: {fcs_team}, AvgLoss: {avg_loss:.1f}, FinalBonus: {penalty:.1f}")
            else:
                # Fallback to static tiered system
                penalty = self.fcs_penalty_elite if fcs_team in self.elite_fcs_teams else self.fcs_penalty_standard
            return -penalty
        else:
            # Both FBS or both non-FBS - no adjustment
            return 0.0

    def _spread_to_probability(self, spread: float) -> float:
        """Convert spread to home win probability.

        Uses logistic function calibrated to historical FBS outcomes.
        WIN_PROB_K controls curve steepness (0.15 ≈ 3% per point).

        Args:
            spread: Point spread (positive = home favored)

        Returns:
            Home team win probability (0-1)
        """
        prob = 1 / (1 + math.exp(-self.WIN_PROB_K * spread))
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
        abs_spread = abs(spread)
        abs_situational = abs(components.situational)

        # Score confidence factors
        score = 0

        if abs_spread >= 14:
            score += 2
        elif abs_spread >= 7:
            score += 1

        if abs_situational <= 1:
            score += 1

        # Large correlated stacks reduce confidence (use smoothed, not raw)
        if abs(components.full_env_stack) <= 4:
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
        historical_rankings: Optional[HistoricalRankings] = None,
        game_date: Optional[datetime] = None,
        game_id: Optional[int] = None,
    ) -> PredictedSpread:
        """Generate predicted spread for a matchup.

        Uses consolidated four-bucket smoothing via AdjustmentAggregator:
        - Bucket A (Venue): HFA - no smoothing
        - Bucket B (Physical): travel + altitude + consecutive_road + short_week
        - Bucket C (Mental): letdown + lookahead + sandwich
        - Bucket D (Boosts): rivalry + bye_week_rest

        Args:
            home_team: Home team name
            away_team: Away team name
            week: Current week number (for situational adjustments)
            schedule_df: Schedule DataFrame (for situational adjustments)
            rankings: Team rankings (current week snapshot, for situational adjustments)
            neutral_site: Whether game is at neutral site
            historical_rankings: Week-by-week historical rankings (for letdown spot detection)
            game_date: Date of current game (for rest day calculation)
            game_id: CFBD game ID (for reliable Vegas line matching)

        Returns:
            PredictedSpread with full component breakdown
        """
        components = SpreadComponents()

        # Base margin from EFM ratings (neutral field)
        components.base_margin = self._get_base_margin(home_team, away_team)

        # =====================================================================
        # Gather raw adjustments (no smoothing yet)
        # =====================================================================

        # Home field advantage (raw)
        # Capture raw HFA before global offset for dual-cap mode reassembly
        if not neutral_site:
            raw_hfa = self.home_field.get_hfa_value(home_team)
            # Get pre-offset value for dual-cap mode spread reassembly
            hfa_pre_offset, _ = self.home_field.get_hfa(home_team, skip_offset=True)
            components.home_field_raw = hfa_pre_offset
        else:
            raw_hfa = 0.0
            components.home_field_raw = 0.0

        # Travel and altitude (raw values only; breakdown dicts discarded)
        raw_travel, _travel_breakdown = self.travel.get_total_travel_adjustment(home_team, away_team)
        raw_altitude, _altitude_breakdown = self.altitude.get_detailed_adjustment(home_team, away_team)

        travel_breakdown = TravelBreakdown(
            travel_penalty=raw_travel,
            altitude_penalty=raw_altitude,
        )

        # Situational factors (raw)
        if week is not None and schedule_df is not None:
            # Determine who's favored for rivalry boost
            prelim_spread = components.base_margin + raw_hfa
            home_is_favorite = prelim_spread > 0

            home_factors, away_factors = self.situational.get_matchup_factors(
                home_team=home_team,
                away_team=away_team,
                current_week=week,
                schedule_df=schedule_df,
                rankings=rankings,
                home_is_favorite=home_is_favorite,
                historical_rankings=historical_rankings,
                game_date=game_date,
            )
        else:
            # No schedule data - use empty factors
            home_factors = SituationalFactors(team=home_team)
            away_factors = SituationalFactors(team=away_team)

        # =====================================================================
        # Aggregate with four-bucket smoothing
        # =====================================================================
        aggregated = self.aggregator.aggregate(
            raw_hfa=raw_hfa,
            travel_breakdown=travel_breakdown,
            home_factors=home_factors,
            away_factors=away_factors,
        )

        # Store components with pro-rata smoothing applied
        # The aggregator applies soft-cap to VENUE stack only (HFA + travel + altitude)
        # Rest and consecutive_road are added linearly, not smoothed
        factor = aggregated.venue_smoothing_factor
        components.home_field = aggregated.raw_hfa * factor
        components.travel = aggregated.raw_travel * factor
        components.altitude = aggregated.raw_altitude * factor
        components.env_score = aggregated.env_score  # Post-smoothing env total
        # Situational is now a clean value from aggregator (rest, consec, mental, boosts)
        # No longer polluted by smoothing delta residuals
        components.situational = aggregated.situational_score

        # Special teams differential (P2.7: applied as adjustment layer)
        raw_st_diff = self.special_teams.get_matchup_differential(
            home_team, away_team
        )
        # Store raw ST differential for dual-cap mode reassembly
        components.special_teams_raw = raw_st_diff
        # Apply margin-level cap if configured (Approach B)
        # This caps the EFFECT on spread, not the rating itself
        if self.st_spread_cap and self.st_spread_cap > 0:
            st_capped = max(-self.st_spread_cap, min(self.st_spread_cap, raw_st_diff))
        else:
            st_capped = raw_st_diff
        # Apply early-season weight (Approach C) - reduce ST impact weeks 1-3
        if week is not None and week <= 3 and self.st_early_season_weight < 1.0:
            components.special_teams = st_capped * self.st_early_season_weight
        else:
            components.special_teams = st_capped

        # Finishing drives differential (red zone efficiency)
        # SHELVED: 4 consecutive backtest rejections confirmed ~70-80% overlap with EFM (IsoPPP).
        # Infrastructure preserved: drive_id pipeline, per-game normalization, dynamic baseline, ±1.5pt cap.
        # Reactivation path: residualize against EFM or integrate RZ metric as EFM feature.
        components.finishing_drives = 0.0  # self.finishing_drives.get_matchup_differential(home_team, away_team)

        # Calculate efficiency-derived spread (positive = home favored)
        # Note: aggregated.net_adjustment already includes HFA+travel+altitude+situational
        # So we add the non-aggregated components
        spread = (
            components.base_margin
            + aggregated.net_adjustment  # All game-context adjustments (smoothed)
            + components.special_teams
            + components.finishing_drives
        )

        # QB adjustment
        # Priority order:
        # 1. qb_continuous (always-on continuous rating, handles manual overrides internally)
        # 2. qb_adjuster (legacy binary injury adjustment)
        # Applied before pace adjustment so triple-option compression
        # correctly dampens QB impact in low-possession games
        if self.qb_continuous:
            components.qb_adjustment = self.qb_continuous.get_adjustment(
                home_team, away_team, pred_week=week
            )
            spread += components.qb_adjustment
        elif self.qb_adjuster:
            components.qb_adjustment = self.qb_adjuster.get_adjustment(
                home_team, away_team
            )
            spread += components.qb_adjustment

        # Pace adjustment for triple-option teams (compress toward 0)
        # Applied to efficiency-derived spread only (before FCS penalty)
        # FCS penalty represents talent gap, not pace-driven quantity
        components.pace_adjustment = self._get_pace_adjustment(
            home_team, away_team, spread
        )
        spread += components.pace_adjustment

        # FCS adjustment (when FBS plays FCS)
        # Applied AFTER pace adjustment - talent gap shouldn't be compressed
        components.fcs_adjustment = self._get_fcs_adjustment(home_team, away_team)
        spread += components.fcs_adjustment

        # Convert to win probability
        win_prob = self._spread_to_probability(spread)

        # Determine confidence
        confidence = self._determine_confidence(spread, components)

        # Create prediction
        prediction = PredictedSpread(
            home_team=home_team,
            away_team=away_team,
            spread=spread,
            home_win_probability=win_prob,
            components=components,
            confidence=confidence,
            game_id=game_id,
        )

        # P2.11: Track adjustment stack for diagnostics
        if self.diagnostics is not None:
            stack = extract_stack_from_prediction(prediction)
            self.diagnostics.add_game(stack)
            if stack.is_extreme_stack:
                # Use smoothed stack for warning (what actually hit the spread)
                smoothed = components.full_env_stack
                raw = components.correlated_stack
                logger.warning(
                    f"Extreme adjustment stack (smoothed={smoothed:.1f}, raw={raw:.1f} pts): "
                    f"{away_team} @ {home_team} "
                    f"(HFA={components.home_field:.1f}, travel={components.travel:.1f}, "
                    f"alt={components.altitude:.1f})"
                )

        return prediction

    def predict_week(
        self,
        games: list[dict],
        week: int,
        schedule_df: Optional[pd.DataFrame] = None,
        rankings: Optional[dict[str, int]] = None,
        historical_rankings: Optional[HistoricalRankings] = None,
    ) -> list[PredictedSpread]:
        """Generate predictions for a full week of games.

        Args:
            games: List of game dicts with keys:
                - home_team (required): Home team name
                - away_team (required): Away team name
                - neutral_site (optional): Whether neutral site game (default False)
                - start_date (optional): Game datetime for rest day calculation
                - id or game_id (optional): CFBD game ID for Vegas line matching
            week: Week number
            schedule_df: Full season schedule (must have start_date for rest calculation)
            rankings: Team rankings (current week snapshot)
            historical_rankings: Week-by-week historical rankings (for letdown spot)

        Returns:
            List of PredictedSpread objects
        """
        predictions = []

        for game in games:
            # Extract game date for rest day calculation
            game_date = game.get("start_date")
            # Extract game_id for reliable Vegas line matching (P0.1)
            game_id = game.get("id") or game.get("game_id")

            pred = self.predict_spread(
                home_team=game["home_team"],
                away_team=game["away_team"],
                week=week,
                schedule_df=schedule_df,
                rankings=rankings,
                neutral_site=game.get("neutral_site", False),
                historical_rankings=historical_rankings,
                game_date=game_date,
                game_id=game_id,
            )
            predictions.append(pred)

        return predictions

    def predictions_to_dataframe(
        self,
        predictions: list[PredictedSpread],
        sort_by_spread: bool = True,
    ) -> pd.DataFrame:
        """Convert predictions to DataFrame.

        Args:
            predictions: List of predictions
            sort_by_spread: If True (default), sort by absolute spread descending
                           (biggest mismatches first). Set False to preserve input order.

        Returns:
            DataFrame with all prediction data
        """
        data = [p.to_dict() for p in predictions]
        df = pd.DataFrame(data)

        if sort_by_spread:
            # Sort by absolute spread (biggest mismatches first)
            df = df.reindex(
                df["spread"].abs().sort_values(ascending=False).index
            ).reset_index(drop=True)

        return df

    def log_stack_diagnostics(self) -> None:
        """Log adjustment stack diagnostics summary (P2.11)."""
        if self.diagnostics is None:
            logger.info("Diagnostics not enabled (set track_diagnostics=True)")
            return
        self.diagnostics.log_summary()

    def evaluate_stack_errors(
        self,
        predictions_df: pd.DataFrame,
        results_df: pd.DataFrame,
    ) -> dict:
        """Evaluate prediction errors for high-stack vs low-stack games (P2.11)."""
        if self.diagnostics is None:
            return {"error": "Diagnostics not enabled"}
        return self.diagnostics.evaluate_errors(predictions_df, results_df)

    def log_stack_error_analysis(
        self,
        predictions_df: pd.DataFrame,
        results_df: pd.DataFrame,
    ) -> None:
        """Log error analysis for high-stack vs low-stack games (P2.11)."""
        if self.diagnostics is None:
            logger.info("Diagnostics not enabled (set track_diagnostics=True)")
            return
        self.diagnostics.log_error_analysis(predictions_df, results_df)

    def get_high_stack_games(self) -> list[AdjustmentStack]:
        """Get games with high correlated stacks (>5 pts)."""
        if self.diagnostics is None:
            return []
        return self.diagnostics.get_high_stack_games()

    def reset_diagnostics(self) -> None:
        """Reset diagnostics for a new prediction run."""
        if self.diagnostics is not None:
            self.diagnostics = AdjustmentStackDiagnostics()
