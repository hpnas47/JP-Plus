"""Selection engine for calibrated spread betting.

This module provides:
1. BetRecommendation - full recommendation for a single game
2. evaluate_game - evaluate a single game for betting
3. evaluate_slate - evaluate all games in a slate

V2 Features:
- Push-aware EV calculation
- Confidence tier assignment (HIGH, MED, BET, PASS, NO_BET)
- Full metadata for auditability
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .calibration import (
    CalibrationResult,
    PushRates,
    breakeven_prob,
    predict_cover_probability,
    get_push_probability,
)

logger = logging.getLogger(__name__)

# Minimum edge threshold - edges below this are treated as zero (no meaningful edge).
# CFB spreads move in half-point increments; 0.05 pts is floating point noise.
MINIMUM_EDGE_PTS = 0.05


@dataclass
class BetRecommendation:
    """Full recommendation for a single game.

    All spreads in Vegas convention (negative = home favored).

    Attributes:
        game_id: Unique game identifier
        home_team: Home team name
        away_team: Away team name
        jp_spread: JP+ spread in Vegas convention
        vegas_spread: Vegas spread
        edge_pts: jp_spread - vegas_spread (negative = JP+ likes home more)
        edge_abs: Absolute edge in points
        jp_favored_side: "HOME" or "AWAY" based on edge direction

        p_cover_no_push: P(cover | no push) from logistic calibration
        p_push: P(push) from push rate lookup
        p_cover: Unconditional P(cover) = p_cover_no_push * (1 - p_push)
        p_breakeven: Breakeven probability for juice (e.g., 0.5238 at -110)
        edge_prob: Unconditional probability edge accounting for push: (p_cover_no_push - p_breakeven) * (1 - p_push)
        ev: Expected value as fraction of stake

        juice: American odds (e.g., -110)
        confidence: Tier: "HIGH", "MED", "BET", "PASS", "NO_BET"

        year: Optional year for tracking
        week: Optional week for tracking
    """

    game_id: str
    home_team: str
    away_team: str
    jp_spread: float
    vegas_spread: float
    edge_pts: float
    edge_abs: float
    jp_favored_side: str

    # Probabilities - None for NO_BET games
    p_cover_no_push: Optional[float]
    p_push: Optional[float]
    p_cover: Optional[float]
    p_breakeven: float
    edge_prob: Optional[float]
    ev: Optional[float]

    # Metadata
    juice: int
    confidence: str

    # Optional extras
    year: Optional[int] = None
    week: Optional[int] = None

    def __repr__(self) -> str:
        if self.confidence == "NO_BET":
            return (
                f"BetRecommendation({self.away_team}@{self.home_team}, "
                f"NO_BET (edge=0))"
            )
        return (
            f"BetRecommendation({self.away_team}@{self.home_team}, "
            f"{self.jp_favored_side} by {self.edge_abs:.1f}, "
            f"p_cover={self.p_cover:.1%}, EV={self.ev:+.3f}, {self.confidence})"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "game_id": self.game_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "jp_spread": self.jp_spread,
            "vegas_spread": self.vegas_spread,
            "edge_pts": self.edge_pts,
            "edge_abs": self.edge_abs,
            "jp_favored_side": self.jp_favored_side,
            "p_cover_no_push": self.p_cover_no_push,
            "p_push": self.p_push,
            "p_cover": self.p_cover,
            "p_breakeven": self.p_breakeven,
            "edge_prob": self.edge_prob,
            "ev": self.ev,
            "juice": self.juice,
            "confidence": self.confidence,
            "year": self.year,
            "week": self.week,
        }


def calculate_ev_with_push(
    p_cover_no_push: float,
    p_push: float,
    juice: int = -110,
) -> float:
    """Calculate EV with explicit push probability.

    Formula:
        p_win = p_cover_no_push * (1 - p_push)
        p_lose = (1 - p_cover_no_push) * (1 - p_push)
        p_push = p_push (stake returned, 0 profit)

        EV = p_win * payout - p_lose * stake + p_push * 0
           = p_win * payout - p_lose * 1

    Example (from spec):
        p_cover_no_push = 0.55
        p_push = 0.05
        juice = -110

        p_win = 0.55 * 0.95 = 0.5225
        p_lose = 0.45 * 0.95 = 0.4275
        p_push = 0.05

        EV = 0.5225 * (100/110) - 0.4275 * 1 + 0.05 * 0
           = 0.475 - 0.4275
           = 0.0475

    Args:
        p_cover_no_push: P(cover | no push) from calibration
        p_push: P(push) from push rate lookup
        juice: American odds (e.g., -110 or +150). Zero is invalid.

    Returns:
        EV as fraction of stake

    Raises:
        ValueError: If juice is zero (invalid American odds)
    """
    if juice == 0:
        raise ValueError("American odds cannot be zero")

    # American to payout conversion (profit per unit risked):
    # Negative odds (e.g., -110): risk 110 to win 100 -> payout = 100/110 = 0.909
    # Positive odds (e.g., +150): risk 100 to win 150 -> payout = 150/100 = 1.5
    if juice < 0:
        payout = 100 / abs(juice)
    else:
        payout = juice / 100

    p_win = p_cover_no_push * (1 - p_push)
    p_lose = (1 - p_cover_no_push) * (1 - p_push)

    ev = p_win * payout - p_lose * 1.0
    return ev


def evaluate_game(
    game: pd.Series,
    calibration: CalibrationResult,
    push_rates: Optional[PushRates] = None,
    min_ev_threshold: float = 0.03,
    juice: int = -110,
    high_ev_offset: float = 0.05,
    med_ev_offset: float = 0.02,
) -> BetRecommendation:
    """Evaluate a single game for betting recommendation.

    CRITICAL CALIBRATION ASSUMPTION:
        The calibration module trains on ABSOLUTE EDGE (edge_abs) with the convention
        that the bet is ALWAYS on the JP-favored side. The returned p_cover_no_push
        represents P(JP-favored side covers | edge_abs). This is intentional because:
        1. Edge direction is already encoded in jp_favored_side
        2. The logistic model only needs edge magnitude to estimate cover probability
        3. By always betting the JP-favored side, we convert a directional problem
           to a magnitude problem

        If calibration.p_cover_at_zero deviates significantly from 0.5, it may indicate
        training data issues or a miscalibrated model.

    Args:
        game: Series with jp_spread, vegas_spread, home_team, away_team, game_id
        calibration: CalibrationResult from calibrate_cover_probability
        push_rates: Optional PushRates for push modeling (V2)
        min_ev_threshold: Minimum EV for BET tier
        juice: American odds (default: -110)
        high_ev_offset: EV above min_ev_threshold for HIGH tier (default: 0.05)
        med_ev_offset: EV above min_ev_threshold for MED tier (default: 0.02)

    Returns:
        BetRecommendation with all fields populated

    Logic:
        1. Compute edge_pts, edge_abs, jp_favored_side
        2. If edge_abs < MINIMUM_EDGE_PTS: NO_BET (all prob/EV fields None)
        3. Determine recommended side from edge sign
        4. Compute p_cover_no_push from calibration (using edge_abs, not signed edge)
        5. Compute p_push from push_rates (0 if None)
        6. Compute p_cover = p_cover_no_push * (1 - p_push)
        7. Compute EV using push-aware formula
        8. Assign confidence tier using configurable offsets
    """
    # Validate calibration assumption: p_cover_at_zero should be ~0.5
    # A significant deviation suggests the calibration may have been trained incorrectly
    if hasattr(calibration, 'p_cover_at_zero'):
        if abs(calibration.p_cover_at_zero - 0.5) > 0.05:
            logger.warning(
                f"Calibration p_cover_at_zero={calibration.p_cover_at_zero:.4f} deviates from 0.5. "
                "This may indicate training data issues. Expected ~0.5 for absolute edge calibration."
            )

    # Extract required fields
    game_id = str(game.get("game_id", ""))
    home_team = str(game.get("home_team", ""))
    away_team = str(game.get("away_team", ""))
    year = game.get("year")
    week = game.get("week")

    # Get spreads - handle different column names
    if "jp_spread" in game.index:
        jp_spread = float(game["jp_spread"])
    elif "predicted_spread" in game.index:
        # Internal convention needs flip
        jp_spread = -float(game["predicted_spread"])
    else:
        raise ValueError("Game must have jp_spread or predicted_spread")

    vegas_spread = float(game["vegas_spread"])

    # Compute edge
    edge_pts = jp_spread - vegas_spread
    edge_abs = abs(edge_pts)
    jp_favored_side = "HOME" if edge_pts < 0 else "AWAY"

    # Breakeven probability (handles positive juice for alternate spreads)
    if juice < 0:
        p_breakeven = breakeven_prob(juice)
    else:
        # For positive odds: breakeven = 100 / (100 + odds)
        # e.g., +150: need to win 100/(100+150) = 40% to break even
        p_breakeven = 100 / (100 + juice)

    # Handle NO_BET (effectively zero edge) - use threshold to avoid floating point issues
    if edge_abs < MINIMUM_EDGE_PTS:
        return BetRecommendation(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            jp_spread=jp_spread,
            vegas_spread=vegas_spread,
            edge_pts=edge_pts,
            edge_abs=edge_abs,
            jp_favored_side=jp_favored_side,
            p_cover_no_push=None,
            p_push=None,
            p_cover=None,
            p_breakeven=p_breakeven,
            edge_prob=None,
            ev=None,
            juice=juice,
            confidence="NO_BET",
            year=int(year) if pd.notna(year) else None,
            week=int(week) if pd.notna(week) else None,
        )

    # Compute P(cover | no push) from calibration
    p_cover_no_push = float(predict_cover_probability(
        np.array([edge_abs]), calibration
    )[0])

    # Get push probability (0 if no push rates provided)
    if push_rates is not None:
        p_push = get_push_probability(vegas_spread, push_rates)
    else:
        p_push = 0.0

    # Compute unconditional P(cover)
    p_cover = p_cover_no_push * (1 - p_push)

    # Unconditional probability edge: both p_cover and breakeven scaled to
    # account for push probability. When p_push=0 (half-point spreads or V1
    # mode), this reduces to p_cover_no_push - p_breakeven.
    edge_prob = (p_cover_no_push - p_breakeven) * (1 - p_push)

    # Compute EV
    ev = calculate_ev_with_push(p_cover_no_push, p_push, juice)

    # Assign confidence tier using configurable offsets
    if ev >= min_ev_threshold + high_ev_offset:
        confidence = "HIGH"
    elif ev >= min_ev_threshold + med_ev_offset:
        confidence = "MED"
    elif ev >= min_ev_threshold:
        confidence = "BET"
    else:
        confidence = "PASS"

    return BetRecommendation(
        game_id=game_id,
        home_team=home_team,
        away_team=away_team,
        jp_spread=jp_spread,
        vegas_spread=vegas_spread,
        edge_pts=edge_pts,
        edge_abs=edge_abs,
        jp_favored_side=jp_favored_side,
        p_cover_no_push=p_cover_no_push,
        p_push=p_push,
        p_cover=p_cover,
        p_breakeven=p_breakeven,
        edge_prob=edge_prob,
        ev=ev,
        juice=juice,
        confidence=confidence,
        year=int(year) if pd.notna(year) else None,
        week=int(week) if pd.notna(week) else None,
    )


def evaluate_slate(
    games: pd.DataFrame,
    calibration: CalibrationResult,
    push_rates: Optional[PushRates] = None,
    min_ev_threshold: float = 0.03,
    juice: int = -110,
    high_ev_offset: float = 0.05,
    med_ev_offset: float = 0.02,
) -> list[BetRecommendation]:
    """Evaluate all games in a slate.

    Args:
        games: DataFrame with required columns for each game
        calibration: CalibrationResult from calibrate_cover_probability
        push_rates: Optional PushRates for push modeling
        min_ev_threshold: Minimum EV for BET tier
        juice: American odds (default: -110)
        high_ev_offset: EV above min_ev_threshold for HIGH tier (default: 0.05)
        med_ev_offset: EV above min_ev_threshold for MED tier (default: 0.02)

    Returns:
        List sorted by EV descending, PASS/NO_BET at bottom
    """
    recommendations = []

    for _, game in games.iterrows():
        try:
            rec = evaluate_game(
                game,
                calibration,
                push_rates,
                min_ev_threshold,
                juice,
                high_ev_offset,
                med_ev_offset,
            )
            recommendations.append(rec)
        except Exception as e:
            logger.warning(f"Failed to evaluate game {game.get('game_id')}: {e}")
            continue

    # Sort by EV descending, PASS/NO_BET at bottom
    def sort_key(rec: BetRecommendation) -> tuple:
        if rec.confidence in ("PASS", "NO_BET"):
            return (1, 0.0)  # Put at bottom
        return (0, -(rec.ev or 0))  # Negative EV for descending sort

    recommendations.sort(key=sort_key)

    return recommendations


def summarize_slate(recommendations: list[BetRecommendation]) -> dict:
    """Summarize a slate of recommendations.

    Args:
        recommendations: List of BetRecommendation objects

    Returns:
        Dictionary with summary statistics (native Python types for JSON compatibility)
    """
    bets = [r for r in recommendations if r.confidence not in ("PASS", "NO_BET")]
    passes = [r for r in recommendations if r.confidence == "PASS"]
    no_bets = [r for r in recommendations if r.confidence == "NO_BET"]

    tier_counts = {
        "HIGH": len([r for r in bets if r.confidence == "HIGH"]),
        "MED": len([r for r in bets if r.confidence == "MED"]),
        "BET": len([r for r in bets if r.confidence == "BET"]),
        "PASS": len(passes),
        "NO_BET": len(no_bets),
    }

    # Guard against None values and cast to native float for JSON compatibility
    ev_values = [r.ev for r in bets if r.ev is not None]
    p_cover_values = [r.p_cover for r in bets if r.p_cover is not None]
    edge_abs_values = [r.edge_abs for r in bets if r.edge_abs is not None]

    avg_ev = float(np.mean(ev_values)) if ev_values else 0.0
    avg_p_cover = float(np.mean(p_cover_values)) if p_cover_values else 0.0
    avg_edge_abs = float(np.mean(edge_abs_values)) if edge_abs_values else 0.0

    return {
        "total_games": len(recommendations),
        "total_bets": len(bets),
        "tier_counts": tier_counts,
        "avg_ev": avg_ev,
        "avg_p_cover": avg_p_cover,
        "avg_edge_abs": avg_edge_abs,
    }
