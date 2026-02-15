"""EV calculation and bet recommendations for win totals.

Pure math module — no API dependencies. Takes WinTotalDistribution
objects and book lines, computes expected value, and generates
bet recommendations with leakage contribution tracking.
"""

import csv
import logging
from dataclasses import dataclass

import numpy as np

from src.win_totals.schedule import WinTotalDistribution

logger = logging.getLogger(__name__)

# Standard American odds juice
STANDARD_JUICE = -110

# Warning threshold: if > 25% of a prediction's signal comes from
# LEAKAGE_RISK features, flag the recommendation
LEAKAGE_WARNING_THRESHOLD = 0.25


@dataclass
class BookLine:
    """Book's posted win total line for a team."""
    team: str
    year: int
    line: float  # e.g., 8.5
    over_odds: int = STANDARD_JUICE  # American odds
    under_odds: int = STANDARD_JUICE


@dataclass
class BetRecommendation:
    """A recommended bet with EV analysis."""
    team: str
    year: int
    side: str  # "Over" or "Under"
    line: float
    odds: int
    model_prob: float
    breakeven_prob: float
    ev: float  # Expected value per $1 risked
    confidence: str  # "Strong", "Moderate", "Lean"
    expected_wins: float
    edge: float  # model_prob - breakeven_prob
    leakage_contribution_pct: float = 0.0  # fraction of signal from LEAKAGE_RISK features
    leakage_warning: bool = False  # True if leakage_contribution_pct > 0.25


def breakeven_prob(odds: int) -> float:
    """Convert American odds to breakeven probability (no-vig implied).

    Args:
        odds: American odds (e.g., -110, +150)

    Returns:
        Breakeven probability in [0, 1]
    """
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    else:
        return 100.0 / (odds + 100.0)


def calculate_ev(
    model_prob: float, odds: int, push_prob: float = 0.0,
) -> float:
    """Calculate expected value per $1 risked.

    Args:
        model_prob: Model's estimated probability of winning the bet
        odds: American odds
        push_prob: Probability of a push (refund, no gain/loss)

    Returns:
        EV per $1 risked (positive = profitable)
    """
    if odds < 0:
        payout = 100.0 / abs(odds)  # profit per $1 risked
    else:
        payout = odds / 100.0

    lose_prob = 1.0 - model_prob - push_prob
    ev = model_prob * payout - lose_prob
    return ev


def prob_over_under(dist: WinTotalDistribution, line: float) -> tuple[float, float]:
    """Get model probability of over and under for a given line.

    Args:
        dist: Win total distribution
        line: Book's win total line

    Returns:
        (prob_over, prob_under) tuple
    """
    p_over = dist.prob_over(line)
    p_under = dist.prob_under(line)
    return p_over, p_under


def leakage_risk_fraction(
    feature_names: list[str],
    feature_metadata: dict[str, dict],
) -> float:
    """Compute fraction of features that have LEAKAGE_RISK status.

    Args:
        feature_names: List of feature names used in model
        feature_metadata: Dict mapping feature name to metadata with 'status'
                          or legacy 'leakage_risk' boolean

    Returns:
        Fraction of features flagged as leakage risk
    """
    if not feature_names:
        return 0.0
    flagged = 0
    for f in feature_names:
        meta = feature_metadata.get(f, {})
        # Support both new 'status' field and legacy 'leakage_risk' boolean
        if meta.get('status') == 'LEAKAGE_RISK' or meta.get('leakage_risk', False):
            flagged += 1
    return flagged / len(feature_names)


def compute_feature_contributions(
    coefficients: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """Compute each feature's contribution to a prediction.

    contribution_i = coef_i * value_i (using standardized z-values)

    Args:
        coefficients: Ridge regression coefficients (on standardized features)
        feature_values: Standardized feature values for a single team (z-scores)
        feature_names: Feature names

    Returns:
        Dict mapping feature name to point contribution
    """
    contributions = coefficients * feature_values
    return {name: float(c) for name, c in zip(feature_names, contributions)}


def compute_leakage_contribution(
    coefficients: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list[str],
    feature_metadata: dict[str, dict],
) -> float:
    """Compute fraction of total |signal| from LEAKAGE_RISK features.

    leakage_pct = sum(|beta_j * z_j| for LEAKAGE_RISK j) / sum(|beta_j * z_j| for all j)

    Args:
        coefficients: Ridge coefficients (on standardized features)
        feature_values: Standardized feature values (z-scores)
        feature_names: Feature names
        feature_metadata: Feature metadata with 'status' field

    Returns:
        Fraction in [0, 1] of signal from LEAKAGE_RISK features
    """
    abs_contributions = np.abs(coefficients * feature_values)
    total = float(np.sum(abs_contributions))

    if total == 0:
        return 0.0

    leakage_sum = 0.0
    for i, name in enumerate(feature_names):
        meta = feature_metadata.get(name, {})
        if meta.get('status') == 'LEAKAGE_RISK' or meta.get('leakage_risk', False):
            leakage_sum += abs_contributions[i]

    return leakage_sum / total


def evaluate_all(
    distributions: list[WinTotalDistribution],
    book_lines: list[BookLine],
    min_ev: float = 0.02,
    leakage_pcts: dict[str, float] | None = None,
) -> list[BetRecommendation]:
    """Evaluate all teams against book lines and generate recommendations.

    Args:
        distributions: Win total distributions for all teams
        book_lines: Book's posted lines
        min_ev: Minimum EV threshold for recommendation
        leakage_pcts: Optional dict {team: leakage_contribution_pct}

    Returns:
        List of BetRecommendation sorted by EV descending
    """
    dist_by_team = {d.team: d for d in distributions}
    leakage_pcts = leakage_pcts or {}

    recommendations = []

    for bl in book_lines:
        dist = dist_by_team.get(bl.team)
        if dist is None:
            logger.warning(f"No distribution for {bl.team}, skipping")
            continue

        p_over, p_under = prob_over_under(dist, bl.line)
        p_push = dist.prob_push(bl.line)
        team_leakage = leakage_pcts.get(bl.team, 0.0)
        leakage_warning = team_leakage > LEAKAGE_WARNING_THRESHOLD

        # Evaluate over
        over_ev = calculate_ev(p_over, bl.over_odds, push_prob=p_push)
        over_be = breakeven_prob(bl.over_odds)

        if over_ev >= min_ev:
            confidence = _classify_confidence(over_ev)
            rec = BetRecommendation(
                team=bl.team,
                year=bl.year,
                side="Over",
                line=bl.line,
                odds=bl.over_odds,
                model_prob=p_over,
                breakeven_prob=over_be,
                ev=over_ev,
                confidence=confidence,
                expected_wins=dist.expected_wins,
                edge=p_over - over_be,
                leakage_contribution_pct=team_leakage,
                leakage_warning=leakage_warning,
            )
            recommendations.append(rec)

        # Evaluate under
        under_ev = calculate_ev(p_under, bl.under_odds, push_prob=p_push)
        under_be = breakeven_prob(bl.under_odds)

        if under_ev >= min_ev:
            confidence = _classify_confidence(under_ev)
            rec = BetRecommendation(
                team=bl.team,
                year=bl.year,
                side="Under",
                line=bl.line,
                odds=bl.under_odds,
                model_prob=p_under,
                breakeven_prob=under_be,
                ev=under_ev,
                confidence=confidence,
                expected_wins=dist.expected_wins,
                edge=p_under - under_be,
                leakage_contribution_pct=team_leakage,
                leakage_warning=leakage_warning,
            )
            recommendations.append(rec)

    # Sort by EV descending
    recommendations.sort(key=lambda r: r.ev, reverse=True)
    return recommendations


def _classify_confidence(ev: float) -> str:
    """Classify bet confidence based on EV."""
    if ev >= 0.08:
        return "Strong"
    elif ev >= 0.04:
        return "Moderate"
    else:
        return "Lean"


def generate_report(
    recommendations: list[BetRecommendation],
    csv_path: str | None = None,
) -> str:
    """Generate console report and optionally save to CSV.

    Args:
        recommendations: Sorted bet recommendations
        csv_path: Optional path to save CSV

    Returns:
        Formatted console report string
    """
    if not recommendations:
        return "No bets meet the minimum EV threshold."

    lines = []
    lines.append(f"{'#':>3}  {'Team':<20} {'Side':<6} {'Line':>5} {'Odds':>5} "
                  f"{'Model%':>7} {'BE%':>7} {'EV':>6} {'Edge':>6} {'E[W]':>5} {'Conf':<8} {'Leak':>5}")
    lines.append("-" * 95)

    for i, rec in enumerate(recommendations, 1):
        warn = " ⚠" if rec.leakage_warning else ""
        lines.append(
            f"{i:>3}  {rec.team:<20} {rec.side:<6} {rec.line:>5.1f} {rec.odds:>5} "
            f"{rec.model_prob:>6.1%} {rec.breakeven_prob:>6.1%} "
            f"{rec.ev:>+5.1%} {rec.edge:>+5.1%} {rec.expected_wins:>5.1f} "
            f"{rec.confidence:<8} {rec.leakage_contribution_pct:>4.0%}{warn}"
        )

    report = "\n".join(lines)

    if csv_path:
        _save_csv(recommendations, csv_path)
        logger.info(f"Saved {len(recommendations)} recommendations to {csv_path}")

    return report


def _save_csv(recommendations: list[BetRecommendation], path: str) -> None:
    """Save recommendations to CSV."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'team', 'year', 'side', 'line', 'odds', 'model_prob',
            'breakeven_prob', 'ev', 'edge', 'expected_wins', 'confidence',
            'leakage_contribution_pct', 'leakage_warning',
        ])
        for rec in recommendations:
            writer.writerow([
                rec.team, rec.year, rec.side, rec.line, rec.odds,
                f"{rec.model_prob:.4f}", f"{rec.breakeven_prob:.4f}",
                f"{rec.ev:.4f}", f"{rec.edge:.4f}",
                f"{rec.expected_wins:.2f}", rec.confidence,
                f"{rec.leakage_contribution_pct:.4f}",
                rec.leakage_warning,
            ])
