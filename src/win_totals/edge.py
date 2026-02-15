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
    edge: float  # model_prob - breakeven_prob (push-adjusted; consistent with EV)
    leakage_contribution_pct: float = 0.0  # fraction of signal from LEAKAGE_RISK features
    leakage_warning: bool = False  # True if leakage_contribution_pct > 0.25


def breakeven_prob(odds: int, push_prob: float = 0.0) -> float:
    """Convert American odds to breakeven win probability.

    This is the minimum win probability needed to break even at these odds.
    With push_prob=0, this equals the raw implied probability (including vig).
    It is NOT the no-vig fair probability; for that, normalize both sides of
    a market to sum to 1.0.

    When push_prob > 0, returns the push-adjusted breakeven: the win probability
    needed to break even given that pushes return the stake.

    Args:
        odds: American odds (e.g., -110, +150)
        push_prob: Probability of a push (refund). Default 0.0.

    Returns:
        Breakeven probability in [0, 1]
    """
    if odds < 0:
        payout = 100.0 / abs(odds)
    else:
        payout = odds / 100.0

    # Solve: p * payout - (1 - p - push_prob) = 0
    # p * (payout + 1) = 1 - push_prob
    return (1.0 - push_prob) / (payout + 1.0)


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


def leakage_feature_count_fraction(
    feature_names: list[str],
    feature_metadata: dict[str, dict],
) -> float:
    """Compute unweighted fraction of features with LEAKAGE_RISK status.

    This counts features, NOT signal magnitude. A model with 1/20 leakage
    features returns 0.05 even if that feature dominates the prediction.

    For the authoritative magnitude-weighted metric that drives the
    leakage_warning flag in BetRecommendation, use compute_leakage_contribution().

    Args:
        feature_names: List of feature names used in model
        feature_metadata: Dict mapping feature name to metadata with 'status'
                          or legacy 'leakage_risk' boolean

    Returns:
        Fraction of features flagged as leakage risk (count-based)
    """
    if not feature_names:
        return 0.0
    flagged = 0
    for f in feature_names:
        meta = feature_metadata.get(f, {})
        if meta.get('status') == 'LEAKAGE_RISK' or meta.get('leakage_risk', False):
            flagged += 1
    return flagged / len(feature_names)


# Backward-compatible alias (deprecated)
leakage_risk_fraction = leakage_feature_count_fraction


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

        # Guard against PMF normalization errors or over/under boundary
        # mismatches on whole-number lines (e.g., push double-counted)
        prob_sum = p_over + p_under + p_push
        if abs(prob_sum - 1.0) >= 0.01:
            # True error: PMF is badly broken (>1% off). Skip.
            logger.error(
                f"PMF severely broken for {bl.team}: "
                f"p_over={p_over:.6f}, p_under={p_under:.6f}, p_push={p_push:.6f}, "
                f"sum={prob_sum:.6f}. Skipping."
            )
            continue
        elif abs(prob_sum - 1.0) >= 1e-9:
            # Normal floating-point noise from convolution. Normalize and proceed.
            p_over /= prob_sum
            p_under /= prob_sum
            p_push /= prob_sum
            logger.debug(
                f"Normalized probabilities for {bl.team}: sum was {prob_sum:.8f}"
            )

        team_leakage = leakage_pcts.get(bl.team, 0.0)
        leakage_warning = team_leakage > LEAKAGE_WARNING_THRESHOLD

        # Evaluate over (push-adjusted breakeven for correct edge display)
        over_ev = calculate_ev(p_over, bl.over_odds, push_prob=p_push)
        over_be = breakeven_prob(bl.over_odds, push_prob=p_push)

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

        # Evaluate under (push-adjusted breakeven for correct edge display)
        under_ev = calculate_ev(p_under, bl.under_odds, push_prob=p_push)
        under_be = breakeven_prob(bl.under_odds, push_prob=p_push)

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

    # Resolve contradictory both-sides recommendations.
    # With standard -110/-110 juice, both sides CANNOT have positive EV
    # (proven: over_ev + under_ev = -0.0909 + 0.0909*p_push < 0 for p_push < 1).
    # If both trigger, it indicates a probability normalization bug or data error.
    # Keep only the higher-EV side.
    teams_by_side: dict[str, dict[str, float]] = {}
    for rec in recommendations:
        teams_by_side.setdefault(rec.team, {})[rec.side] = rec.ev
    for team, sides in teams_by_side.items():
        if "Over" in sides and "Under" in sides:
            lower_side = "Over" if sides["Over"] < sides["Under"] else "Under"
            logger.error(
                f"Both Over and Under have positive EV for {team} "
                f"(Over EV={sides['Over']:.4f}, Under EV={sides['Under']:.4f}). "
                f"This indicates a probability normalization bug or non-standard odds. "
                f"Dropping {lower_side} (lower EV)."
            )
            recommendations = [
                r for r in recommendations
                if not (r.team == team and r.side == lower_side)
            ]

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

    lines.append("")

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
