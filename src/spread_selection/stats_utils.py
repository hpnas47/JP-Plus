"""Statistical utilities for spread calibration analysis.

Provides:
- Wilson score confidence intervals for ATS%
- Bootstrap confidence intervals for ROI
- Matched comparison utilities
- EV decile reliability metrics
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# WILSON SCORE INTERVAL FOR ATS%
# =============================================================================

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval for binomial proportion.

    The Wilson score interval is preferred over the normal approximation
    because it handles edge cases (k=0, k=n) gracefully and has better
    coverage properties for small samples.

    Args:
        k: Number of successes (wins)
        n: Total trials (bets)
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        (lower, upper) bounds of confidence interval

    Examples:
        >>> wilson_ci(60, 100)  # 60% win rate
        (0.502, 0.691)

        >>> wilson_ci(0, 10)  # 0 wins - still gives reasonable bounds
        (0.0, 0.277)

        >>> wilson_ci(10, 10)  # All wins
        (0.723, 1.0)
    """
    if n == 0:
        return (0.0, 1.0)

    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)
    p_hat = k / n

    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    spread = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denominator

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)

    return (lower, upper)


@dataclass
class ATSWithCI:
    """ATS win rate with confidence interval."""
    wins: int
    losses: int
    n_bets: int
    ats_pct: float
    ci_low: float
    ci_high: float

    def __repr__(self) -> str:
        return f"ATS: {self.ats_pct:.1f}% [{self.ci_low:.1f}%, {self.ci_high:.1f}%] (n={self.n_bets})"


def compute_ats_with_ci(
    wins: int,
    losses: int,
    alpha: float = 0.05,
) -> ATSWithCI:
    """Compute ATS% with Wilson confidence interval.

    Args:
        wins: Number of winning bets
        losses: Number of losing bets
        alpha: Significance level

    Returns:
        ATSWithCI with rate and confidence bounds
    """
    n = wins + losses
    if n == 0:
        return ATSWithCI(
            wins=0, losses=0, n_bets=0,
            ats_pct=0.0, ci_low=0.0, ci_high=100.0
        )

    ats_pct = 100 * wins / n
    ci_low, ci_high = wilson_ci(wins, n, alpha)

    return ATSWithCI(
        wins=wins,
        losses=losses,
        n_bets=n,
        ats_pct=ats_pct,
        ci_low=100 * ci_low,
        ci_high=100 * ci_high,
    )


# =============================================================================
# BOOTSTRAP CI FOR ROI
# =============================================================================

def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 123,
) -> tuple[float, float]:
    """Bootstrap confidence interval for mean.

    Uses percentile method for simplicity and robustness.

    Args:
        values: Array of per-bet returns
        n_boot: Number of bootstrap resamples
        alpha: Significance level (default 0.05 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        (lower, upper) bounds of confidence interval

    Examples:
        >>> returns = np.array([0.909, -1.0, 0.909, 0.909, -1.0])
        >>> bootstrap_mean_ci(returns, n_boot=1000, seed=42)
        (-0.45, 0.72)
    """
    if len(values) == 0:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)

    # Vectorized bootstrap sampling
    # Shape: (n_boot, n_samples)
    boot_indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_samples = values[boot_indices]
    boot_means = boot_samples.mean(axis=1)

    # Percentile method
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return (float(lower), float(upper))


@dataclass
class ROIWithCI:
    """ROI with bootstrap confidence interval."""
    roi_pct: float
    ci_low: float
    ci_high: float
    n_bets: int
    total_return: float

    def __repr__(self) -> str:
        return f"ROI: {self.roi_pct:+.1f}% [{self.ci_low:+.1f}%, {self.ci_high:+.1f}%] (n={self.n_bets})"


def compute_roi_with_ci(
    wins: int,
    losses: int,
    pushes: int = 0,
    juice: int = -110,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 123,
) -> ROIWithCI:
    """Compute ROI% with bootstrap confidence interval.

    Args:
        wins: Number of winning bets
        losses: Number of losing bets
        pushes: Number of pushes (return stake)
        juice: American odds (default -110)
        n_boot: Bootstrap resamples
        alpha: Significance level
        seed: Random seed

    Returns:
        ROIWithCI with rate and confidence bounds
    """
    n_bets = wins + losses + pushes

    if n_bets == 0:
        return ROIWithCI(
            roi_pct=0.0, ci_low=0.0, ci_high=0.0,
            n_bets=0, total_return=0.0
        )

    # Compute per-bet returns
    win_return = 100 / abs(juice)  # +0.909 at -110
    loss_return = -1.0
    push_return = 0.0

    returns = np.concatenate([
        np.full(wins, win_return),
        np.full(losses, loss_return),
        np.full(pushes, push_return),
    ])

    # ROI = mean return
    roi = returns.mean()
    total_return = returns.sum()

    # Bootstrap CI
    ci_low, ci_high = bootstrap_mean_ci(returns, n_boot, alpha, seed)

    return ROIWithCI(
        roi_pct=100 * roi,
        ci_low=100 * ci_low,
        ci_high=100 * ci_high,
        n_bets=n_bets,
        total_return=total_return,
    )


# =============================================================================
# MATCHED COMPARISONS
# =============================================================================

def matched_top_n_by_week(
    df1: "pd.DataFrame",
    df2: "pd.DataFrame",
    ev_col: str = "ev_cal",
    week_col: str = "week",
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Match bet counts by week between two DataFrames.

    For each week, keeps top-N bets where N = min(#df1, #df2).
    Both DataFrames must have the same underlying game universe.

    Args:
        df1: First calibration bets (must have ev_col, week_col)
        df2: Second calibration bets
        ev_col: Column with predicted EV for ranking
        week_col: Column with week number

    Returns:
        (matched_df1, matched_df2) with equal bets per week
    """
    import pandas as pd

    matched1_indices = []
    matched2_indices = []

    weeks = sorted(set(df1[week_col].unique()) | set(df2[week_col].unique()))

    for week in weeks:
        week1 = df1[df1[week_col] == week].sort_values(ev_col, ascending=False)
        week2 = df2[df2[week_col] == week].sort_values(ev_col, ascending=False)

        n = min(len(week1), len(week2))

        if n > 0:
            matched1_indices.extend(week1.head(n).index.tolist())
            matched2_indices.extend(week2.head(n).index.tolist())

    matched_df1 = df1.loc[matched1_indices] if matched1_indices else pd.DataFrame()
    matched_df2 = df2.loc[matched2_indices] if matched2_indices else pd.DataFrame()

    return matched_df1, matched_df2


def compare_at_thresholds(
    df1: "pd.DataFrame",
    df2: "pd.DataFrame",
    thresholds: list[float],
    ev_col: str = "ev_cal",
    outcome_col: str = "jp_side_covered",
    push_col: str = "push",
) -> "pd.DataFrame":
    """Compare calibrations at multiple EV thresholds.

    Args:
        df1: First calibration results
        df2: Second calibration results
        thresholds: List of EV thresholds to compare
        ev_col: Column with predicted EV
        outcome_col: Column with outcome (True=win)
        push_col: Column with push flag

    Returns:
        DataFrame with comparison at each threshold
    """
    import pandas as pd

    results = []

    for thresh in thresholds:
        for name, df in [("cal1", df1), ("cal2", df2)]:
            # Filter to threshold
            mask = (df[ev_col] >= thresh) & ~df[push_col]
            subset = df[mask]

            n = len(subset)
            wins = subset[outcome_col].sum() if n > 0 else 0
            losses = n - wins

            if n > 0:
                ats = 100 * wins / n
                roi = (wins * (100/110) - losses) / n * 100
            else:
                ats = 0.0
                roi = 0.0

            results.append({
                "threshold": thresh,
                "calibration": name,
                "n_bets": n,
                "wins": int(wins),
                "losses": int(losses),
                "ats_pct": ats,
                "roi_pct": roi,
            })

    return pd.DataFrame(results)


# =============================================================================
# EV DECILE RELIABILITY
# =============================================================================

@dataclass
class DecileReliability:
    """EV decile reliability metrics."""
    decile_stats: "pd.DataFrame"
    spearman_corr: float
    is_monotonic: bool
    monotonicity_violations: int


def compute_ev_decile_reliability(
    df: "pd.DataFrame",
    ev_col: str = "ev_cal",
    outcome_col: str = "jp_side_covered",
    push_col: str = "push",
    n_deciles: int = 10,
) -> DecileReliability:
    """Compute EV decile reliability metrics.

    For well-calibrated models, higher EV deciles should have higher
    realized win rates. Non-monotonicity suggests calibration issues.

    Args:
        df: DataFrame with predictions and outcomes
        ev_col: Column with predicted EV
        outcome_col: Column with outcome (True=win)
        push_col: Column with push flag
        n_deciles: Number of bins (default 10)

    Returns:
        DecileReliability with stats and monotonicity check
    """
    import pandas as pd
    from scipy.stats import spearmanr

    # Filter to valid bets
    valid_mask = df[ev_col].notna() & ~df[push_col]
    df_valid = df[valid_mask].copy()

    if len(df_valid) < n_deciles:
        # Not enough data for deciles
        return DecileReliability(
            decile_stats=pd.DataFrame(),
            spearman_corr=0.0,
            is_monotonic=False,
            monotonicity_violations=0,
        )

    # Create deciles
    df_valid["decile"] = pd.qcut(
        df_valid[ev_col], q=n_deciles, labels=False, duplicates="drop"
    )

    # Compute stats per decile
    stats = []
    for decile in sorted(df_valid["decile"].unique()):
        subset = df_valid[df_valid["decile"] == decile]
        n = len(subset)
        wins = subset[outcome_col].sum()
        losses = n - wins

        ev_min = subset[ev_col].min()
        ev_max = subset[ev_col].max()
        ev_mean = subset[ev_col].mean()

        ats = 100 * wins / n if n > 0 else 0
        roi = (wins * (100/110) - losses) / n * 100 if n > 0 else 0

        stats.append({
            "decile": int(decile),
            "ev_min": ev_min,
            "ev_max": ev_max,
            "ev_mean": ev_mean,
            "n_bets": n,
            "wins": int(wins),
            "losses": int(losses),
            "ats_pct": ats,
            "roi_pct": roi,
        })

    decile_df = pd.DataFrame(stats)

    # Check monotonicity
    ats_values = decile_df["ats_pct"].values
    violations = 0
    for i in range(1, len(ats_values)):
        if ats_values[i] < ats_values[i-1] - 2.0:  # 2pp tolerance
            violations += 1

    is_monotonic = violations <= 1  # Allow 1 violation

    # Spearman correlation
    if len(decile_df) >= 3:
        corr, _ = spearmanr(decile_df["decile"], decile_df["ats_pct"])
    else:
        corr = 0.0

    return DecileReliability(
        decile_stats=decile_df,
        spearman_corr=float(corr),
        is_monotonic=is_monotonic,
        monotonicity_violations=violations,
    )


# =============================================================================
# RECOMMENDATION LOGIC
# =============================================================================

@dataclass
class CalibrationRecommendation:
    """Recommendation for 2026 calibration policy."""
    recommended_calibration: str
    confidence: str  # "high", "moderate", "low"
    rationale: str
    mode: str  # "conservative", "balanced", "aggressive"
    constraints: list[str]


def generate_recommendation(
    phase2_oos: ROIWithCI,
    weighted_oos: ROIWithCI,
    phase2_ats: ATSWithCI,
    weighted_ats: ATSWithCI,
    matched_phase2_ats: Optional[ATSWithCI] = None,
    matched_weighted_ats: Optional[ATSWithCI] = None,
) -> CalibrationRecommendation:
    """Generate calibration recommendation for 2026.

    Decision rules:
    1. If ROI CIs don't overlap and one dominates -> recommend that one (high confidence)
    2. If ATS CIs don't overlap and one dominates -> recommend that one (moderate confidence)
    3. If both overlap -> recommend based on volume vs edge tradeoff (low confidence)

    Args:
        phase2_oos: Phase 2 OOS ROI with CI
        weighted_oos: Weighted OOS ROI with CI
        phase2_ats: Phase 2 OOS ATS with CI
        weighted_ats: Weighted OOS ATS with CI
        matched_phase2_ats: Optional matched ATS for phase2
        matched_weighted_ats: Optional matched ATS for weighted

    Returns:
        CalibrationRecommendation with decision and rationale
    """
    constraints = [
        "Phase 1 (weeks 0-3): Skip EV bets (phase1_policy='skip')",
        "Max 10 bets per week to reduce variance",
        "Require min 3% EV for selection",
    ]

    # Check ROI CI overlap
    roi_overlap = not (phase2_oos.ci_high < weighted_oos.ci_low or
                       weighted_oos.ci_high < phase2_oos.ci_low)

    # Check ATS CI overlap
    ats_overlap = not (phase2_ats.ci_high < weighted_ats.ci_low or
                       weighted_ats.ci_high < phase2_ats.ci_low)

    # Decision logic
    if not roi_overlap:
        # Clear winner by ROI
        if phase2_oos.roi_pct > weighted_oos.roi_pct:
            return CalibrationRecommendation(
                recommended_calibration="phase2_only",
                confidence="high",
                rationale=(
                    f"Phase 2 ROI ({phase2_oos.roi_pct:+.1f}%) CI does not overlap with "
                    f"Weighted ROI ({weighted_oos.roi_pct:+.1f}%). Clear statistical winner."
                ),
                mode="balanced",
                constraints=constraints,
            )
        else:
            return CalibrationRecommendation(
                recommended_calibration="weighted",
                confidence="high",
                rationale=(
                    f"Weighted ROI ({weighted_oos.roi_pct:+.1f}%) CI does not overlap with "
                    f"Phase 2 ROI ({phase2_oos.roi_pct:+.1f}%). Clear statistical winner."
                ),
                mode="conservative",
                constraints=constraints,
            )

    elif not ats_overlap:
        # Clear winner by ATS
        if phase2_ats.ats_pct > weighted_ats.ats_pct:
            return CalibrationRecommendation(
                recommended_calibration="phase2_only",
                confidence="moderate",
                rationale=(
                    f"Phase 2 ATS ({phase2_ats.ats_pct:.1f}%) CI does not overlap with "
                    f"Weighted ATS ({weighted_ats.ats_pct:.1f}%). ATS-based winner."
                ),
                mode="balanced",
                constraints=constraints,
            )
        else:
            return CalibrationRecommendation(
                recommended_calibration="weighted",
                confidence="moderate",
                rationale=(
                    f"Weighted ATS ({weighted_ats.ats_pct:.1f}%) CI does not overlap with "
                    f"Phase 2 ATS ({phase2_ats.ats_pct:.1f}%). ATS-based winner."
                ),
                mode="conservative",
                constraints=constraints,
            )

    else:
        # CIs overlap - use heuristics
        # Prefer higher volume if ATS is similar
        volume_ratio = phase2_oos.n_bets / max(weighted_oos.n_bets, 1)

        if volume_ratio > 1.3 and phase2_ats.ats_pct > 50:
            return CalibrationRecommendation(
                recommended_calibration="phase2_only",
                confidence="low",
                rationale=(
                    f"CIs overlap. Phase 2 has {volume_ratio:.1f}x more bets "
                    f"({phase2_oos.n_bets} vs {weighted_oos.n_bets}) with positive ATS. "
                    "Higher volume preferred when edge is marginal."
                ),
                mode="aggressive",
                constraints=constraints + ["Consider raising EV threshold to 5% for safety"],
            )
        else:
            # Default to weighted (conservative)
            return CalibrationRecommendation(
                recommended_calibration="weighted",
                confidence="low",
                rationale=(
                    f"CIs overlap and neither dominates. Weighted shows higher OOS ATS "
                    f"({weighted_ats.ats_pct:.1f}% vs {phase2_ats.ats_pct:.1f}%) despite "
                    f"smaller sample. Recommending conservative approach."
                ),
                mode="conservative",
                constraints=constraints,
            )
