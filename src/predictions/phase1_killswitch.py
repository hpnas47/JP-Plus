"""Phase 1 Kill-Switch Risk Control.

Protects against "2022-like" early-season regimes by reducing or disabling
Phase 1 betting if early live results are very poor.

Default OFF - designed for operational safety, not routine use.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class KillswitchAction(Enum):
    """Actions when kill-switch triggers."""
    DISABLE_PHASE1_BETS = "disable_phase1_bets"  # Set value plays to empty
    RAISE_THRESHOLD = "raise_threshold"  # Increase jp_edge_min


@dataclass
class KillswitchConfig:
    """Configuration for Phase 1 kill-switch."""
    enabled: bool = False
    weeks_observed: int = 1  # Evaluate after Week 1 (or 2)
    min_bets: int = 5  # Don't trigger on tiny sample
    trigger_ats: float = 0.40  # If ATS% <= 40%, trigger
    action: str = "disable_phase1_bets"
    raised_jp_edge_min: float = 8.0  # Used only if action="raise_threshold"

    @property
    def kill_action(self) -> KillswitchAction:
        return KillswitchAction(self.action)


@dataclass
class KillswitchResult:
    """Result of kill-switch evaluation."""
    triggered: bool
    reason: str
    action_taken: Optional[str]
    observed_bets: int
    observed_wins: int
    observed_losses: int
    observed_ats_pct: float
    new_jp_edge_min: Optional[float]  # If action was raise_threshold


def evaluate_killswitch(
    config: KillswitchConfig,
    current_week: int,
    season_results_df: pd.DataFrame,
) -> KillswitchResult:
    """
    Evaluate whether the kill-switch should trigger.

    Args:
        config: Kill-switch configuration
        current_week: Week we are predicting for (2 or 3)
        season_results_df: DataFrame with columns:
            - week: Game week
            - bet_placed: bool (True if this was a value play)
            - ats_outcome: 1.0 (win), 0.0 (loss), 0.5 (push), None (no bet)

    Returns:
        KillswitchResult with trigger status and action taken
    """
    if not config.enabled:
        return KillswitchResult(
            triggered=False,
            reason="Kill-switch disabled",
            action_taken=None,
            observed_bets=0,
            observed_wins=0,
            observed_losses=0,
            observed_ats_pct=0.0,
            new_jp_edge_min=None,
        )

    # Only evaluate if we're past the observation period
    if current_week <= config.weeks_observed:
        return KillswitchResult(
            triggered=False,
            reason=f"Week {current_week} <= observation period ({config.weeks_observed})",
            action_taken=None,
            observed_bets=0,
            observed_wins=0,
            observed_losses=0,
            observed_ats_pct=0.0,
            new_jp_edge_min=None,
        )

    # Filter to observed weeks' bets
    observed = season_results_df[
        (season_results_df["week"] <= config.weeks_observed) &
        (season_results_df["bet_placed"] == True)
    ]

    if len(observed) == 0:
        return KillswitchResult(
            triggered=False,
            reason="No bets placed in observation period",
            action_taken=None,
            observed_bets=0,
            observed_wins=0,
            observed_losses=0,
            observed_ats_pct=0.0,
            new_jp_edge_min=None,
        )

    # Count W-L (exclude pushes)
    outcomes = observed["ats_outcome"].dropna()
    wins = int((outcomes == 1.0).sum())
    losses = int((outcomes == 0.0).sum())
    total = wins + losses

    if total < config.min_bets:
        return KillswitchResult(
            triggered=False,
            reason=f"Only {total} graded bets < min_bets ({config.min_bets})",
            action_taken=None,
            observed_bets=total,
            observed_wins=wins,
            observed_losses=losses,
            observed_ats_pct=0.0,
            new_jp_edge_min=None,
        )

    ats_pct = wins / total

    if ats_pct > config.trigger_ats:
        return KillswitchResult(
            triggered=False,
            reason=f"ATS {ats_pct:.1%} > trigger threshold {config.trigger_ats:.0%}",
            action_taken=None,
            observed_bets=total,
            observed_wins=wins,
            observed_losses=losses,
            observed_ats_pct=ats_pct,
            new_jp_edge_min=None,
        )

    # TRIGGER!
    action = config.kill_action

    if action == KillswitchAction.DISABLE_PHASE1_BETS:
        action_str = "Disabling all Phase 1 value plays"
        new_min = None
    else:  # RAISE_THRESHOLD
        action_str = f"Raising jp_edge_min to {config.raised_jp_edge_min}"
        new_min = config.raised_jp_edge_min

    logger.warning(
        f"⚠️ PHASE 1 KILL-SWITCH TRIGGERED! "
        f"Week {current_week}, observed {total} bets with {ats_pct:.0%} ATS "
        f"(<= {config.trigger_ats:.0%} threshold). Action: {action_str}"
    )

    return KillswitchResult(
        triggered=True,
        reason=f"ATS {ats_pct:.1%} <= trigger threshold {config.trigger_ats:.0%}",
        action_taken=action_str,
        observed_bets=total,
        observed_wins=wins,
        observed_losses=losses,
        observed_ats_pct=ats_pct,
        new_jp_edge_min=new_min,
    )


def apply_killswitch_action(
    result: KillswitchResult,
    value_plays_df: pd.DataFrame,
    config: KillswitchConfig,
) -> pd.DataFrame:
    """
    Apply kill-switch action to value plays.

    Args:
        result: KillswitchResult from evaluate_killswitch
        value_plays_df: Current value plays DataFrame
        config: Kill-switch configuration

    Returns:
        Filtered/modified value_plays_df
    """
    if not result.triggered:
        return value_plays_df

    action = config.kill_action

    if action == KillswitchAction.DISABLE_PHASE1_BETS:
        logger.info(f"Kill-switch: Removing all {len(value_plays_df)} Phase 1 value plays")
        return value_plays_df.iloc[0:0]  # Empty DataFrame with same schema

    elif action == KillswitchAction.RAISE_THRESHOLD:
        # Filter to only high-edge bets
        edge_col = 'edge' if 'edge' in value_plays_df.columns else 'edge_recommended'
        if edge_col in value_plays_df.columns:
            pre_count = len(value_plays_df)
            value_plays_df = value_plays_df[
                value_plays_df[edge_col].abs() >= config.raised_jp_edge_min
            ].copy()
            logger.info(
                f"Kill-switch: Raised threshold to {config.raised_jp_edge_min}+ edge, "
                f"{pre_count} → {len(value_plays_df)} value plays"
            )
        return value_plays_df

    return value_plays_df


def build_results_df_from_backtest(
    predictions: list[dict],
    value_plays_game_ids: set[int],
) -> pd.DataFrame:
    """
    Build a results DataFrame from backtest predictions for kill-switch evaluation.

    Args:
        predictions: List of prediction dicts from backtest
        value_plays_game_ids: Set of game_ids that were value plays

    Returns:
        DataFrame with week, bet_placed, ats_outcome columns
    """
    records = []
    for pred in predictions:
        game_id = pred.get("game_id")
        week = pred.get("week")
        actual = pred.get("actual_margin")
        predicted = pred.get("predicted_spread")
        vegas = pred.get("vegas_spread")

        bet_placed = game_id in value_plays_game_ids

        # Compute ATS outcome if bet was placed
        ats_outcome = None
        if bet_placed and vegas is not None and actual is not None and predicted is not None:
            edge = predicted + vegas  # internal + vegas_raw
            if abs(edge) >= 0.5:
                vegas_internal = -vegas
                if edge > 0:
                    ats_outcome = 1.0 if actual > vegas_internal else (0.5 if actual == vegas_internal else 0.0)
                else:
                    ats_outcome = 1.0 if actual < vegas_internal else (0.5 if actual == vegas_internal else 0.0)

        records.append({
            "game_id": game_id,
            "week": week,
            "bet_placed": bet_placed,
            "ats_outcome": ats_outcome,
        })

    return pd.DataFrame(records)
