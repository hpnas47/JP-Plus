"""Betting strategies for spread selection.

Strategies are distinct selection methods that produce separate bet lists.
They do NOT interfere with or modify the primary EV-based selection engine.

Available Strategies:
- Phase1EdgeBaseline: Edge-based selection for Phase 1 (LIST B, default in weeks 0-3)
  Includes optional HYBRID_VETO_2 overlay (OFF by default)
"""

from .phase1_edge_baseline import (
    # Configs
    Phase1EdgeBaselineConfig,
    Phase1EdgeVetoConfig,
    # Recommendation
    Phase1EdgeRecommendation,
    # Enums
    Phase1EdgeResult,
    # Evaluation functions
    evaluate_game_edge_baseline,
    evaluate_slate_edge_baseline,
    # Utility functions
    recommendations_to_dataframe,
    summarize_recommendations,
)

__all__ = [
    # Phase1EdgeBaseline (LIST B)
    "Phase1EdgeBaselineConfig",
    "Phase1EdgeVetoConfig",
    "Phase1EdgeRecommendation",
    "Phase1EdgeResult",
    "evaluate_game_edge_baseline",
    "evaluate_slate_edge_baseline",
    "recommendations_to_dataframe",
    "summarize_recommendations",
]
