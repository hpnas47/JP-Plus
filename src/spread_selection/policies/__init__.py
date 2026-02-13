"""Policy layers for spread selection.

Policy layers filter/gate betting recommendations AFTER EV computation,
without modifying calibration, p_cover, or confidence logic.
"""

from .phase1_sp_gate import (
    Phase1SPGateConfig,
    Phase1SPGateResult,
    SPGateCategory,
    SPGateMode,
    apply_phase1_sp_gate,
    fetch_sp_spreads_vegas,
)

__all__ = [
    "Phase1SPGateConfig",
    "Phase1SPGateResult",
    "SPGateCategory",
    "SPGateMode",
    "apply_phase1_sp_gate",
    "fetch_sp_spreads_vegas",
]
