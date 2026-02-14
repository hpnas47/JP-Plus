"""Policy layers for spread selection.

Policy layers filter/gate betting recommendations AFTER EV computation,
without modifying calibration, p_cover, or confidence logic.
"""

# SP+ gate removed (2026-02-14) - research showed unstable year-to-year results
# See docs/PHASE1_SP_POLICY.md for rationale

__all__ = []
