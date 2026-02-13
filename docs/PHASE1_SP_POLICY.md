# Phase 1 SP+ Policy Documentation

**Status:** Research Frozen (2026-02-13)
**Decision:** SP+ filtering is not stable. Keep SP+ as tagging-only unless new seasons change conclusions.

---

## Overview

This document defines the policy for SP+ usage in Phase 1 (weeks 1-3) spread predictions. Based on extensive backtest validation (2022-2025, N=395 games with 5+ pt edge), we have determined that SP+ filtering is unreliable and should not be operationalized.

## Bet List Architecture

The spread selection system emits TWO separate bet lists in Phase 1:

### List A: ENGINE_EV (Primary Engine)

| Field | Value |
|-------|-------|
| `list_family` | `ENGINE_EV` |
| `list_name` | `PRIMARY` or `ULTRA` |
| `selection_basis` | `EV` |
| `is_official_engine` | `true` (PRIMARY only) |
| `execution_default` | `true` (PRIMARY only) |
| `line_type` | `close` |

**Key Points:**
- EV-based calibrated selection engine output
- **NO SP+ filtering applied** (tagging only for monitoring)
- This is the official betting recommendation
- SP+ data is included for informational purposes but does NOT affect selection

### List B: PHASE1_EDGE (Edge-Based Visibility List)

| Field | Value |
|-------|-------|
| `list_family` | `PHASE1_EDGE` |
| `list_name` | `EDGE_BASELINE` or `EDGE_HYBRID_VETO_2` |
| `selection_basis` | `EDGE` |
| `is_official_engine` | `false` |
| `execution_default` | `false` |
| `line_type` | `open` |

**Key Points:**
- Edge-based selection (|jp_edge| >= 5.0 vs OPEN lines)
- Auto-emitted in weeks 1-3 for visibility
- NOT the official engine recommendation
- Optional HYBRID_VETO_2 overlay available (default OFF)

---

## HYBRID_VETO_2 Rule (Optional Overlay)

HYBRID_VETO_2 is the ONLY SP+ filtering mechanism that passed all guardrails in backtesting. It is **OFF by default** and must be explicitly enabled.

### When to Veto

VETO a bet from List B if ALL of the following are true:
1. `oppose` is True (SP+ and JP+ disagree on side)
2. `|sp_edge|` >= 2.0 (strong SP+ opposition)
3. 5.0 <= `|jp_edge|` < 8.0 (marginal band only)

### Never Veto

**NEVER veto** if `|jp_edge|` >= 8.0 (high-conviction bets are protected)

### Implementation

```python
# Config flags (all default OFF)
phase1_edge_veto_enabled: bool = False
phase1_edge_veto_sp_oppose_min: float = 2.0
phase1_edge_veto_jp_band_low: float = 5.0
phase1_edge_veto_jp_band_high: float = 8.0
```

CLI:
```bash
# Enable HYBRID_VETO_2
python -m src.spread_selection.run_selection predict \
    --slate-csv slate.csv \
    --phase1-edge-veto
```

---

## Backtest Results Summary

### HYBRID_VETO_2 Validation (2022-2025)

| Policy | N | ATS% | ROI% | Retention |
|--------|---|------|------|-----------|
| EDGE_BASELINE | 395 | 49.9% | -3.5% | 100% |
| HYBRID_VETO_2 | 372 | 50.3% | -2.7% | 94.2% |

**Vetoed Bets:** 23 games, 43.5% ATS, -17.0% ROI (correctly removed)

**2025 Guardrail:** PASSED (Vetoed 55.6% < Kept 60.2%)

### Rejected Approaches

| Approach | Result | Why Failed |
|----------|--------|------------|
| SP+ Confirm-Only | 45.9% ATS | Catastrophic in 2022 (18.8% ATS) |
| VETO_OPPOSES_2 | Damaged 2025 | Vetoed 62.5% winners in 2025 |
| VETO_OPPOSES_3 | Vetoes winners | Removed profitable bets (54.5% ATS) |
| HYBRID_VETO_3 | Vetoes winners | Removed profitable bets (75% ATS in 2025) |

---

## Overlap/Conflict Report Interpretation

When both List A and List B are generated, an overlap report is produced:

### File: `overlap_engine_primary_vs_phase1_edge_{year}_week{week}.csv`

| Field | Description |
|-------|-------------|
| `game_id` | Unique game identifier |
| `in_engine_primary` | Whether game is in List A |
| `engine_side` | List A recommended side (HOME/AWAY) |
| `engine_ev` | List A expected value |
| `in_phase1_edge` | Whether game is in List B |
| `phase1_side` | List B recommended side |
| `phase1_edge_abs` | List B edge magnitude |
| `veto_applied` | Whether HYBRID_VETO_2 was applied |
| `side_agrees` | Whether both lists agree on side |
| `conflict` | Whether lists DISAGREE on side |
| `recommended_resolution` | Informational guidance |

### Resolution Categories

| Resolution | Meaning |
|------------|---------|
| `CONSENSUS` | Both lists agree - higher confidence |
| `LIST_A_ONLY` | Only EV-based selection triggered |
| `LIST_B_ONLY` | Only edge-based selection triggered |
| `REVIEW` | Lists disagree - manual review needed |

**Important:** The overlap report is informational. List A (ENGINE_EV PRIMARY) is always the official recommendation. Conflicts should be reviewed but List A takes precedence.

---

## Default Behavior Summary

1. **List A (EV-based):** Always emitted, NO SP+ filtering
2. **List B (Edge-based):** Auto-emitted in weeks 1-3, NO SP+ filtering by default
3. **HYBRID_VETO_2:** Available but OFF by default
4. **Overlap Report:** Always generated when both lists exist

---

## Research Freeze Notice

Phase 1 SP+ filtering research is **frozen** as of 2026-02-13. The following conclusions are final unless new seasons provide compelling counter-evidence:

1. **SP+ confirm-only gating is harmful** - Catastrophic performance in 2022, inconsistent across years
2. **SP+ veto approaches are unstable** - Vetoes profitable bets in 2025
3. **HYBRID_VETO_2 is the only viable option** - But marginal improvement (+0.8% ROI) doesn't justify operational complexity
4. **Default stance: SP+ as tagging-only** - Use for monitoring and research, not selection

To change this policy, new backtest evidence must demonstrate:
- Consistent improvement across all years (2022-2025+)
- No regression on 2025 (where baseline is strong)
- Statistical significance at p < 0.05
