# Transfer Portal Code Map - Quick Reference

**File:** `/Users/jason/Documents/CFB Power Ratings Model/src/models/preseason_priors.py`

## Task 1: Portal Ingestion Code Locations

### Data Fetching (API Layer)
| Function | Lines | Purpose |
|----------|-------|---------|
| `fetch_transfer_portal()` | 308-351 | Fetches transfer portal entries from CFBD API |
| `fetch_player_usage()` | 353-381 | Fetches player PPA usage stats |
| `fetch_returning_production()` | 287-306 | Fetches percent_ppa returning (0-1 scale) |

### Portal Impact Calculation
| Function/Section | Lines | Purpose |
|-----------------|-------|---------|
| `calculate_portal_impact()` | 665-853 | Main portal calculation function |
| Outgoing value calculation | 730-742 | No discount, full value for players leaving |
| Incoming value calculation | 744-756 | WITH level-up discount for G5→P4 |
| Continuity tax application | 765-766 | Amplifies outgoing losses by ~11% |
| Net impact per team | 805-823 | `incoming - outgoing × portal_scale × churn_penalty` |

### G5→P4 Transfer Discounts
| Constant/Function | Lines | Value/Purpose |
|-------------------|-------|---------------|
| `PHYSICALITY_TAX` | 454 | 0.75 (25% discount for trench players) |
| `ATHLETICISM_DISCOUNT` | 455 | 0.90 (10% discount for skill players) |
| `_get_level_up_discount()` | 500-538 | Returns discount multiplier by position |
| `HIGH_CONTACT_POSITIONS` | 448 | OT, IOL, IDL, LB, EDGE |
| `SKILL_POSITIONS` | 451 | WR, RB, CB, S |

### Portal Scale Application
| Location | Lines | Value/Purpose |
|----------|-------|---------------|
| In `calculate_preseason_ratings()` | 1155-1163 | Portal adjustment added to returning production |
| Default portal_scale | 668 | 0.15 (15% weight) |
| Impact cap | 670 | ±0.12 (±12% maximum) |

### Priors Decay Over Season
| Function/Section | Lines | Purpose |
|-----------------|-------|---------|
| `blend_with_inseason()` | 1338-1406 | Blends preseason priors with in-season ratings |
| Week 0 weight | 1366-1367 | 95% prior weight |
| Week 9+ weight | 1369-1371 | 5% prior weight |
| Talent floor persistence | 1394-1403 | 8% all year (uses talent_rating_normalized) |
| Talent floor weight param | 1343, 1359 | Default 0.08 (8%) |

### Continuity Tax (Existing Churn Logic)
| Constant/Location | Line | Value/Purpose |
|-------------------|------|---------------|
| `CONTINUITY_TAX` | 459 | 0.90 (constant) |
| Application | 766 | `outgoing / 0.90` (amplifies losses) |
| Rationale comment | 456-458 | "Hidden cost of replacing incumbents" |

---

## Task 2: calculate_roster_churn_penalty() Implementation

### New Function
| Component | Lines | Details |
|-----------|-------|---------|
| Function definition | 615-674 | `calculate_roster_churn_penalty()` |
| Docstring | 616-653 | Explains Talent Mirage hypothesis |
| Continuity score calculation | 655-658 | `0.7 × ret_pct + 0.3 × (1 - portal_pct)` |
| Sigmoid mapping | 660-674 | Maps continuity → penalty coefficient (0.7-1.0) |

### Integration Points
| Location | Lines | Purpose |
|----------|-------|---------|
| Constructor toggle | 229 | Added `use_churn_penalty: bool = False` parameter |
| Instance variable | 242 | Stored as `self.use_churn_penalty` |
| Portal additions count | 775 | Count transfers per team for churn calc |
| Churn penalty calculation | 807-823 | Apply penalty if `use_churn_penalty == True` |
| Logging | 825-831 | Log significant penalties (>5% reduction) |
| Pass returning_production | 1158-1160 | In `calculate_preseason_ratings()` |

### Reference Points (Churn Penalty Behavior)
| Scenario | Continuity Score | Penalty Coefficient | Interpretation |
|----------|-----------------|---------------------|----------------|
| High continuity (60% ret + 15 portal) | ~0.75 | ~1.0 | No penalty |
| Typical churn (50% ret + 20 portal) | ~0.50 | ~0.85 | Mild penalty |
| Heavy churn (40% ret + 25 portal) | ~0.35 | ~0.75 | Moderate penalty |
| Extreme churn (30% ret + 30 portal) | ~0.25 | ~0.70 | Heavy penalty (floor) |

---

## Testing Workflow

### Phase 1: Unit Test Churn Penalty Math
```python
from src.models.preseason_priors import PreseasonPriors

priors = PreseasonPriors(client=None, use_churn_penalty=True)

# Test reference points
test_cases = [
    (0.70, 10, (0.95, 1.00)),  # High continuity → no penalty
    (0.50, 20, (0.80, 0.90)),  # Typical churn → mild penalty
    (0.30, 30, (0.70, 0.75)),  # Heavy churn → heavy penalty
]

for ret_pct, portal_adds, (min_exp, max_exp) in test_cases:
    penalty = priors.calculate_roster_churn_penalty(ret_pct, portal_adds)
    assert min_exp <= penalty <= max_exp
    print(f"ret={ret_pct:.0%}, adds={portal_adds} → penalty={penalty:.3f}")
```

### Phase 2: Backtest Comparison
1. Run baseline: `python3 scripts/backtest.py --start-week 1`
2. Edit backtest.py line ~1972 to enable churn penalty:
   ```python
   priors_calculator = PreseasonPriors(
       client=client,
       use_churn_penalty=True,  # ADD THIS LINE
   )
   ```
3. Run comparison: `python3 scripts/backtest.py --start-week 1`
4. Compare: MAE, ATS%, 3+ Edge, 5+ Edge

### Phase 3: FSU 2024 Case Study
Extract FSU's 2024 preseason rating with and without churn penalty:
- FSU 2024: ~35% returning production, ~30 portal additions
- Expected continuity score: ~0.44
- Expected penalty: ~0.82 (18% reduction in portal impact)
- Net effect: ~-0.2 rating points

---

## Expected Outcomes

### If Hypothesis is Correct:
- MAE: Decrease (better accuracy)
- ATS%: Increase, especially 5+ Edge
- Portal-heavy teams (FSU, Colorado): Lower preseason ratings
- Continuity teams (Georgia, OSU): Relatively unchanged

### If Hypothesis is Wrong:
- MAE: Increase (worse accuracy)
- ATS%: Decrease
- Recommendation: Revert changes

---

**Full Investigation Report:** `docs/PORTAL_CHURN_INVESTIGATION.md`
