# Transfer Portal "Talent Mirage" Investigation

## Investigation Date: 2026-02-06

## Hypothesis
Portal-heavy teams (like FSU 2024) are overvalued because the model treats incoming transfers too generously. The "Talent Mirage" occurs when:
- Teams replace experienced players with highly-rated transfers
- Transfers face a scheme learning curve, chemistry gaps, and leadership vacuum
- Raw talent doesn't translate immediately to on-field production

## Task 1: Portal Ingestion Code Locations

### Data Ingestion (API Layer)
- **Lines 308-351**: `fetch_transfer_portal()` - Fetches transfer portal entries from CFBD API
  - Returns DataFrame with player name, position, origin, destination, stars, rating
  - Handles nested objects (origin/destination can be str or object)
- **Lines 353-381**: `fetch_player_usage()` - Fetches player PPA usage stats (for incumbent value)
- **Lines 287-306**: `fetch_returning_production()` - Fetches percent_ppa returning (0-1 scale)

### Portal Impact Calculation (Lines 665-853)
- **Lines 665-821**: `calculate_portal_impact()` - Main portal calculation function
  - Uses scarcity-based position weights (QB=1.00, OT=0.90, RB=0.40, etc.)
  - Applies level-up discounts for G5→P4 transfers
  - Applies continuity tax to outgoing losses

- **Lines 730-742**: Calculates OUTGOING value (no discount, full value)
  - `outgoing_value` = position_weight × quality_factor × 1.0

- **Lines 744-756**: Calculates INCOMING value (WITH level-up discount)
  - `incoming_value` = position_weight × quality_factor × level_discount

- **Lines 765-766**: Applies CONTINUITY_TAX to amplify outgoing losses
  - `outgoing = outgoing_raw / 0.90` (dividing by <1.0 makes losses bigger)
  - Reflects hidden cost of replacing incumbents (scheme fit, leadership)

- **Lines 805-809**: Calculates net impact per team
  - `net_val = incoming - outgoing`
  - `scaled_impact = net_val × portal_scale × churn_penalty` (churn_penalty is NEW)
  - Capped at ±12%

### G5→P4 Transfer Discounts (Lines 500-538)
- **Lines 528-531**: PHYSICALITY_TAX = 0.75 (25% discount for trench players)
  - HIGH_CONTACT_POSITIONS: OT, IOL, IDL, LB, EDGE
- **Lines 532-535**: ATHLETICISM_DISCOUNT = 0.90 (10% discount for skill players)
  - SKILL_POSITIONS: WR, RB, CB, S
- **Lines 524-525**: P4→G5 boost = 1.10 (10% boost for "stepping down")

### Portal Scale Application
- **Lines 1155-1163**: In `calculate_preseason_ratings()` - portal adjustment added to returning production
  - `effective_ret_ppa = max(0.0, min(1.0, ret_ppa + portal_adj))`
- **Lines 805-807**: portal_scale = 0.15 (default) applied to net value, capped at ±12%

### Priors Decay Over Season (Lines 1338-1406)
- **Lines 1338-1406**: `blend_with_inseason()` - Blends preseason priors with in-season ratings
  - Non-linear fade curve (sigmoid-style)
  - Week 0: 95% prior weight
  - Week 4-5: ~50% prior weight (tipping point)
  - Week 9+: 5% prior weight

- **Lines 1366-1367**: Week 0 = 95% prior weight (adjusted for talent floor)
- **Lines 1369-1371**: Week 9+ = 5% prior weight + 8% talent floor
- **Lines 1394-1403**: Talent rating persists all year at `talent_floor_weight` (default 8%)
  - Uses `talent_rating_normalized` (z-score × 12 on SP+ scale)
  - Prevents elite-talent teams from dropping too far with bad performance

### Continuity Tax (Existing Churn Logic)
- **Line 459**: CONTINUITY_TAX = 0.90 (constant)
- **Line 766**: Applied by dividing outgoing value (amplifies losses by ~11%)
- **Comment lines 456-458**: "Reflects hidden cost of replacing incumbents (scheme fit, leadership)"

---

## Task 2: calculate_roster_churn_penalty() Implementation

### Function Location
- **Lines 615-674**: New function `calculate_roster_churn_penalty()`

### Function Signature
```python
def calculate_roster_churn_penalty(
    self,
    returning_production_pct: float,
    portal_additions: int,
    roster_size: int = 85,
) -> float:
```

### How It Works

1. **Inputs:**
   - `returning_production_pct`: Percentage of prior-year production returning (0-1)
   - `portal_additions`: Number of incoming portal transfers
   - `roster_size`: Total roster spots (default 85)

2. **Continuity Score Calculation:**
   - `continuity_score = 0.7 × ret_pct + 0.3 × (1.0 - portal_pct)`
   - Weights returning production heavily (70%) since it captures actual PPA returning
   - Weights portal churn moderately (30%) since some churn is normal/healthy
   - Higher score = more stable roster

3. **Penalty Coefficient (Sigmoid Mapping):**
   - Uses sigmoid curve for smooth transitions
   - Formula: `penalty = 0.70 + 0.30 / (1 + exp(-8 × (continuity - 0.50)))`
   - Reference points:
     - continuity = 0.75 (60% ret + 15 portal) → penalty ≈ 1.0 (no penalty)
     - continuity = 0.50 (typical churn) → penalty ≈ 0.85
     - continuity = 0.35 (40% ret + 25 portal) → penalty ≈ 0.75
     - continuity = 0.25 (heavy churn) → penalty ≈ 0.70 (floor)

4. **Application:**
   - Only applied to POSITIVE portal impact (gains, not losses)
   - Rationale: Losses already hurt via continuity tax; no need to double-penalize
   - Integrated at line 807: `scaled_impact = net_val × portal_scale × churn_penalty`

### Toggle Mechanism
- **Line 229**: Added `use_churn_penalty: bool = False` parameter to `PreseasonPriors.__init__()`
- **Line 242**: Stored as instance variable `self.use_churn_penalty`
- **Lines 807-823**: Churn penalty only calculated if `self.use_churn_penalty == True`
- **Default: False** - Must be explicitly enabled for A/B testing

### Integration Points
- **Lines 665-853**: Modified `calculate_portal_impact()` to:
  - Accept `returning_production` parameter (line 674)
  - Count portal additions per team (line 775)
  - Calculate churn penalty per team (lines 807-823)
  - Log significant penalties (lines 825-831)
- **Lines 1154-1161**: Modified `calculate_preseason_ratings()` to:
  - Pass `returning_production=returning_prod` to `calculate_portal_impact()` (line 1159)

---

## Testing Plan

### Phase 1: Validate Churn Penalty Math
```python
from src.models.preseason_priors import PreseasonPriors

# Create instance (doesn't need real client for this test)
priors = PreseasonPriors(client=None, use_churn_penalty=True)

# Test reference points
test_cases = [
    # (ret_pct, portal_adds, expected_penalty_range)
    (0.70, 10, (0.95, 1.00)),  # High continuity → no penalty
    (0.50, 20, (0.80, 0.90)),  # Typical churn → mild penalty
    (0.30, 30, (0.70, 0.75)),  # Heavy churn → heavy penalty
]

for ret_pct, portal_adds, (min_exp, max_exp) in test_cases:
    penalty = priors.calculate_roster_churn_penalty(ret_pct, portal_adds)
    assert min_exp <= penalty <= max_exp, f"Failed for ret={ret_pct}, adds={portal_adds}"
    print(f"ret={ret_pct:.0%}, adds={portal_adds} → penalty={penalty:.3f}")
```

### Phase 2: Backtest Comparison (2022-2025)
Run two backtests and compare:

#### Baseline (Current Model)
```bash
python3 scripts/backtest.py --start-week 1
# Report: MAE, ATS%, 3+ Edge, 5+ Edge
```

#### With Churn Penalty Enabled
Modify backtest.py to pass `use_churn_penalty=True`:
```python
# In backtest.py, line ~300 (where PreseasonPriors is instantiated)
priors_calculator = PreseasonPriors(
    client=client,
    use_churn_penalty=True,  # ADD THIS LINE
)
```

Run backtest again:
```bash
python3 scripts/backtest.py --start-week 1
# Compare: Did MAE improve? Did ATS% improve? Did 5+ Edge improve?
```

### Phase 3: FSU 2024 Case Study
Extract FSU's 2024 preseason rating with and without churn penalty:

```python
# FSU 2024 Stats (example - verify actual numbers):
# - Returning production: ~35% (low)
# - Portal additions: ~30 transfers (high)
# - Expected continuity score: 0.7 × 0.35 + 0.3 × (1 - 30/85) ≈ 0.44
# - Expected penalty: ~0.82 (18% reduction in portal impact)

# If FSU's net portal value was +8.0 with portal_scale=0.15:
# Baseline impact: +8.0 × 0.15 = +1.2 rating points
# With churn penalty: +8.0 × 0.15 × 0.82 ≈ +1.0 rating points
# Net effect: -0.2 rating points (brings FSU down closer to reality)
```

---

## Expected Outcomes

### If Hypothesis is Correct:
- **MAE Improvement**: Should decrease (better accuracy)
- **ATS% Improvement**: Should increase, especially for 5+ Edge category
- **Portal-Heavy Teams**: FSU, Colorado, etc. should have lower preseason ratings
- **Continuity Teams**: Georgia, Ohio State (high returning production) should be relatively unchanged

### If Hypothesis is Wrong:
- **MAE Regression**: Increase (worse accuracy)
- **ATS% Regression**: Decrease
- **Recommendation**: Revert changes, investigate other factors (e.g., talent floor decay too slow)

---

## Files Modified
- `/Users/jason/Documents/CFB Power Ratings Model/src/models/preseason_priors.py`
  - Line 229: Added `use_churn_penalty` parameter to constructor
  - Lines 615-674: Added `calculate_roster_churn_penalty()` function
  - Lines 665-853: Modified `calculate_portal_impact()` to apply churn penalty
  - Lines 1154-1161: Modified `calculate_preseason_ratings()` to pass returning_production

---

## Outcome: REJECTED (2026-02-06)

### Chemistry Tax — 3-0 Council Vote to Reject

The churn penalty was evaluated by the full Audit Council as the "Chemistry Tax" proposal: a -3.0 pt penalty for teams with <50% Returning Production in Week 1, decaying linearly to 0 by Week 5.

**Council Findings:**

1. **Model Strategist (REJECT — Redundant):** Three existing mechanisms already penalize low-RetProd teams 5-8 pts: RetProd regression, talent decay (0.08→0.03), and prior fade. Adding a 4th layer is redundancy rot.

2. **Code Auditor (REJECT — Architecture):** The <50% RetProd threshold captures 46-56% of FBS teams — a median split, not an outlier detector. Any "penalty" applied to half the league is noise, not signal.

3. **Quant Auditor (REJECT — No Signal):** Observed bias for low-RetProd teams was only +1.28 pts (t=0.87, not statistically significant). The -3.0 tax overcorrects to -1.72 pts in the wrong direction. Signal reverses by Week 4; the 50-70% RetProd group actually has 57.5% ATS 3+ edge — penalizing them would destroy real edge.

**Key Lessons:**
- 14.94 early-season MAE is a **sample-size problem** (608 games), not a fixable bias.
- The churn penalty infrastructure is preserved (`use_churn_penalty=False` by default) but should not be activated.
- This was the 5th consecutive rejection in a pattern of trying to surgically fix 1-3 team anomalies (MOV → Fraud Tax → Chemistry Tax → Zombie Prior → Talent Abandonment). All fail the same way: any game-outcome or roster-composition signal either (a) is already captured by existing mechanisms or (b) degrades ATS edge.

### Infrastructure Status
- `calculate_roster_churn_penalty()` remains in `preseason_priors.py` (lines 615-674)
- Toggle: `use_churn_penalty=False` (default, dormant)
- Temporal talent decay (`0.08→0.03`) was APPROVED and is active (separate from this rejection)
