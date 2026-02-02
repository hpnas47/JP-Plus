# JP+ Power Ratings Model - Architecture & Documentation

**Last Updated:** February 2, 2026

## Overview

**JP+** is a College Football power ratings model designed for sports betting analysis, inspired by Bill Connelly's SP+. The model generates predicted point spreads for games and compares them against Vegas lines to identify betting opportunities.

### Goals
- Predict game margins with low Mean Absolute Error (MAE)
- Achieve >52% win rate Against The Spread (ATS) for profitable betting
- Identify high-confidence plays where model disagrees significantly with Vegas

---

## Model Architecture

JP+ is built on a foundation model plus an adjustments layer.

### Efficiency Foundation Model (EFM)

The core engine of JP+. Built on play-level efficiency metrics rather than game margins. This approach is more predictive because it measures *how* teams play, not just final scores.

**Key Insight:** "Do not regress on the final score. Regress on the Success Rate per game so that we are measuring efficiency, not just the scoreboard outcome."

#### Components

| Component | Weight | Description |
|-----------|--------|-------------|
| **Success Rate** | 54% | Opponent-adjusted success rate via ridge regression |
| **Explosiveness (IsoPPP)** | 36% | Average EPA on successful plays only |
| **Turnover Margin** | 10% | Per-game turnover differential (see below) |
| **Finishing Drives** | Adjustment | Red zone efficiency regressed toward mean (see below) |

#### Success Rate Definition
- **1st down:** Gain ≥50% of yards needed
- **2nd down:** Gain ≥70% of yards needed
- **3rd/4th down:** Gain 100% of yards needed (first down or TD)

#### Garbage Time Filter (Asymmetric)
Garbage time is handled asymmetrically based on which team is winning:
- **Winning team (offense):** Full weight (1.0x) - they earned the blowout
- **Trailing team (offense):** Down-weighted (0.1x) - garbage time noise

Garbage time thresholds:
- 28+ points in 2nd half (quarters 3-4)
- 21+ points in 3rd quarter
- 14+ points in 4th quarter

This preserves signal from dominant teams (Indiana: 56% SR in garbage time) while filtering noise from trailing teams.

#### Red Zone Regression (Finishing Drives)

Red zone TD rate has high variance - a team at 80% will regress toward the mean (~58%). We apply Bayesian regression:

```
regressed_rate = (observed_TDs + prior_TDs) / (total_trips + prior_strength)
```

With `prior_strength = 10` (equivalent to 10 RZ trips of league-average data), this:
- Pulls extreme rates toward 58% expected
- Has more effect on small samples (early season)
- Trusts actual performance more by late season (150+ RZ plays)

**Why this matters:** Getting TO the red zone is sustainable skill (captured by Success Rate). Scoring TDs IN the red zone has some variance early in the season, but over 15 games becomes a reliable signal of scheme and talent. The prior_strength was reduced from 20 to 10 to better credit elite red zone teams at end of season.

#### Turnover Margin Component

Turnovers are a significant predictor of team success that pure efficiency metrics miss. A team that forces turnovers and protects the ball gains a systematic advantage not fully captured by Success Rate.

**Implementation:**
1. Identify turnover plays from play-by-play data (interceptions, fumble recoveries)
2. Count turnovers forced (defense) and turnovers lost (offense) per team per game
3. Calculate per-game turnover margin: (forced - lost) / games_played
4. Apply Bayesian shrinkage toward 0 (see below)
5. Convert to point value using `POINTS_PER_TURNOVER = 4.5`

**Turnover play types detected** (verified against CFBD API):
- Fumble Recovery (Opponent)
- Pass Interception Return
- Interception
- Fumble Return Touchdown
- Interception Return Touchdown

**Bayesian Shrinkage:** Turnover margin is 50-70% luck (fumble bounces, tipped passes). To prevent overweighting small-sample noise, JP+ applies shrinkage:

```
shrinkage = games_played / (games_played + prior_strength)
shrunk_margin = raw_margin × shrinkage
```

With `turnover_prior_strength = 10`:
- 5 games: keeps 33% of margin (5/15)
- 10 games: keeps 50% of margin (10/20)
- 15 games: keeps 60% of margin (15/25)

This means early-season turnover outliers regress heavily, while sustained end-of-season performance is mostly trusted.

**Why 10% weight?** This mirrors SP+'s approach where turnovers contribute 10% of the overall rating. Higher weights would overfit to turnover luck, while lower weights would miss legitimate ball-security/ball-hawking skill.

**Impact:** Adding the turnover component captures teams like Indiana (2025: +15 margin) and Notre Dame (+17 margin) who create systematic turnover advantages. Ohio State's narrower +3 margin means their efficiency advantage is partially offset by weaker turnover performance.

#### FBS-Only Filtering

The ridge regression training data **excludes plays involving FCS opponents**. When an FBS team plays an FCS opponent (e.g., Indiana vs Indiana State), those plays are removed from the regression entirely.

**Why this matters:**
- FCS teams have too few games against FBS opponents to estimate reliable coefficients
- Including FCS plays pollutes the regression with unreliable team strength estimates
- FCS opponents are handled separately via the FCS Penalty adjustment (+24 pts)

This is implemented in `backtest.py` by filtering plays where both offense and defense are in the FBS teams set before passing to the EFM.

#### Key Files
- `src/models/efficiency_foundation_model.py` - Core EFM implementation
- `src/models/finishing_drives.py` - Red zone efficiency with Bayesian regression
- `src/models/special_teams.py` - Field goal efficiency ratings
- Ridge regression on Success Rate (Y=0/1 success, X=sparse team/opponent IDs)
- Converts efficiency metrics to point equivalents for spread prediction

---

## Adjustments Layer

EFM ratings feed into `SpreadGenerator` which applies additional adjustments:

| Adjustment | Typical Range | Description |
|------------|---------------|-------------|
| **Home Field Advantage** | 1.5-4.0 pts | Team-specific HFA based on stadium environment |
| **Travel** | 0-1.5 pts | Distance and time zone penalties |
| **Altitude** | 0-3 pts | High altitude venues (BYU, Air Force, Colorado) |
| **Situational** | -2 to +2 pts | Lookahead, letdown, rivalry, bye week |
| **FCS Penalty** | +18/+32 pts | Tiered: Elite FCS (+18), Standard FCS (+32) |
| **FG Efficiency** | -1.5 to +1.5 pts | Kicker PAAE differential between teams |

### Home Field Advantage Detail

JP+ uses team-specific HFA values based on stadium environment, crowd intensity, and historical factors:

| Tier | HFA Range | Example Teams |
|------|-----------|---------------|
| Elite | 3.5 - 4.0 | LSU, Alabama, Ohio State, Penn State |
| Strong | 3.0 - 3.25 | Nebraska, Wisconsin, Auburn, Boise State |
| Above Average | 2.75 | Texas, Miami, Virginia Tech, James Madison |
| Conference Default | 2.0 - 2.75 | Varies by conference (see below) |
| Below Average | 2.0 - 2.25 | Maryland, Rutgers, Vanderbilt |
| Weak | 1.5 - 1.75 | Kent State, Akron, Temple, UMass |

**Conference Defaults** (for teams without specific values):
- SEC / Big Ten: 2.75
- Big 12 / ACC / Independents: 2.50
- AAC / Mountain West / Sun Belt: 2.25
- MAC / Conference USA: 2.00

#### Trajectory Modifier

HFA is adjusted for rising or declining programs. A team experiencing sustained improvement will have a more energized home environment, while a declining program may see diminished crowd intensity.

**Timing:** Calculated ONCE at the start of each season using the prior completed year as "recent". Locked in for the whole season—not updated weekly. This reflects that stadium atmosphere builds over years, not weeks.

**Calculation:**
- Compare recent win % (prior 1 year) to baseline win % (3 years before that)
- Scale the difference to a modifier in the range ±0.5 points
- Only apply if the change is meaningful (≥0.1 point adjustment)

| Win % Improvement | HFA Modifier |
|-------------------|--------------|
| +30% or more | +0.5 pts |
| +15% | +0.25 pts |
| 0% (stable) | 0 pts |
| -15% | -0.25 pts |
| -30% or more | -0.5 pts |

**Examples:**
- **Vanderbilt 2024**: Baseline ~25% win rate (2021-2023) → Recent ~58% (2024) = +0.5 modifier → HFA rises from 2.0 to 2.5
- **Indiana 2024**: Similar trajectory, HFA boost for newfound competitive environment
- Declining program: A team falling from 60% to 30% win rate would see HFA penalty

**Natural decay:** No explicit decay is needed. As successful years roll into the 3-year baseline, the improvement gap shrinks automatically. Indiana's +0.5 modifier in 2024-25 will naturally decrease by 2027 as their success becomes the new baseline.

**Conference parity:** Same formula applies to all conferences. Elite G5 programs (Boise State, James Madison) already have elevated base HFA (2.75-3.0), so no special trajectory treatment is needed.

This ensures JP+ captures the reality that home field advantage is not static—it evolves with program trajectory.

### FCS Opponent Penalty (Tiered)

When an FBS team plays an FCS opponent, JP+ applies a tiered penalty based on FCS team quality:

| FCS Tier | Penalty | Examples |
|----------|---------|----------|
| **Elite FCS** | +18 pts | North Dakota State, Montana State, South Dakota State, Sacramento State, Idaho |
| **Standard FCS** | +32 pts | All other FCS teams |

**Why tiered?**
- Analysis of 359 FCS games (2022-2024) showed vastly different margins by FCS quality
- Mean FBS margin vs standard FCS: ~30 points
- Mean FBS margin vs elite FCS: only +2 to +15 points
- Elite FCS teams are FCS playoff regulars or have proven track records vs FBS opponents

**Elite FCS Classification (23 teams):**
Top performers vs FBS (data-driven): Sacramento State, Idaho, Incarnate Word, North Dakota State, William & Mary, Southern Illinois, Holy Cross, Weber State, Fordham, Monmouth, South Dakota State, Montana State, Montana

FCS playoff regulars: Villanova, UC Davis, Eastern Washington, Northern Iowa, Delaware, Richmond, Furman

**Impact (vs flat 24pt penalty):**
- 5+ edge ATS improved from 56.0% → 56.9% (+0.9%)
- 3+ edge ATS improved from 52.4% → 52.5% (+0.1%)

**Note:** This adjustment only affects the ~3% of games involving FCS opponents. FBS vs FBS games are unchanged.

#### Key Files
- `src/predictions/spread_generator.py` - Combines all components
- `src/adjustments/home_field.py` - Team-specific and conference HFA values
- `src/adjustments/travel.py` - Travel distance/timezone
- `src/adjustments/altitude.py` - Altitude adjustment
- `src/adjustments/situational.py` - Situational factors

---

## Preseason Priors

Bayesian blending of preseason expectations with in-season performance.

### Prior Sources (weighted blend)
1. **Previous year's SP+ ratings** (60%) - Regressed toward mean based on returning production
2. **Composite recruiting rankings** (40%) - Talent level

### Returning Production Adjustment

The regression factor for prior year ratings is adjusted based on returning production (% of PPA returning):

| Returning PPA | Regression Factor | Effect |
|---------------|-------------------|--------|
| 100% | 10% | Trust prior rating (same team) |
| 50% | 30% | Baseline regression |
| 0% | 50% | Heavy regression (roster turnover) |

This prevents overvaluing teams that lost key players and undervaluing teams returning most of their production.

### Transfer Portal Adjustment

The returning production metric only captures players who stayed—it doesn't account for incoming transfers. JP+ addresses this by fetching transfer portal data and calculating net production impact:

1. **Fetch transfers** from CFBD API for the upcoming season
2. **Match transfers** to prior-year player usage (PPA contribution)
3. **Calculate net impact**: incoming_ppa - outgoing_ppa for each team
4. **Adjust effective returning production**: base_returning + (net_portal × scale)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `portal_scale` | 0.15 | How much to weight portal impact |
| `portal_cap` | ±15% | Maximum adjustment to returning production |

**Example 2024 Portal Winners:**
- Missouri: +15% (gained significant portal production)
- Washington: +15% (rebuilt through portal after title run losses)
- Notre Dame: +11% (Riley Leonard transfer from Duke)

**Example 2024 Portal Losers:**
- New Mexico State: -12% (lost Diego Pavia to Vanderbilt)
- Arizona State: -11% (roster turnover before Dillingham year 2)

**Match rate:** ~18% of transfers match to prior-year PPA. Unmatched transfers (FCS players, walk-ons) are excluded.

**Backtest impact (2022-2025):** 5+ pt edge improved from 53.3% → 53.7% (+0.4%)

### Coaching Change Regression

When a new head coach arrives at an underperforming team (talent rank > performance rank), JP+ dampens the prior year's "drag" and weights talent more heavily. This captures the reality that a talented team stuck under a bad coach may improve significantly with new leadership.

#### The "Forget Factor"

```
talent_gap = performance_rank - talent_rank  (positive = underperformer)
base_forget = min(0.5, talent_gap / 60)
final_forget = min(0.5, base_forget × coach_pedigree)

prior_weight = 0.6 × (1 - final_forget)
talent_weight = 1 - prior_weight
```

#### Coach Pedigree (Data-Driven)

Pedigree scores are calculated from historical coaching records:
- **Career win %** - Primary driver (+0.25 for 70%+, -0.15 for <45%)
- **P5 HC experience** - +0.03 per year, capped at +0.12
- **Longevity bonus** - +0.03 if 5+ years HC experience

| Tier | Pedigree | Example Coaches |
|------|----------|-----------------|
| Elite | 1.27 - 1.30 | Kiffin (67%, 10 P5 yrs), Kirby Smart (85%), DeBoer (78%) |
| Strong | 1.20 - 1.25 | Cignetti (83% at JMU), Sumrall (78% at Tulane) |
| Above Avg | 1.10 - 1.19 | Rhule (51%, 5 P5 yrs), Venables (56%) |
| Average | 1.05 - 1.09 | Brent Key (53%), Sam Pittman (49%) |
| Neutral | 1.00 | First-time HCs, no record |
| Below Avg | 0.88 - 0.97 | Clark Lea (33%), Jeff Lebby (17%) |

#### Impact Examples

| Scenario | Talent | Perf | Gap | Pedigree | Weight Shift | Rating Δ |
|----------|--------|------|-----|----------|--------------|----------|
| Florida 2025 (Sumrall) | #8 | #45 | 37 | 1.25 | 60/40 → 30/70 | +2 to +4 pts |
| Indiana 2024 (Cignetti) | #50 | #85 | 35 | 1.25 | 60/40 → 30/70 | +2 to +4 pts |
| LSU 2025 (Kiffin) | #6 | #15 | 9 | 1.30 | 60/40 → 48/52 | +0.5 pts |
| Alabama 2024 (DeBoer) | #3 | #5 | 2 | 1.30 | No change | 0 pts |

**Key insight:** First-time HCs (like Brent Key at Georgia Tech, Deion Sanders at Colorado) are **excluded entirely** from the adjustment. We have no basis to predict they'll improve the program, so they receive no boost. The model won't predict their breakout, but it also won't penalize the program.

**Known limitation:** Career win % embeds "opportunity" (better jobs → higher win %), not purely coaching skill. A 65% win rate at Alabama means something different than 65% at Kansas. Ideally we'd normalize by prior team talent, but this adds complexity for a feature with small sample size.

**Backtest validation:** The coaching change adjustment showed neutral impact on MAE/ATS across 2023-2025 (sample of affected games too small). This is a qualitative signal with limited statistical power—included for conceptual completeness but shouldn't be expected to provide measurable ATS lift.

### Decay Schedule
| Week | Preseason Weight | In-Season Weight |
|------|------------------|------------------|
| 1 | 100% | 0% |
| 4 | 62% | 38% |
| 8 | 15% | 85% |
| 12+ | 0% | 100% |

#### Key Files
- `src/models/preseason_priors.py` - Prior calculation, returning production adjustment, and blending

---

## Data Pipeline

### Data Sources
- **College Football Data API (CFBD)** - Games, plays, betting lines, team info
- API client: `src/api/cfbd_client.py`

### Data Flow
```
CFBD API
    │
    ├── Games (scores, neutral site, dates)
    ├── Betting Lines (spreads, totals)
    ├── Play-by-Play (down, distance, yards, PPA)
    ├── Field Goal Plays (distance, made/missed)
    ├── Transfer Portal (player movements, ratings)
    ├── Player Usage (prior-year PPA by player)
    ├── Returning Production (% PPA returning)
    └── Team Info (FBS teams, conferences)
    │
    ▼
Preseason Priors
    │
    ├── Prior SP+ ratings (regressed)
    ├── Talent composite
    ├── Transfer portal net impact
    └── Coaching change adjustment
    │
    ▼
EFM Training
    │
    ├── Success Rate calculation per play
    ├── Garbage time filtering
    ├── Ridge regression for opponent adjustment
    └── FG efficiency calculation (PAAE)
    │
    ▼
SpreadGenerator
    │
    └── Apply adjustments → Predicted Spread
```

---

## Evaluation Results

### JP+ Multi-Year Results (2022-2025)

| Metric | Value |
|--------|-------|
| **MAE** | 12.48 |
| **ATS Record** | ~1260-1195 |
| **ATS %** | 51.3% |

#### ATS by Edge Threshold (vs Opening and Closing Lines)

| Edge | vs Closing Line | vs Opening Line |
|------|-----------------|-----------------|
| **3+ pts** | 52.1% (684-630) | **55.0%** (763-624) |
| **5+ pts** | 56.9% (449-340) | **58.3%** (500-358) |

Opening lines contain more inefficiencies; by closing, sharp money has moved lines toward true value.

#### ATS by Season Phase
| Phase | Record | ATS % |
|-------|--------|-------|
| Weeks 4-6 (early) | 242-252 | 49.0% |
| Weeks 7-10 (mid) | 336-321 | 51.1% |
| Weeks 11+ (late) | 371-327 | 53.2% |

### 2025 Season Results (with Turnover Component)

| Metric | Value |
|--------|-------|
| **MAE** | 12.44 |
| **ATS Record** | 322-310-6 |
| **ATS %** | 50.9% |
| **3+ pt edge** | 53.6% (179-155) |
| **5+ pt edge** | 54.5% (110-92) |

### Key Findings
1. JP+ has significantly better prediction accuracy than margin-based approaches (MAE ~1.5 points lower)
2. Consistent ATS improvement across all four years tested (2022-2025)
3. Red zone regression provides modest but consistent lift (~0.1 MAE, ~1% ATS)
4. Turnover component (10%) with Bayesian shrinkage improves team rankings alignment with consensus
4. FG efficiency adds ~0.6% ATS improvement

---

## File Structure

```
CFB Power Ratings Model/
├── config/
│   └── settings.py              # Configuration management
├── src/
│   ├── api/
│   │   └── cfbd_client.py       # CFBD API client
│   ├── data/
│   │   └── processors.py        # Data processing
│   ├── models/
│   │   ├── efficiency_foundation_model.py  # Core EFM engine
│   │   ├── preseason_priors.py  # Preseason ratings & coaching regression
│   │   ├── special_teams.py     # FG efficiency model
│   │   └── finishing_drives.py  # Red zone efficiency
│   ├── adjustments/
│   │   ├── home_field.py        # Team-specific HFA & trajectory
│   │   ├── travel.py            # Travel adjustments
│   │   ├── altitude.py          # Altitude adjustments
│   │   └── situational.py       # Situational factors
│   └── predictions/
│       ├── spread_generator.py  # Combines all components
│       └── vegas_comparison.py  # Compare to Vegas lines
├── scripts/
│   └── backtest.py              # Walk-forward backtesting
└── docs/
    └── MODEL_ARCHITECTURE.md    # This file
```

---

## Usage

### Running Backtests

```bash
# Standard JP+ backtest (uses optimized defaults)
python scripts/backtest.py --use-efm

# Parameter sweep
python scripts/backtest.py --use-efm --sweep

# Custom parameters
python scripts/backtest.py --years 2024 2025 --use-efm --alpha 50 --fcs-penalty-elite 18 --fcs-penalty-standard 32
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--years` | 2022-2025 | Years to backtest |
| `--start-week` | 4 | First week to predict |
| `--use-efm` | Required | Use JP+ (EFM-based) model - always include this flag |
| `--alpha` | 50.0 | Ridge regularization strength (optimized via sweep) |
| `--hfa` | 2.5 | Base home field advantage in points |
| `--fcs-penalty-elite` | 18.0 | Points added for FBS vs elite FCS |
| `--fcs-penalty-standard` | 32.0 | Points added for FBS vs standard FCS |
| `--no-asymmetric-garbage` | False | Disable asymmetric garbage time (enabled by default) |
| `--no-portal` | False | Disable transfer portal adjustment |
| `--portal-scale` | 0.15 | Weight for transfer portal impact |
| `--sweep` | False | Run parameter grid search |
| `--no-priors` | False | Disable preseason priors |

---

## Key Design Decisions

### 1. Why Efficiency-Based (not Margin-Based)?
- Margins are noisy (turnovers, garbage time, late scores)
- Success Rate is more stable and predictive
- Play-level data captures *how* teams perform, not just outcomes

### 2. Why Ridge Regression?
- Handles opponent adjustment naturally (sparse team IDs as features)
- Regularization prevents overfitting to small samples
- Fast and interpretable

### 3. Why Preseason Priors?
- Early season has insufficient data
- Recruiting rankings and prior performance are predictive
- Bayesian blending smoothly transitions to in-season data

### 4. Double-Counting Prevention
- Base ratings contain ONLY the EFM output + preseason blend
- All adjustments (HFA, FCS penalty, FG efficiency, etc.) applied ONCE at prediction time
- SpreadGenerator is the single point where components combine

---

## Open Items

### Needs Validation
- [x] **EFM alpha parameter sweep** - ✅ DONE. Swept alphas 25-200 across 2022-2025. Optimal: alpha=50 (MAE 12.54, 5+ edge 56.0% vs 55.3% at alpha=100). Updated defaults.

---

## Future Improvements

### High Priority
- [x] **Expose separate O/D/ST ratings** - ✅ DONE. JP+ now exposes separate offensive, defensive, and special teams ratings via `get_offensive_rating()`, `get_defensive_rating()`, `get_special_teams_rating()`, and in `get_ratings_df()` output. This enables future game totals prediction.
- [ ] **Game totals prediction (over/under)** - Formula validated: `Total = 2×Avg + (Off_A + Off_B) - (Def_A + Def_B)`. Each team's expected points: `Team_A_points = Avg + Off_A - Def_B`. Ready to implement.
- [ ] Integrate Finishing Drives component into EFM foundation
- [ ] Add quarterback-specific adjustments for transfers/injuries
- [ ] Improve situational adjustment calibration

### Medium Priority
- [ ] Multi-year backtesting to validate stability
- [ ] Weather impact modeling
- [ ] Expand special teams beyond FG (punting, kickoffs, returns)

### Low Priority
- [ ] Real-time line movement tracking
- [ ] Expected value calculations with Kelly criterion
- [ ] Automated betting recommendations

### Parking Lot (Needs Evidence Before Implementation)
- [ ] **Soft cap on asymmetric garbage time** - Concern: winning team can accumulate unlimited full-weight plays in blowouts, potentially inflating ratings. Proposed fix: decay weight after +35 margin, cap full-weight GT plays per game. **Status:** Theoretically valid but no evidence of actual problem. Test first: are blowout-heavy teams systematically over-rated vs Vegas?
- [ ] **Reduce turnover weight to improve 3+ edge** - Turnovers help 5+ edge (+0.9%) but slightly hurt 3+ edge (-0.2%). Could test 5% weight instead of 10%. **Status:** Tradeoff exists but current 10% matches SP+ and helps high-conviction bets.
- [ ] **Normalize coaching pedigree by prior team talent** - Career win % embeds opportunity (better jobs → higher win %), not purely skill. Normalizing by talent level of teams coached would be more accurate. **Status:** Small sample size (~10-15 coaching changes/year) makes this hard to validate. Current neutral backtest impact suggests feature is already appropriately weighted.

---

## References

### Methodology Inspiration
- **SP+ (Bill Connelly)** - Success Rate + Explosiveness foundation; JP+ naming is an homage to SP+
- FPI (ESPN) - Efficiency-based ratings
- Sagarin - Ridge regression approach

### Key Metrics
- **Success Rate:** % of plays achieving down-specific yardage thresholds
- **IsoPPP (Isolated Points Per Play):** EPA on successful plays only
- **EPA (Expected Points Added):** Point value added by each play

---

## Changelog

### February 2026
- **Added Turnover Margin Component (10%)** - JP+ now includes per-game turnover margin as 10% of the overall rating, matching SP+'s approach. Turnovers are identified from play-by-play data (interceptions, fumble recoveries) and converted to point value using 4.5 points per turnover. The efficiency and explosiveness weights were adjusted from 60/40 to 54/36 to accommodate the new component. **Bayesian shrinkage** (prior_strength=10) regresses turnover margin toward 0 based on games played, preventing overweighting of small-sample luck while trusting sustained performance. This captures teams like Indiana (+15 margin) and Notre Dame (+17 margin) who create systematic turnover advantages that pure efficiency metrics miss.
- **Reduced Red Zone Regression Strength (20→10)** - The prior_strength for RZ TD rate regression was reduced from 20 to 10 to better credit elite RZ teams at end of season. Impact: 5+ edge ATS improved from 54.8% to 55.3%.
- **Added FBS-Only Filtering** - Ridge regression training data now excludes plays involving FCS opponents. FCS teams have insufficient games against FBS opponents to estimate reliable coefficients, so including them pollutes the regression. FCS games are handled separately via the FCS Penalty (+24 pts). Impact: 5+ pt edge improved from 54.0% → 54.8% (+0.8%). This is the cleanest data pipeline for opponent adjustment.
- **Validated Asymmetric Garbage Time** - After implementing FBS-only filtering, re-tested symmetric vs asymmetric GT. Asymmetric remains superior: 5+ edge 54.8% vs 53.0% for symmetric. Asymmetric GT also improves Indiana's ranking (#4 → #2), properly crediting teams that maintain efficiency in blowouts.

### January 2026
- **Added Transfer Portal Integration** - Preseason priors now incorporate net transfer portal impact on rosters. Fetches portal entries from CFBD API, matches transfers to prior-year player usage (PPA), and calculates net incoming - outgoing production for each team. The portal adjustment modifies effective returning production with `portal_scale=0.15` and caps at ±15%. This addresses the gap where returning production only captured players who stayed, not incoming transfers. Impact: 5+ pt edge improved from 53.3% → 53.7% (+0.4%). Added `fetch_transfer_portal()`, `fetch_player_usage()`, `calculate_portal_impact()` to `preseason_priors.py`. New CLI flags: `--no-portal`, `--portal-scale`.
- **Tuned efficiency/explosiveness weights from 65/35 to 60/40** - Comparison with SP+ identified that explosive teams (Ole Miss, Texas Tech) were being underrated with 35% explosiveness weight. Tested weight configurations across 2022-2025: 60/40 showed best multi-year performance (MAE 12.63, ATS 51.3%, 5+ edge 54.5%). This better captures big-play ability while maintaining efficiency as the dominant signal.
- **Exposed separate O/D/ST ratings** - EFM now calculates and exposes separate offensive, defensive, and special teams ratings. The `TeamEFMRating` dataclass includes `offensive_rating`, `defensive_rating`, and `special_teams_rating` fields. New methods: `get_offensive_rating()`, `get_defensive_rating()`, `get_special_teams_rating()`, `set_special_teams_rating()`. The `get_ratings_df()` output now includes offense, defense, and special_teams columns. This enables future game totals prediction by predicting each team's expected points scored.
- **Added FG Efficiency Adjustment** - Integrated field goal Points Above Average Expected (PAAE) into spread predictions. Calculates each team's kicking efficiency vs expected make rates by distance (<30yd: 92%, 30-40: 83%, 40-50: 72%, 50-55: 55%, 55+: 30%). The FG differential is applied as an adjustment to the spread. Impact: ATS improved from 50.6% → 51.2% (+0.6%), MAE improved from 12.57 → 12.47 (-0.10). Added `calculate_fg_ratings_from_plays()` and `get_fg_differential()` to `src/models/special_teams.py`.
- **Added FCS Penalty (+24 pts)** - When FBS teams play FCS opponents, add 24 points to the FBS team's predicted margin. Diagnostic analysis revealed JP+ was under-predicting blowouts by 26 points (99.5% of errors were under-predictions). The penalty directly addresses this: MAE improved from 13.11 → 12.57 (-0.54 pts), 5+ edge ATS improved to 53.2%. Only affects ~3% of games (FCS matchups).
- **Named the overall model JP+** (homage to SP+); EFM remains the core engine
- **Added Returning Production adjustment** - Prior year ratings now regressed based on % of PPA returning. High returning production = less regression, low returning = more regression. Improved ATS from 51.0% to 51.3%.
- **Implemented team-specific HFA** - Replaced flat 2.5 pt HFA with curated team-specific values (1.5-4.0 range) based on stadium environment. Elite venues like LSU (4.0) get more credit; weak environments like UMass (1.5) get less. Conference defaults used for unlisted teams.
- **Added HFA trajectory modifier** - Dynamic adjustment (±0.5 pts) for rising/declining programs. Compares recent win % (1 year) to baseline (prior 3 years). Rising programs like Vanderbilt and Indiana get HFA boost; declining programs get penalty. Captures that home field advantage evolves with program trajectory.
- **Added Coaching Change Regression** - When a new HC arrives at an underperforming team, JP+ dampens prior year drag and weights talent more heavily. Uses data-driven coach pedigree scores (calculated from career win %, P5 experience) to determine how much to "forget" prior underperformance. Elite coaches (Kiffin, Cignetti, Sumrall) at talent-rich programs can see +2 to +4 point rating boosts. First-time HCs receive neutral treatment (no prediction, no penalty).
- Created Efficiency Foundation Model (EFM) as the core engine
- Added garbage time filtering (down-weight blowout plays)
- Fixed double-counting issues in adjustment layer
- Comprehensive documentation created
- **Added Red Zone regression (Finishing Drives)** - Implemented Bayesian regression for red zone TD rate toward league mean (58%). Uses prior_strength=10 to pull extreme rates toward average while trusting actual performance. Multi-year validation showed consistent improvement: MAE improved 0.04-0.10 pts and ATS improved 1-2% across all four years (2022-2025). This is now part of the EFM pipeline.
- **Reduced Red Zone regression strength (prior_strength 20→10)** - Analysis showed the original prior_strength=20 was too aggressive for end-of-season ratings. With 150+ RZ plays per team, elite RZ performance (like Indiana's 87% TD rate) is a genuine skill, not luck. Reducing to 10 better credits teams that sustain elite RZ efficiency over a full season. This fixed an issue where regression was flipping Indiana's RZ advantage over Ohio State.

### Explored but Not Included
- **Turnover-Worthy Plays (TWP) proxy** - Built model using pass breakups and sacks as proxies for interceptable passes and fumble-worthy plays. Multi-year validation showed inconsistent results (helped 2023/2025, hurt 2024). Removed to avoid overfitting.
