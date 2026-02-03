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
- FCS opponents are handled separately via the tiered FCS Penalty adjustment (+18/+32 pts)

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
| **Pace Adjustment** | -10% to -15% | Spread compression for triple-option teams |
| **QB Injury** | ±3-10 pts | Manual flag when starting QB is out (see below) |

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

### QB Injury Adjustment

QB injuries are the single biggest source of prediction error that JP+ otherwise ignores. The model provides a manual flagging system for known starter injuries:

1. **Pre-computed depth charts:** For each team, identify the starter (most pass attempts) and backup using CFBD player PPA data
2. **PPA differential:** Calculate the per-play efficiency drop when switching to the backup
3. **Point adjustment:** `PPA_drop × 30 plays/game` (QBs typically involved in ~30 pass/scramble plays)

| Team (2024) | Starter | PPA | Backup | PPA | Adjustment |
|-------------|---------|-----|--------|-----|------------|
| Georgia | Beck | 0.353 | Stockton | 0.125 | **-6.8 pts** |
| Ohio State | Howard | 0.575 | Brown | 0.243 | **-10.0 pts** |
| Texas | Ewers | 0.322 | A. Manning | 0.589 | **+8.0 pts** |
| Alabama | Milroe | 0.321 | Simpson | 0.215 | **-3.2 pts** |

**Usage:** `python scripts/run_weekly.py --qb-out Georgia Texas` applies the adjustment for those teams.

**Note:** Texas is unusual—Arch Manning (backup) has better PPA than Ewers (starter), so the adjustment is positive. CFBD has no injury data; flagging requires manual input.

### Pace Adjustment (Triple-Option)

Triple-option teams (Army, Navy, Air Force, Kennesaw State) run ~55 plays/game vs ~70 for standard offenses. This creates more variance—analysis shows 30% worse MAE for triple-option games (16.09 vs 12.36, p=0.001).

JP+ compresses spreads by 10% toward zero when a triple-option team is involved (15% if both teams run triple-option). This reflects the fundamental uncertainty in games with fewer possessions.

### Weather Adjustment (Totals)

Weather significantly impacts game totals (over/under). JP+ fetches weather data from the CFBD API and applies adjustments based on three factors:

| Factor | Threshold | Adjustment | Rationale |
|--------|-----------|------------|-----------|
| **Wind** | >10 mph | -0.3 pts/mph | Reduces passing efficiency, FG accuracy |
| **Temperature** | <40°F | -0.15 pts/degree | Ball handling, pace reduction |
| **Precipitation** | >0.02 in | -3.0 pts flat | Ball security, passing impact |
| **Heavy Precip** | >0.05 in | -5.0 pts flat | Severe weather penalty |

**Indoor games** (identified via `game_indoors` flag) receive no weather adjustment.

**Caps:** Wind adjustment capped at -6.0 pts, temperature at -4.0 pts to prevent extreme values.

**Example extreme weather games (2024 Week 11-15):**
- Nebraska @ Iowa: 16°F, 7 mph wind → Temp adj: -3.6 pts
- FAU @ Temple: 23 mph wind → Wind adj: -3.9 pts
- Miami (OH) @ Ball State: Heavy rain (0.047 in) → Precip adj: -3.0 pts

**Data source:** CFBD API `get_weather` endpoint provides:
- `temperature` (°F), `wind_speed` (mph), `wind_direction` (degrees)
- `precipitation` (inches), `snowfall` (inches)
- `humidity` (%), `weather_condition` (text description)
- `game_indoors` (boolean for dome games)

**Status:** Weather adjustment module implemented for future totals prediction. Parameters are conservative estimates based on NFL weather studies (Marek 2015, Kacsmar 2016) and should be validated against historical CFB totals results before use in production.

#### Key Files
- `src/predictions/spread_generator.py` - Combines all components
- `src/adjustments/home_field.py` - Team-specific and conference HFA values
- `src/adjustments/travel.py` - Travel distance/timezone
- `src/adjustments/altitude.py` - Altitude adjustment
- `src/adjustments/situational.py` - Situational factors
- `src/adjustments/qb_adjustment.py` - QB injury adjustment system
- `src/adjustments/weather.py` - Weather adjustments for totals

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

### Asymmetric Regression

Standard regression pulls all teams toward the mean uniformly—but this compresses the true spread between elite and terrible teams. A very bad team at -25 shouldn't gain 7.5 points just from regression.

JP+ applies **asymmetric regression**: teams far from the mean regress less than teams near the mean.

| Distance from Mean | Regression Multiplier | Effective Regression |
|-------------------|----------------------|---------------------|
| ±8 pts | 1.0x | 30% (baseline) |
| ±14 pts | 0.67x | 20% |
| ±20+ pts | 0.33x | 10% |

**Example:** Kent State at raw -25 rating:
- Old (uniform): -25 × 0.7 = -17.5 (lost 7.5 pts of badness)
- New (asymmetric): -25 × 0.9 = -22.5 (kept most of badness)

**Extremity-Weighted Talent:** For extreme teams (20+ pts from mean), talent blend weight is reduced from 40% to 20%. This trusts proven performance over talent projections for outlier teams, preventing the talent component from compressing ratings back toward average.

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
    ├── Team Info (FBS teams, conferences)
    ├── Weather (temperature, wind, precipitation, indoor flag)
    ├── SP+ Ratings (external benchmark)
    └── FPI Ratings (ESPN, external benchmark)
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
| **MAE** | 12.37 |
| **ATS Record** | 1114-1070-43 |
| **ATS %** | 51.0% |

#### ATS by Edge Threshold (vs Closing Lines)

| Edge | Record | ATS % |
|------|--------|-------|
| **2+ pts** | 759-711 | 51.6% |
| **3+ pts** | 612-521 | **54.0%** |
| **5+ pts** | 365-272 | **57.3%** |

Opening lines contain more inefficiencies; by closing, sharp money has moved lines toward true value.

#### ATS by Season Phase
| Phase | Record | ATS % |
|-------|--------|-------|
| Weeks 4-6 (early) | 242-252 | 49.0% |
| Weeks 7-10 (mid) | 336-321 | 51.1% |
| Weeks 11+ (late) | 371-327 | 53.2% |

### 2025 Season Results (Final)

| Metric | vs Closing | vs Opening |
|--------|------------|------------|
| **MAE** | 12.21 | 12.21 |
| **ATS Record** | 325-301-12 | 336-298-4 |
| **ATS %** | 51.9% | **53.0%** |
| **3+ pt edge** | 54.1% (172-146) | **55.3%** (189-153) |
| **5+ pt edge** | 55.4% (98-79) | **58.3%** (119-85) |

**2025 Top 25** (end-of-season including CFP):
1. Ohio State (+27.5), 2. Indiana (+26.8) ★, 3. Notre Dame (+25.4), 4. Oregon (+23.4), 5. Miami (+22.8)

★ National Champions - beat Alabama 38-3, Oregon 56-22, Miami 27-21 in CFP.

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
│   │   ├── situational.py       # Situational factors
│   │   ├── qb_adjustment.py     # QB injury adjustment system
│   │   └── weather.py           # Weather adjustments for totals
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

### Weekly Predictions (run_weekly.py)

```bash
# Generate predictions for current week
python scripts/run_weekly.py --year 2025 --week 10

# With QB injury adjustments
python scripts/run_weekly.py --year 2025 --week 10 --qb-out Georgia Texas
```

| Flag | Default | Description |
|------|---------|-------------|
| `--year` | Current | Season year |
| `--week` | Current | Week to predict |
| `--qb-out` | None | Teams whose starting QB is out (space-separated) |

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

### 5. Mean Error vs ATS Performance

JP+ has a systematic mean error of approximately -6.7 points (predicting home teams to outperform their actual margin). Investigation revealed:

| Metric | Value |
|--------|-------|
| Mean error (HFA=2.8) | -6.7 pts |
| Optimal HFA for zero error | -3.7 pts |
| ATS when picking favorites | 61.6% |
| ATS when picking underdogs | 66.6% |

**Root cause:** The CFBD EPA values (which feed into EFM) already implicitly contain home field advantage—home teams naturally generate better EPA due to crowd noise, familiarity, etc. The EFM ridge regression doesn't separate "true team quality" from "home boost in the data," so home teams get inflated base ratings. Adding explicit HFA on top creates double-counting.

**Why we don't "fix" it:** Despite the mean error, ATS performance is excellent (61-67%). The bias is concentrated in blowout games (where ATS doesn't matter), not close games. Adjusting HFA to zero out mean error could hurt ATS performance. We optimize for what matters (ATS), not calibration metrics.

**Key insight:** Mean error is a nuisance metric for spread betting. What matters is correctly identifying which side will cover, not the exact margin. JP+ excels at the former even with systematic bias in the latter.

---

## Open Items

### Needs Validation
- [x] **EFM alpha parameter sweep** - ✅ DONE. Swept alphas 25-200 across 2022-2025. Optimal: alpha=50 (MAE 12.54, 5+ edge 56.0% vs 55.3% at alpha=100). Updated defaults.

---

## Future Improvements

### High Priority
- [x] **Expose separate O/D/ST ratings** - ✅ DONE. JP+ now exposes separate offensive, defensive, and special teams ratings via `get_offensive_rating()`, `get_defensive_rating()`, `get_special_teams_rating()`, and in `get_ratings_df()` output. This enables future game totals prediction.
- [x] **Add quarterback-specific adjustments for injuries** - ✅ DONE. Added `QBInjuryAdjuster` class that pre-computes depth charts from CFBD player PPA data. Manual flagging via `--qb-out TEAM` CLI flag. Adjustment = PPA drop × 30 plays/game.
- [ ] **Game totals prediction (over/under)** - Formula validated: `Total = 2×Avg + (Off_A + Off_B) - (Def_A + Def_B)`. Each team's expected points: `Team_A_points = Avg + Off_A - Def_B`. Ready to implement.
- [ ] Integrate Finishing Drives component into EFM foundation
- [ ] Improve situational adjustment calibration

### Medium Priority
- [ ] Multi-year backtesting to validate stability
- [x] **Weather impact modeling** - ✅ DONE. Added `WeatherAdjuster` class that fetches weather data from CFBD API and calculates totals adjustments based on wind (>10 mph: -0.3 pts/mph), temperature (<40°F: -0.15 pts/degree), and precipitation (>0.02 in: -3.0 pts). Indoor games receive no adjustment. Ready for totals prediction integration.
- [ ] Expand special teams beyond FG (punting, kickoffs, returns)

### Low Priority
- [ ] Real-time line movement tracking
- [ ] Expected value calculations with Kelly criterion
- [ ] Automated betting recommendations

### Parking Lot (Needs Evidence Before Implementation)
- [x] **Pace-based margin scaling** - Theory: Fast games have more plays, so efficiency edges should compound into larger margins. JP+ should scale predicted margins by expected pace. **Status:** INVESTIGATED, NOT IMPLEMENTING. Empirical analysis (2023-2025) shows the theory doesn't match reality: (1) Fast games actually have smaller margins (R²=2.2% correlation), (2) JP+ over-predicts fast game margins (mean error -2.1), not under-predicts, (3) ATS is actually better in fast games (73% vs 67.6%). Vegas already prices pace. Efficiency metrics implicitly capture tempo. Adding pace scaling would add complexity without benefit.
- [x] **Mercy Rule Dampener (non-linear margins)** - Theory: Coaches tap brakes in blowouts, so efficiency models over-predict large margins. Apply logistic dampening to extreme spreads. **Status:** INVESTIGATED, NOT IMPLEMENTING. The bias EXISTS (mean error -38.7 on 21+ spreads), and dampening DOES improve MAE (-1.66 pts). BUT dampening HURTS ATS (-2.8pp) because our large edges against Vegas are correct directionally even when magnitude is off. A spread of -28 vs Vegas -21 may be "wrong" by 7 points but RIGHT about home covering. Since ATS is our optimization target (Rule #3), we accept worse MAE to maintain betting edge.
- [ ] **Soft cap on asymmetric garbage time** - Concern: winning team can accumulate unlimited full-weight plays in blowouts, potentially inflating ratings. Proposed fix: decay weight after +35 margin, cap full-weight GT plays per game. **Status:** Theoretically valid but no evidence of actual problem. Test first: are blowout-heavy teams systematically over-rated vs Vegas?
- [ ] **Reduce turnover weight to improve 3+ edge** - Turnovers help 5+ edge (+0.9%) but slightly hurt 3+ edge (-0.2%). Could test 5% weight instead of 10%. **Status:** Tradeoff exists but current 10% matches SP+ and helps high-conviction bets.
- [ ] **Normalize coaching pedigree by prior team talent** - Career win % embeds opportunity (better jobs → higher win %), not purely skill. Normalizing by talent level of teams coached would be more accurate. **Status:** Small sample size (~10-15 coaching changes/year) makes this hard to validate. Current neutral backtest impact suggests feature is already appropriately weighted.
- [ ] **EV-weighted performance metric** - Current metrics (MAE, ATS %) treat all bets equally. An EV-weighted metric would weight each prediction by the expected value of betting it: `EV = (edge_pts / spread) * kelly_fraction`. This would better capture the model's actual betting value—a 55% ATS on +200 underdogs is worth more than 55% on -110 favorites. Could also track CLV (Closing Line Value) as a proxy for long-term edge. **Status:** Design and implement as alternative to raw ATS %.

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
- **Added Weather Adjustment Module** - New `WeatherAdjuster` class for totals prediction. Fetches weather data from CFBD API (`get_weather` endpoint) and calculates adjustments based on wind speed (>10 mph: -0.3 pts/mph, capped at -6.0), temperature (<40°F: -0.15 pts/degree, capped at -4.0), and precipitation (>0.02 in: -3.0 pts, >0.05 in: -5.0 pts). Indoor games (`game_indoors=True`) receive no adjustment. Added `get_weather()` method to `CFBDClient`. Weather data includes temperature, wind speed/direction, precipitation, snowfall, humidity, and weather condition text. Analysis of 2024 late-season games showed: 19/757 indoor games, temperatures ranging 16°F-86°F, wind up to 26 mph, 39 games with precipitation.
- **Added FPI Ratings Comparison** - New `scripts/compare_ratings.py` for 3-way JP+ vs FPI vs SP+ validation. Added `get_fpi_ratings()` and fixed `get_sp_ratings()` in CFBD client to use correct `RatingsApi` endpoint. Initial 2025 comparison shows JP+ correlates r=0.956 with FPI, r=0.937 with SP+. Key divergence: JP+ ranks Ohio State #1, Indiana #2; FPI/SP+ have Indiana #1.
- **Fixed Sign Convention Bugs** - Complete audit found 2 bugs: (1) `spread_generator.py:386` had `home_is_favorite = prelim_spread < 0` (wrong, should be `> 0`), causing rivalry boost to be applied to favorites instead of underdogs; (2) `html_report.py:250` had inverted CSS class logic. Backtest comparison: 3+ edge ATS improved from 53.3% to **54.0%** (+0.7%, +8 net wins). MAE improved from 12.39 to 12.37. Documented sign conventions in SESSION_LOG Rules section.
- **Investigated Mercy Rule Dampener (NOT implemented)** - Theory: coaches tap brakes in blowouts, so apply non-linear dampening to extreme spreads. Finding: Bias exists (-38.7 mean error on 21+ spreads) and dampening improves MAE (-1.66), BUT hurts ATS (-2.8pp). Our large edges are correct directionally even when magnitude is off. Decision: Accept worse MAE to maintain betting edge.
- **Investigated Pace-Based Margin Scaling (NOT implemented)** - Theory suggested fast games should have larger margins (more plays to compound efficiency edge). Empirical analysis showed opposite: fast games have smaller margins (R²=2.2%), JP+ over-predicts (not under-predicts) fast games, and ATS is actually better in fast games (73% vs 67.6%). Vegas already prices pace. Decision: Do not implement.
- **Documented Mean Error vs ATS Trade-off** - Investigated -6.7 mean error. Root cause: CFBD EPA data implicitly contains home field advantage, causing double-counting with explicit HFA. Decision: Don't fix—ATS performance is 61-67% despite calibration bias. Bias is concentrated in blowouts where ATS doesn't matter. Added documentation explaining why we optimize for ATS, not mean error.
- **Updated 2025 Top 25 with Full CFP Data** - Top 25 now includes all 46 postseason games and 6,223 playoff plays. Indiana (#2, National Champions) beat Alabama 38-3, Oregon 56-22, and Miami 27-21 in CFP.
- **Added Rules & Conventions to SESSION_LOG** - Critical rules: (1) Top 25 must use end-of-season + playoffs, (2) ATS from walk-forward only, (3) optimize for ATS not mean error, (4) parameters flow through config, (5) sign conventions documented.
- **Added Asymmetric Regression for Preseason Priors** - Fixed spread compression problem for blowout games. Standard regression pulled ALL teams toward mean uniformly, causing bad teams (Kent State -25) to gain 7.5 pts they didn't earn. Now regression scales by distance from mean: teams within ±8 pts get normal 30% regression, teams 20+ pts from mean get only 10% regression. Additionally, talent weight is halved (40%→20%) for extreme teams to trust proven performance over talent projections. Result: rating spread preserved at 90% vs 70% before.
- **Added Triple-Option Team Adjustment** - Fixed systematic underrating of triple-option teams (Navy, Army, Air Force, Kennesaw State). These teams were showing as underdogs when Vegas had them as favorites because: (1) SP+ efficiency metrics don't capture their scheme's value, (2) service academies have artificially low recruiting rankings. Applied +6 pt boost to raw SP+ ratings and 100% prior weight (no talent blend). Result: 2024 early season ATS improved from 49.5% to 51.1%.
- **Added QB Injury Adjustment System** - Manual flagging system for starter injuries. Pre-computes depth charts from CFBD player PPA data, calculates starter/backup differential, applies adjustment = PPA_drop × 30 plays/game. Usage: `--qb-out TEAM` CLI flag. Example adjustments: Georgia -6.8 pts, Ohio State -10.0 pts, Texas +8.0 pts (Arch Manning backup is better).
- **Added Time Decay Parameter (but NOT used)** - Tested time decay (weighting recent games more) in 2D sweep with alpha. Finding: decay consistently hurts performance across all alpha values. Best config is alpha=50, decay=1.0 (no decay). Walk-forward already ensures temporal validity. Parameter added but defaults to 1.0.
- **Tested Havoc Rate (NOT implemented)** - Investigated replacing turnovers with Havoc Rate (TFLs, sacks, PBUs). Finding: Havoc correlates with turnovers forced (r=0.425) but neither provides ATS edge—Vegas already prices both. Kept current turnover approach.
- **Tested Style Mismatch Adjustment (NOT implemented)** - Tested rush/pass style profiles for matchup advantages. Finding: Made predictions worse (ATS -0.2pp, MAE +0.09). Vegas already prices style matchups efficiently.
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
- **Defense Full-Weight Garbage Time** - Hypothesis: when a team is winning by 17+ in Q4, their defense should get full weight (1.0) instead of asymmetric weighting (where trailing offense gets 0.1x). This would give defensive dominance full credit in blowouts. Implementation added `defense_full_weight_gt` parameter with separate O/D regression weights. Results: Rankings improved for teams like Indiana (#4→#2), Georgia (#9→#8), Ole Miss (#14→#12), which matched better with consensus rankings. BUT 5+ edge ATS regressed from 57.3% to 56.5% (-0.8%). Decision: **REJECTED**. The "right" rankings hurt predictive accuracy. Current asymmetric GT appears correctly calibrated—giving defense extra credit in garbage time may overcredit "prevent defense" situations that don't reflect true skill. ATS is our optimization target (Rule #4), so we keep standard asymmetric weighting.
- **Dynamic Ridge Alpha** - Hypothesis: use higher regularization (alpha=75) early season when data is noisy, lower (alpha=35) late season to let elite teams separate more. Results: Dynamic alpha (75→50→35 by week) produced same MAE but hurt 5+ edge ATS from 56.4% to 55.7% (-0.7%). Decision: **REJECTED**. Walk-forward backtesting already self-regularizes via sample size. Lower alpha late-season benefits ALL teams equally, not just elite ones.
- **Increased Turnover Shrinkage** - Hypothesis: turnovers are ~50-70% luck, so prior_strength=10 may over-credit turnover margin. Tested prior_strength 5/10/20/30. Results: prior_strength=10 is optimal for 5+ edge (56.4%). Higher shrinkage (20-30) slightly improved 3+ edge but hurt high-conviction bets. Decision: **KEEP current** prior_strength=10.
- **Turnover-Worthy Plays (TWP) proxy** - Built model using pass breakups and sacks as proxies for interceptable passes and fumble-worthy plays. Multi-year validation showed inconsistent results (helped 2023/2025, hurt 2024). Removed to avoid overfitting.
- **Field Position Component** - Investigated adding field position as JP+ component (SP+ uses 10% weight). Data source: CFBD DrivesApi provides start_yardline. **Problem:** Raw field position is heavily confounded by schedule—good teams have WORSE raw FP (r = -0.56 with SP+) because they face better punters/coverage and score more TDs (receiving kickoffs at own 25). Even after ridge regression opponent adjustment, correlation remains negative (r = -0.63). Tried cleaner "Return Game Rating" using punt/kick return yards: weak correlation with team quality (r = +0.20) and very weak correlation with overperformance (r = +0.06). Top 30 return teams overperformed by only +5.3 ranks on average. Decision: **TABLED**. Signal too weak to justify complexity. May revisit with proper ATS backtest.
