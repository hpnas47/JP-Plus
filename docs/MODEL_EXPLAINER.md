# JP+ College Football Power Ratings

## What It Does

**JP+** is a power ratings model that predicts point spreads for college football games. For any matchup, it answers: "By how many points should Team A beat Team B?"

The name is an homage to Bill Connelly's SP+, which pioneered the efficiency-based approach this model builds upon.

When the model's prediction differs significantly from the Vegas line, that's a potential betting opportunity.

**Example:**
- Vegas says: Ohio State -7 vs Penn State
- JP+ says: Ohio State -10.5
- Edge: JP+ likes Ohio State 3.5 points more than Vegas → potential bet on Ohio State

---

## The Core Philosophy

Most people look at final scores to judge teams. But final scores are noisy—a team can win by 21 because they played great, or because the other team threw 4 interceptions that won't happen again.

**JP+ measures the process, not just the outcome.**

Instead of asking "how many points did they score?", JP+ asks "how efficiently did they move the ball on every single play?" This approach filters out luck and captures sustainable team quality.

---

## What We Measure

### Efficiency Components

| Component | Weight | What It Captures |
|-----------|--------|------------------|
| **Success Rate** | 45% | Did the offense "succeed" on each play? (gain enough yards to stay on schedule) |
| **Explosiveness** | 45% | When successful, how explosive? (big-play ability via EPA on successful plays) |
| **Turnover Margin** | 10% | Ball-hawking and ball-security skill with separate INT/fumble shrinkage |

**Turnover Shrinkage:** Interceptions are treated as skill (moderate Bayesian shrinkage, k=10) while fumble recoveries are treated as luck (strong shrinkage, k=30). This reflects the empirical finding that INT rates correlate year-to-year (QB decision-making, defensive scheme) while fumble recovery is essentially random.

### Key Features

- **Opponent Adjustment:** Ridge regression solves for true team strength after accounting for schedule difficulty
- **Garbage Time Filtering:** Blowout plays are down-weighted for the trailing team, but kept for the winning team (they earned the dominance)
- **Red Zone Leverage:** Plays inside the 20 are weighted 1.5x and inside the 10 are weighted 2.0x, while empty-calorie yards (midfield gains that don't lead to scoring) are weighted 0.7x
- **Conference Strength Anchor:** Out-of-conference games are weighted 1.5x in ridge regression, plus a Bayesian conference strength adjustment corrects inter-conference rating bias
- **Special Teams:** Complete ST model (field goals, punting, kickoffs) expressed as points better than average, with spread impact capped at ±2.5 pts to prevent outlier-driven predictions

---

## Game Adjustments

JP+ applies game-specific adjustments for factors that affect the spread beyond team ratings:

- **Home Field Advantage** (1.0-3.5 pts) — team-specific based on stadium environment, with -0.5 pt global offset to correct systematic home bias
- **Travel & Altitude** — cross-country trips and high-altitude venues penalize visitors
- **Rest Differential** — bye weeks, short weeks, and MACtion scheduling
- **Situational Spots** — letdown games after big wins, lookahead to rivalry/ranked opponents
- **FCS Opponents** — dynamic penalty (10-45 pts) when FBS plays FCS, based on prior game margins
- **QB Continuous** (Weeks 1-3 only) — walk-forward QB quality estimates from PPA data, improving Phase 1 5+ Edge by +0.6%

All adjustments pass through a smoothing layer to prevent over-prediction when multiple factors stack.

### QB Continuous Rating

**The Problem:** In early-season games (Weeks 1-3), the EFM doesn't have enough play-by-play data to accurately capture QB quality. A team starting an elite QB vs a team starting a first-time starter will look similar until the efficiency numbers accumulate.

**The Solution:** QB Continuous uses **Predicted Points Added (PPA)** — a metric from CFBD that measures how much value a QB adds per play compared to average. A QB with +0.30 PPA adds ~0.30 expected points per dropback above replacement level.

**How It Works:**
1. **Prior Season Data:** For Week 1, we use last year's PPA with 0.3 decay (QBs regress toward mean over offseason)
2. **Current Season Data:** As games are played, current-year PPA gradually replaces the prior
3. **Shrinkage (K=200):** QBs with few dropbacks are pulled toward average. A QB needs ~250 dropbacks to get 55% weight on their raw PPA; a backup with 50 dropbacks gets mostly the prior
4. **Walk-Forward Safe:** Only uses data available before the game being predicted — no future leakage

**Why Phase 1 Only?** By Week 4, the EFM has already "baked in" QB quality through success rate and explosiveness metrics. The efficiency data shows who has a good QB. Applying additional QB adjustment for Core weeks causes double-counting and slightly degrades ATS performance.

| Metric | Without QB | With QB Phase1-only | Improvement |
|--------|------------|---------------------|-------------|
| Phase 1 5+ Edge (Close) | 50.2% | 50.5% | **+0.3%** |
| Phase 1 MAE | 15.33 | 14.00 | -1.33 |
| Core 5+ Edge | 54.7% | 54.7% | unchanged |

### Credible Rebuild Adjustment

**The Problem:** The model penalizes teams with low Returning Production (RP) — the percentage of last year's statistical production returning. This makes sense: a team losing 93% of their production should regress heavily. But sometimes low-RP teams have legitimate reasons for optimism: elite recruiting classes or strong portal additions that won't show up in RP data.

**The Solution:** Credible Rebuild identifies teams with extremely low RP (≤35%) who have strong talent signals (recruiting or portal), and reduces their regression penalty. This helps calibration in Weeks 1-3 when the efficiency model hasn't yet captured the new roster's quality.

**Qualification Criteria:**
- Returning Production ≤ 35%
- AND at least one of:
  - Talent (recruiting) normalized score ≥ 0.65
  - Portal normalized score ≥ 0.65

**How It Works:**
1. **Credibility Score:** Average of talent_norm and portal_norm (0-1 scale)
2. **Max Relief:** 25% reduction in regression factor (conservative cap)
3. **Linear Week Taper:** Full relief at Week 1, decreases linearly to 0 by Week 5
4. **Bounded:** Relief can never make a triggered team better than a team at the RP cutoff

**Trigger Rate by Year:**

| Year | Triggered | Total | Rate | Mean Relief |
|------|-----------|-------|------|-------------|
| 2022 | 14 | 234 | 6.0% | +0.31 pts |
| 2023 | 14 | 239 | 5.9% | +0.18 pts |
| 2024 | 13 | 135 | 9.6% | +0.24 pts |
| 2025 | 17 | 137 | 12.4% | +0.34 pts |

**Notable Triggered Teams:**
- 2022 Ole Miss (RP=3%): +0.97 pts — Lane Kiffin's portal army
- 2024 Florida State (RP=21%): +0.39 pts — post-CFP exodus
- 2025 Colorado (RP=7%): +0.35 pts — qualifies on talent despite portal loss

| Metric | Without Rebuild | With Rebuild | Delta |
|--------|-----------------|--------------|-------|
| Phase 1 5+ Edge | 50.2% | 50.4% | +0.2% |
| Core 5+ Edge | 55.1% | 55.1% | unchanged |

The feature is conservative by design: triggers on <15% of teams, applies modest relief, and tapers to zero before Core phase begins.

### Fixed vs LSA: Two Approaches to Situational Adjustments

JP+ applies situational adjustments (home field, rest, travel, letdown spots, etc.) to modify the base spread. There are two methods for calculating these adjustments:

**Fixed Mode:** Each situational factor has a constant point value derived from historical research:
- Bye week advantage: +1.5 pts
- Cross-country travel: -1.0 pts
- Letdown after ranked win: -1.5 pts
- etc.

These values are simple, interpretable, and work well on opening lines before the market has fully priced in situational factors.

**LSA Mode (Learned Situational Adjustments):** Instead of fixed constants, LSA uses **Ridge regression** to learn optimal coefficients from historical prediction errors. The model asks: "Given that JP+ was wrong by X points in this game, and these situational factors were present, what weights minimize future errors?"

Key differences:
- **LSA learns from mistakes** — If JP+ consistently over-predicts home favorites, LSA will reduce that adjustment
- **LSA captures interactions** — Factors that stack (e.g., bye + home + letdown) may need different weights than when isolated
- **LSA is regularized (alpha=300)** — Prevents overfitting by keeping coefficients close to zero unless strong evidence exists
- **LSA clamps extremes** — Maximum ±4.0 pt adjustment to prevent outlier-driven predictions

### Edge-Aware Production Mode

The prediction engine automatically selects Fixed or LSA based on **timing and edge size**. No flags needed — this is the default behavior.

| Bet Timing | Edge Size | Mode | Historical ATS |
|------------|-----------|------|----------------|
| Opening (4+ days) | Any | Fixed | **56.5%** at 5+ |
| Closing (<4 days) | **5+ pts** | LSA | **55.1%** at 5+ |
| Closing (<4 days) | 3-5 pts | Fixed | **52.9%** at 3+ |

**Why this works:**
- **Opening lines** — Fixed dominates because the market hasn't yet priced in all information; our simple constants capture value the books miss early in the week
- **Closing lines, high conviction (5+)** — By game time, obvious situational factors are priced in. LSA's learned coefficients better identify which adjustments still have residual value
- **Closing lines, moderate conviction (3-5)** — Smaller edges are noisier; Fixed's simpler approach avoids LSA's occasional overcorrection (52.9% vs 52.0%)

**Full Fixed vs LSA Comparison (Core Weeks 4-15):**

| Mode | 3+ Edge (Close) | 3+ Edge (Open) | 5+ Edge (Close) | 5+ Edge (Open) |
|------|-----------------|----------------|-----------------|----------------|
| Fixed | **52.9%** | **55.1%** | 54.0% | **56.5%** |
| LSA | 52.0% | 54.9% | **55.1%** | 55.9% |

*For detailed configuration and CLI usage, see [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md).*

---

## JP+ Performance

### Understanding the Metrics

Before diving into the numbers, here's what each metric means:

- **MAE (Mean Absolute Error):** The average number of points JP+ misses by. An MAE of 12.5 means our predictions are off by about 12.5 points on average. For context, Vegas closing lines typically have an MAE of ~11-12 points against actual margins — college football is inherently unpredictable.
- **RMSE (Root Mean Squared Error):** Similar to MAE but penalizes large misses more heavily. RMSE is always higher than MAE; the gap between them indicates how often we have big misses vs. consistent small misses. An RMSE of 16 with an MAE of 13 suggests occasional blowout misses pulling the RMSE up.
- **ATS (Against The Spread):** Win rate when betting JP+'s picks against the Vegas spread. 52.4%+ is the breakeven threshold at standard -110 odds. Anything above that is profitable.
- **CLV (Closing Line Value):** How much better our entry price is vs. the closing line. Positive CLV means the market moved toward our prediction after we identified the edge — widely considered the gold standard for measuring real betting edge.

### Multi-Year Backtest (2022-2025)

Walk-forward backtest across 4 seasons (3,657 games). Model trained only on data available at prediction time — no future leakage.

*All metrics from walk-forward backtest, 2022–2025. Verified 2026-02-12.*

#### Performance by Season Phase

| Phase | Weeks | Games | MAE | RMSE | ATS % (Close) | ATS % (Open) | 3+ Edge (Close) | 3+ Edge (Open) | 5+ Edge (Close) | 5+ Edge (Open) |
|-------|-------|-------|-----|------|---------------|--------------|-----------------|----------------|-----------------|----------------|
| Calibration | 1–3 | 960 | 14.01 | 17.51 | 46.9% | 47.1% | 47.9% (297-323) | 49.1% (304-315) | 51.1% (237-227) | 50.9% (234-226) |
| **Core** | **4–15** | **2,485** | **12.51** | **15.82** | **51.7%** | **53.0%** | **53.1%** (737-652) | **55.4%** (791-637) | **55.1%** (463-378) | **57.0%** (509-384) |
| **Regular Season** | **1–15** | **3,445** | **12.90** | **16.27** | **50.3%** | **51.5%** | **51.5%** (1034-975) | **53.5%** (1095-952) | **53.6%** (700-605) | **54.9%** (743-610) |

**The profitable zone is Weeks 4-15 (Core).** Early-season predictions rely too heavily on preseason priors. Postseason (bowls, CFP) is excluded from JP+ metrics — opt-outs, coaching changes, transfer portal, and motivation variance make it essentially a different sport.

#### Core Season ATS by Edge (Weeks 4–15, 2,485 games)

| Edge | vs Closing Line | vs Opening Line |
|------|-----------------|-----------------|
| All picks | 1,259-1,178 (51.7%) | 1,299-1,151 (53.0%) |
| 3+ pts | 746-650 (53.4%) | 796-636 (55.6%) |
| **5+ pts** | **463-383 (54.7%)** | **515-386 (57.2%)** |

*Games: 1,396 at 3+ edge, 846 at 5+ edge.*

**Key insight:** 5+ point edge is the model's highest-conviction signal. At 54.5% vs closing lines and 57.0% vs opening lines, these are solidly profitable at standard -110 odds (breakeven = 52.4%).

### ATS by Season and Phase (vs Closing Line)

| Year | Phase | Games | ATS % | 3+ Edge | 5+ Edge |
|------|-------|-------|-------|---------|---------|
| 2022 | Core (4-15) | 605 | 52.5% | 188-167 (53.0%) | 115-100 (53.5%) |
| 2023 | Core (4-15) | 611 | 52.5% | 195-162 (54.6%) | 122-98 (55.5%) |
| 2024 | Core (4-15) | 631 | 49.9% | 186-171 (52.1%) | 126-99 (56.0%) |
| 2025 | Core (4-15) | 638 | 51.9% | 177-150 (54.1%) | 100-86 (53.8%) |

**Notes:** 2024 had the weakest overall ATS year (49.9%), but the Core 5+ edge still hit 56.0%. The model's edge concentrates in high-conviction plays regardless of year.

### ATS by Season and Phase (vs Opening Line)

| Year | Phase | Games | ATS % | 3+ Edge | 5+ Edge |
|------|-------|-------|-------|---------|---------|
| 2022 | Core (4-15) | 606 | 52.6% | 197-154 (56.1%) | 116-97 (54.5%) |
| 2023 | Core (4-15) | 614 | 54.8% | 197-155 (56.0%) | 138-96 (59.0%) |
| 2024 | Core (4-15) | 631 | 52.0% | 210-174 (54.7%) | 148-104 (58.7%) |
| 2025 | Core (4-15) | 638 | 53.7% | 192-153 (55.7%) | 113-89 (55.9%) |

Opening line performance significantly exceeds closing line, indicating the model captures value that the market prices out by game time.

### MAE & RMSE by Season (Regular Season Only)

| Year | Games | MAE | RMSE | MAE (Core) | MAE (Cal) |
|------|-------|-----|------|------------|-----------|
| 2022 | 852 | 13.30 | 16.88 | 12.67 | 14.96 |
| 2023 | 866 | 12.74 | 16.13 | 12.48 | 13.06 |
| 2024 | 872 | 13.08 | 16.35 | 12.66 | 14.97 |
| 2025 | 855 | 12.47 | 15.80 | 12.25 | 13.09 |
| **All** | **3,445** | **12.90** | **16.27** | **12.51** | **14.01** |

2025 was JP+'s best year by MAE (12.25 Core), improving from 12.67 in 2022. The trend reflects more seasons of data improving prior calibration.

### Closing Line Value (CLV)

*CLV measures how the market moves after we identify an edge. Positive CLV = the closing line moved toward our prediction, meaning sharp money agrees with us.*

#### Regular Season (Weeks 1-15, 3,445 games)

| Edge Filter | N | Mean CLV (vs Close) | CLV > 0 | ATS % (Close) |
|-------------|---|-------------------|---------|---------------|
| All picks | 3,540 | -0.31 | 28.5% | 50.8% |
| 3+ pt edge | 2,156 | -0.44 | 25.8% | 51.8% |
| 5+ pt edge | 1,389 | -0.50 | 24.8% | 52.9% |
| 7+ pt edge | 834 | -0.56 | 21.9% | 52.0% |

#### Core Season (Weeks 4-15, 2,485 games)

| Edge Filter | N | Mean CLV | CLV > 0 | ATS % (Close) |
|-------------|---|----------|---------|---------------|
| All picks | 2,485 | -0.31 | 30.8% | 51.7% |
| 3+ pt edge | 1,396 | -0.47 | 28.2% | 53.4% |
| **5+ pt edge** | **846** | **-0.51** | **27.3%** | **54.7%** |
| 7+ pt edge | 504 | -0.58 | 24.8% | 54.7% |

**Interpretation:** CLV vs closing is slightly negative, indicating the market does not consistently move toward our predictions. However, the model still generates strong ATS performance — JP+ finds value in spots the market doesn't fully adjust for even by closing. The negative CLV with positive ATS suggests the model exploits structural inefficiencies (public bias, schedule spots) rather than information the sharps eventually price in.

#### CLV vs Opening Line (Captures Value Available at Bet Time)

| Edge Filter | N | Mean CLV (Open→Close) | CLV > 0 | ATS % (Open) |
|-------------|---|----------------------|---------|--------------|
| All picks | 3,540 | +0.41 | 38.4% | 51.6% |
| 3+ pt edge | 2,162 | +0.62 | 40.1% | 53.5% |
| **5+ pt edge** | **1,436** | **+0.74** | **41.3%** | **54.6%** |
| 7+ pt edge | 866 | +0.96 | 41.0% | 54.9% |

When measured against opening lines (the price available when bets are placed), CLV is strongly positive (+0.74 at 5+ edge) and monotonically increasing with edge size — meaning the market moves toward JP+'s predictions by closing. This is a classic indicator of real edge.

---

## 2025 Season Performance

JP+'s most recent season — best Core MAE (12.25) and solid 5+ Edge performance.

| Phase | Weeks | Games | MAE | RMSE | ATS % (Close) | ATS % (Open) | 3+ Edge (Close) | 3+ Edge (Open) | 5+ Edge (Close) | 5+ Edge (Open) |
|-------|-------|-------|-----|------|---------------|--------------|-----------------|----------------|-----------------|----------------|
| Calibration | 1-3 | 244 | 13.09 | 16.60 | 51.2% | 49.6% | 86-66 (56.6%) | 83-69 (54.6%) | 67-47 (58.8%) | 65-49 (57.0%) |
| **Core** | **4-15** | **638** | **12.25** | **15.60** | **51.9%** | **52.5%** | **177-150 (54.1%)** | **192-153 (55.7%)** | **100-86 (53.8%)** | **113-89 (55.9%)** |
| **Regular Season** | **1-15** | **882** | **12.47** | **15.80** | **51.7%** | **51.7%** | **263-216 (54.9%)** | **275-222 (55.3%)** | **167-133 (55.7%)** | **178-138 (56.3%)** |

**2025 highlights:**
- Calibration phase shows strongest 5+ Edge (58.8% Close, 57.0% Open) — QB Continuous helping early-season predictions
- Core 5+ Edge solid at 53.8% (Close) and 55.9% (Open)
- Regular Season 5+ Edge: 55.7% (Close), 56.3% (Open)

---

## 2025 JP+ Top 25

End-of-season power ratings including all games through the National Championship. Note: Backtest metrics use regular season only (weeks 1-15) due to postseason betting edge degradation, but final ratings include postseason to capture each team's full body of work.

| Rank | Team | Overall | Off (rank) | Def (rank) | ST (rank) |
|------|------|---------|------------|------------|-----------|
| 1 | **Ohio State** | +28.8 | +12.7 (8) | +15.6 (3) | +0.54 (34) |
| 2 | Indiana | +28.6 | +16.9 (2) | +10.9 (8) | +0.79 (18) |
| 3 | Notre Dame | +25.0 | +12.9 (6) | +12.5 (5) | -0.28 (91) |
| 4 | Miami | +24.4 | +10.5 (14) | +13.4 (4) | +0.53 (35) |
| 5 | Texas Tech | +23.4 | +4.9 (36) | +17.4 (1) | +1.14 (8) |
| 6 | Oregon | +23.2 | +12.9 (7) | +10.2 (10) | +0.11 (62) |
| 7 | Alabama | +21.6 | +9.7 (16) | +11.8 (6) | +0.06 (68) |
| 8 | Utah | +21.2 | +13.6 (5) | +7.6 (21) | +0.01 (73) |
| 9 | Oklahoma | +19.2 | +2.0 (51) | +15.9 (2) | +1.26 (6) |
| 10 | Vanderbilt | +18.8 | +18.5 (1) | -1.1 (73) | +1.33 (5) |
| 11 | Georgia | +18.3 | +8.0 (21) | +8.7 (18) | +1.62 (2) |
| 12 | Washington | +16.6 | +10.9 (13) | +6.4 (28) | -0.73 (114) |
| 13 | Ole Miss | +15.9 | +12.1 (11) | +2.1 (46) | +1.67 (1) |
| 14 | Louisville | +15.8 | +6.4 (29) | +8.5 (19) | +0.87 (15) |
| 15 | BYU | +15.0 | +7.5 (23) | +7.1 (24) | +0.46 (38) |
| 16 | Missouri | +14.9 | +4.5 (40) | +10.4 (9) | +0.14 (60) |
| 17 | South Florida | +14.8 | +8.1 (20) | +7.0 (25) | -0.34 (98) |
| 18 | Texas A&M | +14.5 | +7.5 (24) | +7.3 (23) | -0.34 (97) |
| 19 | James Madison | +14.3 | +4.8 (37) | +9.2 (15) | +0.32 (50) |
| 20 | Florida State | +14.1 | +12.3 (10) | +1.7 (51) | -0.01 (75) |
| 21 | Tennessee | +13.9 | +11.1 (12) | +1.8 (49) | +1.11 (12) |
| 22 | Penn State | +13.1 | +8.6 (18) | +3.6 (38) | +0.98 (14) |
| 23 | SMU | +12.7 | +7.3 (26) | +5.4 (32) | -0.04 (78) |
| 24 | Texas | +12.4 | +3.2 (48) | +9.5 (13) | -0.34 (99) |
| 25 | USC | +11.8 | +12.4 (9) | -0.6 (67) | +0.02 (72) |

**Ohio State** — JP+ #1 despite losing to Indiana in CFP semifinal. Best combination of offense (#8) and elite defense (#3). The model values consistent efficiency over tournament results.

**Indiana** — National Champions. Beat Alabama 38-3, Oregon 56-22, and Miami 27-21 in CFP. JP+ #2 overall with the #2 offense in the country. Their championship run validates the explosive offense that JP+ identified all season.

**Spread Calculation:** Team A rating - Team B rating = expected point spread (before game-specific adjustments).

---

## JP+ Totals Model (Over/Under)

In addition to spreads, JP+ includes a separate **Totals Model** for predicting game over/unders. This uses a different architecture than the spreads model.

### Why a Separate Model?

The spreads model (EFM) measures *efficiency* — how well teams move the ball on each play. But efficiency ratings don't directly translate to total points scored. A team rated +10 isn't going to score 10 more points than average.

The Totals Model uses **game-level scoring data** (points scored and allowed) rather than play-level efficiency. It solves for each team's offensive and defensive scoring adjustments via Ridge regression, just like EFM, but on actual point totals.

### How It Works

- **Training data:** Each game produces 2 observations: home team scores X against away defense, away team scores Y against home defense
- **Ridge regression:** Solves for team offensive/defensive adjustments relative to FBS average (~24 ppg)
- **Learned HFA:** Home field advantage is learned from the data (+3.5 to +4.5 pts typical), not assumed
- **Walk-forward:** Only uses games from weeks prior to the prediction week

**Prediction formula:**
```
home_expected = baseline + (home_off_adj + away_def_adj) / 2 + hfa_coef
away_expected = baseline + (away_off_adj + home_def_adj) / 2
predicted_total = home_expected + away_expected
```

### Totals Model Performance (2023-2025)

*Walk-forward backtest across 3 seasons (1,993 regular season games). 2022 excluded — scoring environment transition year with 49% ATS.*

#### Performance by Phase (vs Closing Line)

| Phase | Weeks | Games | MAE | ATS % | 3+ Edge | 5+ Edge |
|-------|-------|-------|-----|-------|---------|---------|
| Calibration | 1-3 | 169 | 12.42 | 57.9% | 56.7% (59-45) | **61.1%** (44-28) |
| **Core** | **4-15** | **1,824** | **13.09** | **53.9%** | **54.7%** (539-446) | **54.5%** (334-279) |
| **Regular Season** | **1-15** | **1,993** | **13.03** | **54.3%** | **54.9%** (598-491) | **55.0%** (378-307) |

#### Performance by Phase (vs Opening Line)

| Phase | Weeks | Games | MAE | ATS % | 3+ Edge | 5+ Edge |
|-------|-------|-------|-----|-------|---------|---------|
| Calibration | 1-3 | 169 | 12.42 | 55.4% | 57.0% (57-43) | **58.7%** (37-26) |
| **Core** | **4-15** | **1,824** | **13.09** | **53.4%** | **54.2%** (528-447) | **55.3%** (330-267) |
| **Regular Season** | **1-15** | **1,993** | **13.03** | **53.6%** | **54.4%** (585-490) | **55.6%** (367-293) |

#### Full Season by Year

| Year | Games | MAE | 3+ Edge (Close) | 3+ Edge (Open) | 5+ Edge (Close) | 5+ Edge (Open) |
|------|-------|-----|-----------------|----------------|-----------------|----------------|
| 2023 | 713 | 13.37 | 55.0% (204-167) | 53.5% (192-167) | 54.0% (143-122) | 55.3% (135-109) |
| **2024** | **706** | **13.03** | **56.2%** (230-179) | **55.6%** (225-180) | **58.6%** (143-101) | **58.2%** (139-100) |
| 2025 | 708 | 12.80 | 53.5% (208-181) | 54.3% (213-179) | 53.2% (118-104) | 53.2% (117-103) |

### Key Insights

- **Calibration phase is strongest** — opposite of spreads model. Early-season totals are more predictable because scoring baseline is stable.
- **2024 was exceptional** — 59.5% 5+ Edge. May reflect market recalibration after multi-year scoring environment shift (57 → 53 PPG since 2018).
- **Core 5+ Edge: 54.5% (Close), 55.3% (Open)** — solidly above the 52.4% breakeven threshold.

### Configuration

- **Years:** 2023-2025 (2022 excluded — transition year)
- **Ridge Alpha:** 10.0
- **Decay Factor:** 1.0 (no within-season decay)
- **Learned HFA:** Via Ridge column (+3.5 to +4.5 pts typical)
- **Weather:** Available — see Weather Adjustments section below

---

## Weather Adjustments (Totals)

Weather affects totals through wind, cold, and precipitation. The model uses **non-linear thresholds** based on sharp betting research — the key insight is that **light weather doesn't matter, but severe weather kills scoring**.

### The Timing Edge

The market prices weather, but slowly. The edge is in **timing**:
- **Thursday 6 AM** — Capture forecasts 72 hours out, bet before books adjust
- **Saturday 8 AM** — Confirm with accurate 6-12h forecasts (final model input)

### Confidence Gating

**All adjustments are scaled by forecast confidence** based on hours until kickoff:

| Hours Out | Confidence | Effect on -6.0 raw |
|-----------|------------|-------------------|
| ≤6h | 0.95 | -5.7 pts |
| 6-12h | 0.90 | -5.4 pts |
| 12-24h | 0.85 | -5.1 pts |
| 24-48h | 0.75 | -4.5 pts |
| >48h | 0.65 | -3.9 pts |

**HIGH_VARIANCE Flag:** If confidence < 0.75 AND raw adjustment > 3.0 pts, the game is flagged `high_variance=True`. **Rule: Never bet OVER on these games** — weather is uncertain but potentially severe.

### Wind (The #1 Factor)

Wind is king of unders. Uses **effective wind = (wind_speed + wind_gust) / 2**.

| Effective Wind | Base Adjustment |
|----------------|-----------------|
| <12 mph | 0 pts (no impact) |
| 12-15 mph | -1.5 pts |
| 15-20 mph | -4.0 pts |
| >20 mph | -6.0 pts |

Wind impact scales by team passing tendency:

| Team Style | Example | 20 mph Wind |
|------------|---------|-------------|
| Run-heavy | Army | -4.2 pts |
| Balanced | Average | -6.0 pts |
| Pass-heavy | Ole Miss | -7.8 pts |

### Temperature

Cold turns the ball into a rock (harder to catch and kick).

| Temperature | Adjustment |
|-------------|------------|
| >32°F | 0 pts |
| 20-32°F | -1.0 pts |
| <20°F | -3.0 pts |

### Precipitation

**The "Slick Trap":** Light rain does NOT hurt totals. Defenders slip, miss tackles, games can even go OVER. Only heavy rain/snow causes conservative playcalling.

| Condition | Adjustment |
|-----------|------------|
| Light rain (<0.1 in/hr) | 0 pts (the "slick trap") |
| Heavy rain (>0.3 in/hr) | -2.5 pts |
| Snow with wind (≥12 mph) | -3.0 pts |
| Snow without wind | 0 pts ("overreaction fade") |

**"Snow Overreaction Fade":** Public loves betting "Snow Unders" but sharps know snow without wind often goes OVER. Defenders slip, receivers know their routes. Only apply snow penalty when wind is also present.

### Backtest Note

Historical backtest shows weather provides no ATS improvement (market already prices it). The edge is in **timing** — acting on Thursday forecasts before the market adjusts totals.

---

## Reality Check

Vegas lines are set by professionals with decades of experience and access to information we don't have (injury reports, locker room intel, sharp money). Beating them consistently is hard. The goal is to find spots where JP+ has an edge, not to win every bet.

**What JP+ is NOT:**
- **Not a betting system** — It's a tool for analysis, not guaranteed profits
- **Not magic** — College football is chaotic; upsets happen
- **Not complete** — Still being refined and improved

---

## Data Sources

- **Play-by-play data:** College Football Data API (collegefootballdata.com)
- **Historical betting lines (2022-2025):** CFBD API (91% opening line coverage)
- **Future betting lines (2026+):** The Odds API (opening + closing captures)

---

## Learn More

For technical details on the model architecture, adjustment formulas, parameter values, and implementation:

**[MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md)** — Complete technical documentation

---

## Summary

JP+ rates college football teams by analyzing play-by-play efficiency rather than final scores. It adjusts for opponent strength, home field, and situational factors to predict point spreads. When JP+ significantly disagrees with Vegas, that's worth investigating.

**The philosophy: Measure the process (efficiency), not just the outcome (points).**
