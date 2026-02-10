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
| **Turnover Margin** | 10% | Ball-hawking and ball-security skill (regressed to account for luck) |

### Key Features

- **Opponent Adjustment:** Ridge regression solves for true team strength after accounting for schedule difficulty
- **Garbage Time Filtering:** Blowout plays are down-weighted for the trailing team, but kept for the winning team (they earned the dominance)
- **Red Zone Leverage:** Plays inside the 20 are weighted 1.5x and inside the 10 are weighted 2.0x, while empty-calorie yards (midfield gains that don't lead to scoring) are weighted 0.7x
- **Conference Strength Anchor:** Out-of-conference games are weighted 1.5x in ridge regression, plus a Bayesian conference strength adjustment corrects inter-conference rating bias
- **Special Teams:** Complete ST model (field goals, punting, kickoffs) expressed as points better than average

---

## Game Adjustments

JP+ applies game-specific adjustments for factors that affect the spread beyond team ratings:

- **Home Field Advantage** (1.0-3.5 pts) — team-specific based on stadium environment, with -0.5 pt global offset to correct systematic home bias
- **Travel & Altitude** — cross-country trips and high-altitude venues penalize visitors
- **Rest Differential** — bye weeks, short weeks, and MACtion scheduling
- **Situational Spots** — letdown games after big wins, lookahead to rivalry/ranked opponents
- **FCS Opponents** — tiered penalty when FBS plays FCS

All adjustments pass through a smoothing layer to prevent over-prediction when multiple factors stack.

*For detailed adjustment values and formulas, see [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md).*

---

## JP+ Performance

### Understanding the Metrics

Before diving into the numbers, here's what each metric means:

- **MAE (Mean Absolute Error):** The average number of points JP+ misses by. An MAE of 12.5 means our predictions are off by about 12.5 points on average. For context, Vegas closing lines typically have an MAE of ~11-12 points against actual margins — college football is inherently unpredictable.
- **RMSE (Root Mean Squared Error):** Similar to MAE but penalizes large misses more heavily. RMSE is always higher than MAE; the gap between them indicates how often we have big misses vs. consistent small misses. An RMSE of 16 with an MAE of 13 suggests occasional blowout misses pulling the RMSE up.
- **ATS (Against The Spread):** Win rate when betting JP+'s picks against the Vegas spread. 52.4%+ is the breakeven threshold at standard -110 odds. Anything above that is profitable.
- **CLV (Closing Line Value):** How much better our entry price is vs. the closing line. Positive CLV means the market moved toward our prediction after we identified the edge — widely considered the gold standard for measuring real betting edge.

### Multi-Year Backtest (2022-2025)

Walk-forward backtest across 4 seasons (3,273 games). Model trained only on data available at prediction time — no future leakage.

*All metrics from walk-forward backtest, 2022–2025. Verified 2026-02-08.*

#### Performance by Season Phase

| Phase | Weeks | Games | MAE | RMSE | ATS % (Close) | ATS % (Open) | 3+ Edge (Close) | 5+ Edge (Close) | 5+ Edge (Open) |
|-------|-------|-------|-----|------|---------------|--------------|-----------------|-----------------|----------------|
| Calibration | 1–3 | 608 | 14.77 | 18.61 | 47.1% | 48.6% | 47.6% | 48.6% | 49.8% |
| **Core** | **4–15** | **2,489** | **12.50** | **15.82** | **52.2%** | **53.5%** | **54.0%** | **54.7%** | **57.8%** |
| Postseason | 16+ | 176 | 13.41 | 16.82 | 47.4% | 48.3% | 46.7% | 46.7% | 46.8% |
| **Full** | **All** | **3,273** | **12.97** | **16.43** | **51.0%** | **52.3%** | **52.2%** | **52.7%** | **55.2%** |

**The profitable zone is Weeks 4-15.** Early-season predictions rely too heavily on preseason priors, and bowl games have unmodeled factors (opt-outs, motivation, long layoffs).

#### Core Season ATS by Edge (Weeks 4–15, 2,489 games)

| Edge | vs Closing Line | vs Opening Line |
|------|-----------------|-----------------|
| All picks | 1,272-1,165 (52.2%) | 1,311-1,139 (53.5%) |
| 3+ pts | 764-650 (54.0%) | 810-648 (55.6%) |
| **5+ pts** | **473-391 (54.7%)** | **525-384 (57.8%)** |

**Key insight:** 5+ point edge is the model's highest-conviction signal. At 54.7% vs closing lines and 57.8% vs opening lines, these are solidly profitable at standard -110 odds (breakeven = 52.4%).

### ATS by Season and Phase (vs Closing Line)

| Year | Phase | Games | ATS % | 3+ Edge | 5+ Edge |
|------|-------|-------|-------|---------|---------|
| 2022 | Core (4-15) | 605 | 53.2% | 188-172 (52.2%) | 117-105 (52.7%) |
| 2023 | Core (4-15) | 611 | 52.5% | 203-165 (55.2%) | 124-101 (55.1%) |
| 2024 | Core (4-15) | 631 | 50.4% | 193-167 (53.6%) | 124-101 (55.1%) |
| 2025 | Core (4-15) | 638 | 52.7% | 180-146 (55.2%) | 108-84 (56.2%) |

**Notes:** 2024 was the weakest overall ATS year, but the Core 5+ edge still hit 55.1%. The model's edge concentrates in high-conviction plays regardless of year.

### ATS by Season and Phase (vs Opening Line)

| Year | Phase | Games | ATS % | 3+ Edge | 5+ Edge |
|------|-------|-------|-------|---------|---------|
| 2022 | Core (4-15) | 605 | 52.6% | 192-160 (54.5%) | 118-100 (54.1%) |
| 2023 | Core (4-15) | 611 | 54.8% | 202-158 (56.1%) | 132-97 (57.6%) |
| 2024 | Core (4-15) | 631 | 52.0% | 200-173 (53.6%) | 144-101 (58.8%) |
| 2025 | Core (4-15) | 638 | 53.7% | 197-151 (56.6%) | 115-84 (57.8%) |

Opening line performance significantly exceeds closing line, indicating the model captures value that the market prices out by game time.

### MAE & RMSE by Season

| Year | Games (Full) | MAE (Full) | RMSE (Full) | MAE (Core) | RMSE (Core) | MAE (Cal) | MAE (Post) |
|------|-------------|------------|-------------|------------|-------------|-----------|------------|
| 2022 | 802 | 13.28 | 16.95 | 12.72 | 16.24 | 15.63 | 12.90 |
| 2023 | 816 | 13.04 | 16.49 | 12.42 | 15.75 | 14.60 | 16.11 |
| 2024 | 818 | 13.13 | 16.39 | 12.68 | 15.76 | 15.48 | 12.09 |
| 2025 | 837 | 12.45 | 15.90 | 12.19 | 15.55 | 13.42 | 12.72 |
| **All** | **3,273** | **12.97** | **16.43** | **12.50** | **15.82** | **14.77** | **13.41** |

2025 was JP+'s best year by MAE (12.19 Core), improving from 12.72 in 2022. The trend reflects more seasons of data improving prior calibration.

### Closing Line Value (CLV)

*CLV measures how the market moves after we identify an edge. Positive CLV = the closing line moved toward our prediction, meaning sharp money agrees with us.*

#### Full Season (Weeks 1+, 3,258 games with lines)

| Edge Filter | N | Mean CLV (vs Close) | CLV > 0 | ATS % (Close) |
|-------------|---|-------------------|---------|---------------|
| All picks | 3,258 | -0.32 | 29.0% | 51.1% |
| 3+ pt edge | 1,997 | -0.43 | 26.2% | 51.8% |
| 5+ pt edge | 1,294 | -0.50 | 23.3% | 52.3% |
| 7+ pt edge | 779 | -0.51 | 21.2% | 53.0% |

#### Core Season (Weeks 4-15, 2,489 games)

| Edge Filter | N | Mean CLV | CLV > 0 | ATS % (Close) |
|-------------|---|----------|---------|---------------|
| All picks | 2,489 | -0.31 | — | 52.2% |
| 3+ pt edge | 1,414 | — | — | 54.0% |
| **5+ pt edge** | **864** | **-0.51** | **—** | **54.7%** |
| 7+ pt edge | — | — | — | — |

**Interpretation:** CLV vs closing is slightly negative, indicating the market does not consistently move toward our predictions. However, the model still generates strong ATS performance — JP+ finds value in spots the market doesn't fully adjust for even by closing. The negative CLV with positive ATS suggests the model exploits structural inefficiencies (public bias, schedule spots) rather than information the sharps eventually price in.

#### CLV vs Opening Line (Captures Value Available at Bet Time)

| Edge Filter | N | Mean CLV (Open→Close) | ATS % (Open) |
|-------------|---|----------------------|--------------|
| All picks | 3,258 | +0.44 | 52.7% |
| 3+ pt edge | 2,001 | +0.61 | 53.6% |
| **5+ pt edge** | **1,339** | **+0.75** | **54.4%** |
| 7+ pt edge | 824 | +0.93 | 55.4% |

When measured against opening lines (the price available when bets are placed), CLV is strongly positive (+0.75 at 5+ edge) and monotonically increasing with edge size — meaning the market moves toward JP+'s predictions by closing. This is a classic indicator of real edge.

### Reality Check

Vegas lines are set by professionals with decades of experience and access to information we don't have (injury reports, locker room intel, sharp money). Beating them consistently is hard. The goal is to find spots where JP+ has an edge, not to win every bet.

---

## What JP+ Is NOT

- **Not a betting system** — It's a tool for analysis, not guaranteed profits
- **Not magic** — College football is chaotic; upsets happen
- **Not complete** — Still being refined and improved

---

## 2025 Season Performance

JP+'s most recent season — best Core MAE (12.19) and strongest 5+ Edge performance across all years.

| Phase | Weeks | Games | MAE | RMSE | ATS % (Close) | ATS % (Open) | 3+ Edge (Close) | 5+ Edge (Close) | 5+ Edge (Open) |
|-------|-------|-------|-----|------|---------------|--------------|-----------------|-----------------|----------------|
| Calibration | 1-3 | 153 | 13.42 | 17.33 | 50.7% | 50.7% | 54-51 (51.4%) | 37-34 (52.1%) | 37-32 (53.6%) |
| **Core** | **4-15** | **638** | **12.19** | **15.55** | **52.7%** | **54.1%** | **180-146 (55.2%)** | **108-84 (56.2%)** | **118-84 (58.4%)** |
| Postseason | 16+ | 46 | 12.72 | 15.66 | 43.5% | 43.5% | 11-15 (42.3%) | 8-9 (47.1%) | 8-7 (53.3%) |
| **Full Season** | **All** | **837** | **12.45** | **15.90** | **51.8%** | **52.9%** | **245-212 (53.6%)** | **153-127 (54.6%)** | **163-123 (57.0%)** |

**2025 highlights:**
- Core 5+ Edge at 56.2% (Close) and 58.4% (Open) — strongest single-season performance
- Full-season 5+ Edge profitable at 54.6% (Close) even including weak Calibration/Postseason phases
- Postseason weakness persists (43.5% ATS) — bowl opt-outs and motivation remain unmodeled

---

## 2025 JP+ Top 25

End-of-season power ratings including all postseason (bowls + CFP through National Championship):

| Rank | Team | Overall | Off (rank) | Def (rank) | ST (rank) |
|------|------|---------|------------|------------|-----------|
| 1 | **Ohio State** | +30.0 | +12.1 (9) | +15.9 (2) | +2.00 (13) |
| 2 | Indiana | +28.7 | +16.5 (2) | +10.7 (7) | +1.54 (19) |
| 3 | Notre Dame | +26.5 | +12.9 (6) | +12.6 (5) | +0.97 (33) |
| 4 | Miami | +25.8 | +10.2 (14) | +13.1 (4) | +2.49 (8) |
| 5 | Texas Tech | +25.1 | +4.9 (36) | +17.3 (1) | +2.87 (5) |
| 6 | Oregon | +23.5 | +12.6 (7) | +10.1 (10) | +0.82 (44) |
| 7 | Alabama | +22.1 | +9.8 (15) | +11.8 (6) | +0.39 (61) |
| 8 | Utah | +21.6 | +13.0 (5) | +7.7 (20) | +0.87 (39) |
| 9 | Vanderbilt | +20.5 | +18.0 (1) | -0.3 (66) | +2.84 (6) |
| 10 | Georgia | +19.8 | +8.0 (20) | +8.8 (15) | +3.04 (4) |
| 11 | Oklahoma | +19.3 | +2.8 (48) | +15.3 (3) | +1.20 (27) |
| 12 | Ole Miss | +17.8 | +11.8 (11) | +2.5 (44) | +3.49 (2) |
| 13 | Tennessee | +17.4 | +10.6 (12) | +2.7 (43) | +4.09 (1) |
| 14 | Louisville | +17.2 | +6.4 (29) | +8.4 (17) | +2.48 (9) |
| 15 | Washington | +16.6 | +10.6 (13) | +6.6 (26) | -0.57 (91) |
| 16 | Florida State | +16.4 | +12.0 (10) | +2.3 (46) | +2.06 (12) |
| 17 | Texas A&M | +15.5 | +7.7 (21) | +7.3 (23) | +0.52 (56) |
| 18 | BYU | +15.2 | +7.6 (22) | +6.9 (24) | +0.58 (53) |
| 19 | Missouri | +14.5 | +4.7 (38) | +10.4 (9) | -0.56 (90) |
| 20 | South Florida | +14.1 | +7.0 (25) | +6.2 (27) | +0.94 (34) |
| 21 | USC | +13.3 | +12.4 (8) | +0.3 (60) | +0.59 (52) |
| 22 | Penn State | +12.9 | +8.5 (18) | +3.9 (37) | +0.40 (60) |
| 23 | SMU | +12.9 | +7.4 (24) | +5.4 (30) | +0.09 (71) |
| 24 | James Madison | +12.5 | +4.1 (41) | +7.5 (22) | +0.89 (38) |
| 25 | Virginia | +12.3 | +1.3 (56) | +9.8 (11) | +1.25 (26) |

**Indiana** — National Champions. Beat Alabama 38-3, Oregon 56-22, and Miami 27-21 in CFP. JP+ ranks them #1 overall with the #2 offense and #9 defense in the country.

**Spread Calculation:** Team A rating - Team B rating = expected point spread (before game-specific adjustments).

---

## Data Sources

- **Play-by-play data:** College Football Data API (collegefootballdata.com)
- **Historical betting lines (2022-2025):** CFBD API (91% opening line coverage)
- **Future betting lines (2026+):** The Odds API (opening + closing captures)

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

*Walk-forward backtest across 3 seasons (2,127 games). 2022 excluded — scoring environment transition year with 49% ATS.*

#### Performance by Phase (vs Closing Line)

| Phase | Weeks | Games | MAE | ATS % | 3+ Edge | 5+ Edge |
|-------|-------|-------|-----|-------|---------|---------|
| Calibration | 1-3 | 169 | 12.42 | 57.9% | 56.7% (59-45) | **61.1%** (44-28) |
| **Core** | **4-15** | **1,824** | **13.09** | **53.9%** | **54.7%** (539-446) | **54.5%** (334-279) |
| Postseason | 16+ | 134 | 13.57 | 53.2% | 55.0% (44-36) | 56.5% (26-20) |
| **Full Season** | **All** | **2,127** | **13.07** | **54.1%** | **54.9%** (642-527) | **55.3%** (404-327) |

#### Performance by Phase (vs Opening Line)

| Phase | Weeks | Games | MAE | ATS % | 3+ Edge | 5+ Edge |
|-------|-------|-------|-----|-------|---------|---------|
| Calibration | 1-3 | 169 | 12.42 | 55.4% | 57.0% (57-43) | **58.7%** (37-26) |
| **Core** | **4-15** | **1,824** | **13.09** | **53.4%** | **54.2%** (528-447) | **55.3%** (330-267) |
| Postseason | 16+ | 134 | 13.57 | 54.8% | 55.6% (45-36) | 55.8% (24-19) |
| **Full Season** | **All** | **2,127** | **13.57** | **53.6%** | **54.5%** (630-526) | **55.6%** (391-312) |

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

**"Passing Team" Multiplier:** Continuous scaling based on combined pass rate:
```
multiplier = combined_pass_rate / 0.50 (clamped to [0.5, 1.5])
```

| Team Style | Pass Rate | Multiplier | 20 mph Wind |
|------------|-----------|------------|-------------|
| Triple Option (Army) | 35% | 0.70x | -4.2 pts |
| Balanced | 50% | 1.00x | -6.0 pts |
| Air Raid (Ole Miss) | 65% | 1.30x | -7.8 pts |

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

## Learn More

For technical details on the model architecture, adjustment formulas, parameter values, and implementation:

**[MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md)** — Complete technical documentation

---

## Summary

JP+ rates college football teams by analyzing play-by-play efficiency rather than final scores. It adjusts for opponent strength, home field, and situational factors to predict point spreads. When JP+ significantly disagrees with Vegas, that's worth investigating.

**The philosophy: Measure the process (efficiency), not just the outcome (points).**
