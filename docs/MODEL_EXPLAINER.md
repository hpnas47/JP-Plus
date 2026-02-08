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
| 1 | **Indiana** | +30.9 | +19.3 (2) | +11.6 (9) | +1.09 (30) |
| 2 | Ohio State | +30.8 | +16.6 (6) | +14.2 (3) | +1.06 (32) |
| 3 | Oregon | +28.2 | +16.5 (7) | +11.6 (8) | +0.63 (52) |
| 4 | Notre Dame | +28.2 | +15.9 (9) | +12.3 (5) | -0.76 (114) |
| 5 | Miami | +27.4 | +14.3 (14) | +13.1 (4) | +1.25 (24) |
| 6 | Alabama | +25.7 | +14.0 (15) | +11.7 (6) | -0.16 (90) |
| 7 | Texas Tech | +25.5 | +9.3 (42) | +16.2 (1) | +1.47 (14) |
| 8 | Utah | +24.9 | +16.6 (5) | +8.2 (24) | +0.43 (62) |
| 9 | Oklahoma | +24.6 | +8.5 (48) | +16.1 (2) | +1.76 (9) |
| 10 | Texas A&M | +23.6 | +13.0 (17) | +10.6 (12) | -0.63 (112) |
| 11 | Georgia | +23.4 | +13.0 (18) | +10.4 (14) | +2.74 (1) |
| 12 | Washington | +23.0 | +15.0 (13) | +7.9 (26) | -0.55 (109) |
| 13 | Vanderbilt | +22.5 | +19.8 (1) | +2.7 (60) | +2.55 (2) |
| 14 | Ole Miss | +21.8 | +16.0 (8) | +5.8 (37) | +2.33 (3) |
| 15 | Missouri | +21.6 | +10.3 (35) | +11.3 (11) | +0.03 (83) |
| 16 | Texas | +20.8 | +9.0 (44) | +11.7 (7) | +0.63 (51) |
| 17 | Louisville | +20.6 | +11.7 (26) | +8.9 (20) | +1.12 (28) |
| 18 | BYU | +20.2 | +12.0 (24) | +8.2 (23) | +0.61 (53) |
| 19 | Tennessee | +19.2 | +15.4 (12) | +3.9 (54) | +1.66 (13) |
| 20 | Penn State | +19.1 | +12.8 (19) | +6.3 (31) | +2.25 (4) |
| 21 | Florida State | +19.1 | +15.5 (11) | +3.6 (56) | +0.49 (60) |
| 22 | Auburn | +18.7 | +9.6 (38) | +9.1 (19) | +0.73 (44) |
| 23 | USC | +18.6 | +15.6 (10) | +3.0 (59) | +0.49 (59) |
| 24 | Iowa | +18.4 | +8.2 (53) | +10.2 (15) | +0.93 (35) |
| 25 | Michigan | +18.1 | +13.1 (16) | +5.0 (43) | -0.58 (110) |

**Indiana** — National Champions. Beat Alabama 38-3, Oregon 56-22, and Miami 27-21 in CFP. JP+ ranks them #1 overall with the #2 offense and #9 defense in the country.

**Spread Calculation:** Team A rating - Team B rating = expected point spread (before game-specific adjustments).

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
