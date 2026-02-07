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

- **Home Field Advantage** (1.5-4.0 pts) — team-specific based on stadium environment
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

*All metrics from walk-forward backtest, 2022–2025. Verified 2026-02-06.*

#### Performance by Season Phase

| Phase | Weeks | Games | MAE | RMSE | ATS % (Close) | ATS % (Open) | 5+ Edge (Close) | 5+ Edge (Open) |
|-------|-------|-------|-----|------|---------------|--------------|-----------------|----------------|
| Calibration | 1–3 | 597 | 14.94 | — | 47.1% | 48.6% | 47.4% | 48.8% |
| **Core** | **4–15** | **2,485** | **12.52** | **15.84** | **52.4%** | **54.0%** | **54.6%** | **56.9%** |
| Postseason | 16+ | 176 | 13.43 | — | 47.4% | 48.3% | 46.7% | 47.4% |
| **Full** | **All** | **3,273** | **13.02** | **16.50** | **51.1%** | **52.7%** | **52.3%** | **54.4%** |

**The profitable zone is Weeks 4-15.** Early-season predictions rely too heavily on preseason priors, and bowl games have unmodeled factors (opt-outs, motivation, long layoffs).

#### Core Season ATS by Edge (Weeks 4–15, 2,485 games)

| Edge | vs Closing Line | vs Opening Line |
|------|-----------------|-----------------|
| All picks | 1,276-1,161 (52.4%) | 1,322-1,128 (54.0%) |
| 3+ pts | 764-669 (53.3%) | 811-650 (55.5%) |
| **5+ pts** | **473-393 (54.6%)** | **525-397 (56.9%)** |

**Key insight:** 5+ point edge is the model's highest-conviction signal. At 54.6% vs closing lines and 56.9% vs opening lines, these are solidly profitable at standard -110 odds (breakeven = 52.4%).

### ATS by Season and Phase (vs Closing Line)

| Year | Phase | Games | ATS % | 3+ Edge | 5+ Edge |
|------|-------|-------|-------|---------|---------|
| 2022 | Full | 786 | 52.0% | 250-249 (50.1%) | 165-161 (50.6%) |
| 2022 | Core (4-15) | 596 | 53.5% | 182-177 (50.7%) | 112-105 (51.6%) |
| 2023 | Full | 794 | 51.8% | 274-243 (53.0%) | 185-161 (53.5%) |
| 2023 | Core (4-15) | 598 | 53.0% | 206-168 (55.1%) | 128-102 (55.7%) |
| 2024 | Full | 796 | 48.5% | 243-236 (50.7%) | 157-152 (50.8%) |
| 2024 | Core (4-15) | 617 | 50.1% | 193-170 (53.2%) | 123-100 (55.2%) |
| 2025 | Full | 824 | 52.2% | 250-219 (53.3%) | 157-132 (54.3%) |
| 2025 | Core (4-15) | 626 | 52.9% | 183-154 (54.3%) | 110-86 (56.1%) |

**Notes:** 2024 was the weakest overall ATS year, but the Core 5+ edge still hit 55.2%. The model's edge concentrates in high-conviction plays regardless of year.

### ATS by Season and Phase (vs Opening Line)

| Year | Phase | Games | ATS % | 3+ Edge | 5+ Edge |
|------|-------|-------|-------|---------|---------|
| 2022 | Core (4-15) | 593 | 53.6% | 190-158 (54.6%) | 116-102 (53.2%) |
| 2023 | Core (4-15) | 598 | 55.5% | 209-168 (55.4%) | 139-101 (57.9%) |
| 2024 | Core (4-15) | 625 | 52.5% | 210-169 (55.4%) | 153-106 (59.1%) |
| 2025 | Core (4-15) | 634 | 54.3% | 202-155 (56.6%) | 117-88 (57.1%) |

Opening line performance significantly exceeds closing line, indicating the model captures value that the market prices out by game time.

### MAE & RMSE by Season

| Year | Games (Full) | MAE (Full) | RMSE (Full) | MAE (Core) | RMSE (Core) | MAE (Cal) | MAE (Post) |
|------|-------------|------------|-------------|------------|-------------|-----------|------------|
| 2022 | 802 | 13.37 | 17.07 | 12.75 | 16.29 | 15.97 | 12.84 |
| 2023 | 816 | 13.14 | 16.64 | 12.43 | 15.77 | 15.07 | 16.18 |
| 2024 | 818 | 13.13 | 16.40 | 12.67 | 15.76 | 15.45 | 12.19 |
| 2025 | 837 | 12.46 | 15.90 | 12.22 | 15.56 | 13.37 | 12.70 |
| **All** | **3,273** | **13.02** | **16.50** | **12.52** | **15.84** | **14.94** | **13.43** |

2025 was JP+'s best year by MAE (12.22 Core), improving from 12.75 in 2022. The trend reflects more seasons of data improving prior calibration.

### Closing Line Value (CLV)

*CLV measures how the market moves after we identify an edge. Positive CLV = the closing line moved toward our prediction, meaning sharp money agrees with us.*

#### Full Season (Weeks 1+, 3,258 games with lines)

| Edge Filter | N | Mean CLV (vs Close) | CLV > 0 | ATS % (Close) |
|-------------|---|-------------------|---------|---------------|
| All picks | 3,258 | -0.32 | 29.0% | 51.1% |
| 3+ pt edge | 1,997 | -0.43 | 26.2% | 51.8% |
| 5+ pt edge | 1,294 | -0.50 | 23.3% | 52.3% |
| 7+ pt edge | 779 | -0.51 | 21.2% | 53.0% |

#### Core Season (Weeks 4-15, 2,485 games)

| Edge Filter | N | Mean CLV | CLV > 0 | ATS % (Close) |
|-------------|---|----------|---------|---------------|
| All picks | 2,485 | -0.31 | — | 52.4% |
| 3+ pt edge | 1,433 | — | — | 53.3% |
| **5+ pt edge** | **866** | **-0.51** | **—** | **54.6%** |
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
