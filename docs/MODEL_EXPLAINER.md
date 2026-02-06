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
| **Success Rate** | 54% | Did the offense "succeed" on each play? (gain enough yards to stay on schedule) |
| **Explosiveness** | 36% | When successful, how explosive? (big-play ability via EPA on successful plays) |
| **Turnover Margin** | 10% | Ball-hawking and ball-security skill (regressed to account for luck) |

### Key Features

- **Opponent Adjustment:** Ridge regression solves for true team strength after accounting for schedule difficulty
- **Garbage Time Filtering:** Blowout plays are down-weighted for the trailing team, but kept for the winning team (they earned the dominance)
- **Red Zone Regression:** Early-season red zone variance is smoothed; late-season elite performance is trusted
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

### Multi-Year Backtest (2022-2025)

Walk-forward backtest across 4 seasons. Model trained only on data available at prediction time—no future leakage.

| Phase | Weeks | Games | MAE | ATS % | 5+ Edge |
|-------|-------|-------|-----|-------|---------|
| Calibration | 1-3 | 597 | 14.75 | 47.1% | 48.7% |
| **Core** | 4-15 | 2,485 | 12.54 | 52.0% | **53.2%** |
| Postseason | 16+ | 176 | 13.41 | 45.1% | 48.7% |

**The profitable zone is Weeks 4-15.** Early-season predictions rely too heavily on preseason priors, and bowl games have unmodeled factors (opt-outs, motivation, long layoffs).

### Core Season ATS by Edge (Weeks 4-15)

| Edge | vs Closing Line | vs Opening Line |
|------|-----------------|-----------------|
| All picks | 51.0% (1238-1190) | 53.1% (1277-1130) |
| 3+ pts | 51.8% (727-676) | **54.6%** (783-651) |
| 5+ pts | 53.2% (454-400) | **57.0%** (516-389) |

**Why opening lines are easier to beat:** Opening lines contain more inefficiencies. By closing, sharp money has moved lines toward true value. JP+ captures some of the same information that sharps use. At 5+ point edge, we hit 57% vs opening lines.

### Closing Line Value (CLV)

CLV measures how the market moves after we identify an edge. Positive CLV = sharp money agrees with us.

| Edge Filter | Mean CLV | CLV > 0 | ATS % |
|-------------|----------|---------|-------|
| **All picks** | +0.52 | 41.1% | 51.4% |
| **3+ pt edge** | +0.68 | 42.0% | 53.3% |
| **5+ pt edge** | +0.89 | 43.6% | 54.8% |
| **7+ pt edge** | +1.03 | 43.1% | 55.9% |

At 5+ point edge, the market moves **toward** our prediction by 0.89 points on average. This validates the edge is real—we're finding value that sharps eventually agree with.

### 2025 Season Results

| Metric | Value |
|--------|-------|
| Games (Core) | 638 |
| MAE | 12.16 |
| ATS (Close) | 53.2% |
| 3+ Edge (Close) | 57.0% |
| 5+ Edge (Close) | 55.7% |
| ATS (Open) | 53.5% |
| 3+ Edge (Open) | 57.2% |
| 5+ Edge (Open) | **59.1%** |

2025 was JP+'s best year: lowest MAE (12.16) and highest 5+ edge vs opening lines (59.1%).

### Reality Check

Vegas lines are set by professionals with decades of experience and access to information we don't have (injury reports, locker room intel, sharp money). Beating them consistently is hard. The goal is to find spots where JP+ has an edge, not to win every bet.

---

## What JP+ Is NOT

- **Not a betting system** — It's a tool for analysis, not guaranteed profits
- **Not magic** — College football is chaotic; upsets happen
- **Not complete** — Still being refined and improved

---

## 2025 JP+ Top 25

End-of-season power ratings including CFP:

| Rank | Team | Overall | Off | Def |
|------|------|---------|-----|-----|
| 1 | Ohio State | +27.5 | +12.9 | +14.6 |
| 2 | **Indiana** | +26.8 | +15.1 | +10.7 |
| 3 | Notre Dame | +25.4 | +12.5 | +11.9 |
| 4 | Oregon | +23.4 | +11.4 | +11.8 |
| 5 | Miami | +22.8 | +9.3 | +13.1 |
| 6 | Texas Tech | +22.0 | +3.4 | +17.9 |
| 7 | Texas A&M | +19.2 | +10.1 | +9.8 |
| 8 | Alabama | +18.8 | +7.0 | +11.1 |
| 9 | Georgia | +17.5 | +8.1 | +9.4 |
| 10 | Utah | +17.5 | +11.3 | +6.2 |
| 11 | Vanderbilt | +17.4 | +17.1 | +0.3 |
| 12 | Oklahoma | +16.8 | +0.9 | +16.0 |
| 13 | Missouri | +16.5 | +4.4 | +12.5 |
| 14 | Ole Miss | +16.2 | +12.3 | +4.0 |
| 15 | Washington | +16.1 | +9.3 | +6.6 |
| 16 | Louisville | +15.1 | +6.1 | +8.6 |
| 17 | Tennessee | +14.1 | +12.1 | +1.6 |
| 18 | James Madison | +14.1 | +3.2 | +11.0 |
| 19 | BYU | +13.7 | +7.7 | +5.6 |
| 20 | Texas | +13.4 | +3.5 | +9.4 |
| 21 | South Florida | +12.7 | +6.4 | +5.7 |
| 22 | Florida State | +12.3 | +10.9 | +1.5 |
| 23 | Penn State | +12.0 | +7.9 | +4.1 |
| 24 | USC | +11.8 | +12.6 | -0.9 |
| 25 | Auburn | +11.5 | +3.0 | +8.1 |

**Indiana** — National Champions. Beat Alabama 38-3, Oregon 56-22, and Miami 27-21 in CFP.

**Spread Calculation:** Team A rating - Team B rating = expected point spread (before adjustments).

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
