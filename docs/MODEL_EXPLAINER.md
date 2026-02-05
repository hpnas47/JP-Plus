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

## How JP+ Works

### The Core Idea

Most people look at final scores to judge teams. But final scores are noisy—a team can win by 21 because they played great, or because the other team threw 4 interceptions that won't happen again.

Instead, JP+ looks at **how efficiently teams move the ball on every single play**.

### What We Measure

**1. Success Rate (54% of rating)**

On each play, did the offense "succeed"?
- 1st down: Gain at least 50% of yards needed
- 2nd down: Gain at least 70% of yards needed
- 3rd/4th down: Get the first down or touchdown

A team that succeeds on 45% of plays is consistently moving the chains. A team at 38% is struggling. This is more predictive than points because it measures *process*, not just *outcomes*.

**2. Explosiveness (36% of rating)**

When a team does succeed, how explosive are they? This is measured by IsoPPP (Isolated Points Per Play)—the average EPA on successful plays only. It captures big-play ability without being fooled by garbage time touchdowns.

**3. Turnover Margin (10% of rating)**

Turnovers are a systematic skill that pure efficiency metrics miss. JP+ tracks per-game turnover margin (turnovers forced minus turnovers lost) and converts it to points using 4.5 points per turnover.

Because turnover margin is 50-70% luck (fumble bounces, tipped passes), JP+ applies Bayesian shrinkage toward zero. A team with 15 games keeps ~60% of their raw margin; a team with only 5 games keeps ~33%. This prevents early-season outliers from distorting ratings while trusting sustained end-of-season performance.

A team like Indiana (+15 turnover margin in 2025) or Notre Dame (+17) gains a real advantage by forcing mistakes and protecting the ball. Ohio State's narrower +3 margin means their efficiency edge is partially offset by weaker ball-hawking and ball-security.

### Opponent Adjustment

Raw efficiency stats are misleading—a team playing cupcakes looks better than one playing a tough schedule. JP+ uses ridge regression to solve for each team's true offensive and defensive efficiency after accounting for opponent strength.

**FBS-Only Filtering:** The regression only includes plays from FBS vs FBS matchups. When an FBS team plays an FCS opponent (like Indiana vs Indiana State), those plays are excluded from the regression. FCS teams have too few games against FBS opponents to estimate reliable strength coefficients. FCS games are handled separately via the FCS Penalty adjustment.

### Garbage Time Filtering (Asymmetric)

Plays in blowout situations are handled asymmetrically:
- **Winning team**: Keeps full weight (they earned the blowout through efficiency)
- **Trailing team**: Down-weighted to 10% (garbage time noise)

Garbage time thresholds:
- 28+ points in the 2nd half
- 21+ points in the 3rd quarter
- 14+ points in the 4th quarter

This asymmetric approach rewards teams that maintain dominance throughout games (like Indiana's 56% SR in garbage time) while filtering noise from trailing teams. Traditional symmetric filtering was hiding signal from truly dominant teams.

### Red Zone Regression (Finishing Drives)

Red zone TD rate has some variance early in the season, but over 15 games becomes a reliable signal of scheme and talent. JP+ applies light Bayesian regression:
- Getting TO the red zone is a sustainable skill (captured by Success Rate)
- Scoring TDs IN the red zone has variance in small samples (early season)
- By late season (150+ RZ plays), elite red zone efficiency (like Indiana's 87%) is genuine skill
- The regression (prior_strength=10) trusts actual performance while smoothing early-season noise

### Special Teams (PBTA - Points Better Than Average)

JP+ includes a complete special teams model that captures the marginal point contribution of each unit compared to a league-average unit. All components are expressed as PBTA—positive means the unit gains points for the team, negative means it costs points.

**Components:**

**1. Field Goals (PAAE)**
- **Expected make rates by distance:** <30 yards (92%), 30-40 (83%), 40-50 (72%), 50-60 (55%), 60+ (30%)
- **Calculation:** Actual points (3 if made, 0 if missed) minus expected points (3 × expected rate)
- **Per-game rating:** Total PAAE divided by games played

**2. Punting (Field Position Value)**
- Net yards vs expected (40 yards), converted to points at ~0.04 pts/yard
- Inside-20 bonus: +0.5 pts (better field position for defense)
- Touchback penalty: -0.3 pts (opponent starts at 25 instead of worse)

**3. Kickoffs (Coverage + Returns)**
- Coverage: Touchback rate vs expected (60%) + return yards allowed vs expected (23 yds)
- Returns: Return yards gained vs expected (23 yds)
- All converted to points at ~0.04 pts/yard of field position

**Overall ST Rating = FG + Punt + Kickoff** (all in points per game)

**Example:** Vanderbilt 2024 had the best ST unit at +2.34 pts/game PBTA—their special teams gained them over 2 points per game compared to an average unit. UTEP at -2.83 pts/game was costing their team nearly 3 points per game.

**FBS Distribution:** Mean ~0, Std ~1.0, 95% of teams fall within ±2 pts/game.

---

## Adjustments Layer

After calculating base efficiency ratings, JP+ applies game-specific adjustments. These fall into four categories:

### Game Context Adjustments

**Home Field Advantage (1.5 - 4.0 pts):** Not all home fields are equal. LSU at night (4.0 pts) is much tougher than Kent State (1.75 pts). JP+ uses curated HFA values for ~50 teams based on stadium environment, with conference-based defaults for others.

- **Trajectory Modifier:** HFA changes as programs rise or fall. Vanderbilt/Indiana in 2024 got +0.5 pt HFA boost for their newfound competitive environment.

**Travel (0.25 - 2.5 pts):** Cross-country travel hurts teams:
- **Timezone:** ~0.5 pts/zone (slightly less going west)
- **Distance:** 0.25-1.0 pts (300mi to 2000+ mi thresholds)
- **Dampening:** Short-distance games crossing timezone lines (<400mi) get no TZ penalty—no fatigue for a 75-mile trip

**Altitude (0 - 3 pts):** BYU (4,551 ft), Air Force (6,621 ft), and Colorado (5,328 ft) create oxygen debt for visiting sea-level teams.

**Correlated Stack Smoothing:** When HFA + travel + altitude all favor home team, they stack. Games with >5 pts combined adjustment over-predicted home margins by ~2.3 pts. JP+ applies soft cap: excess above 5 pts reduced by 50%.

### Scheduling Adjustments

**Rest Differential (±0.5 pts/day, capped at ±1.5 pts):** CFB isn't just Saturdays—MACtion and short weeks matter:

| Scenario | Days Rest | Example |
|----------|-----------|---------|
| Bye Week | 14+ days | Team didn't play previous week |
| Mini-Bye | 9-10 days | Thursday → Saturday |
| Normal | 6-7 days | Saturday → Saturday |
| Short Week | 4-5 days | Saturday → Thursday |

Formula: `(home_rest - away_rest) × 0.5 pts/day`. Example: Oregon (9 days after Thursday game) vs Texas (7 days after Saturday game) = +1.0 pts for Oregon.

**Letdown Spot (-2.0 to -2.5 pts):** Team had a "big win" last week, now facing unranked opponent. Big win = beat top-15 team OR beat arch-rival (rivalry hangover). *Uses historical rankings at time of game, not current rankings.*

- **Home letdown:** -2.0 pts (crowd keeps team engaged)
- **Away letdown:** -2.5 pts (sleepy road game multiplier)

**Lookahead Spot (-1.5 pts):** Team has a rival or top-10 opponent next week.

**Sandwich Spot (extra -1.0 pts):** The most dangerous scheduling spot in CFB—when BOTH letdown AND lookahead apply. Team just had a big win and has a big game on deck next week. The unranked team in the middle is the "meat" of the sandwich. Total penalty: -4.5 to -5.0 pts.

**Rivalry Boost (+1.0 pts):** Underdog in rivalry game only.

### Opponent & Pace Adjustments

**FCS Penalty (Tiered):** When FBS plays FCS:
- **Elite FCS (+18 pts):** NDSU, Montana State, South Dakota State, Sacramento State, etc.
- **Standard FCS (+32 pts):** All other FCS teams

**Special Teams Differential:** Full ST PBTA difference (FG + Punt + Kickoff) applied to spread.

**Pace Adjustment (Triple-Option):** Army, Navy, Air Force run ~55 plays/game vs ~70 normal. This creates variance (30% worse MAE). JP+ compresses spreads 10% toward zero for triple-option games.

**Triple-Option Rating Boost (+6 pts):** Service academies are systematically underrated by efficiency metrics. JP+ boosts their raw ratings and uses 100% prior (no talent blend) to correct this.

### Weather Adjustments (Totals Only)

| Factor | Threshold | Adjustment |
|--------|-----------|------------|
| Wind | >10 mph | -0.3 pts/mph (cap: -6.0) |
| Temperature | <40°F | -0.15 pts/degree (cap: -4.0) |
| Precipitation | >0.02 in | -3.0 pts flat |
| Heavy Precip | >0.05 in | -5.0 pts flat |

Indoor games receive no weather adjustment.

### Preseason Priors

In weeks 1-4, there's insufficient current-season data. JP+ blends in:
- Previous year's SP+ ratings (60%) - adjusted for returning production
- Recruiting rankings / talent composite (40%)

**Asymmetric Regression:** Standard regression pulls all teams toward the mean uniformly—but this compresses the true spread between elite and terrible teams. A team rated -25 (very bad) shouldn't gain 7 points from regression. JP+ applies asymmetric regression: teams far from the mean (±20 pts) regress only ~10% vs the normal ~30% for mid-tier teams. This preserves the true gap between Ohio State (+27) and Kent State (-25), improving blowout game accuracy.

**Extremity-Weighted Talent Blend:** For extreme teams (20+ pts from mean), the talent weight is reduced from 40% to 20%. Talent composites are more compressed than performance—a terrible team isn't terrible because of talent alone (scheme, coaching, roster construction matter). Trusting proven performance for outliers prevents artificial spread compression.

**Returning Production Adjustment:** Teams returning most of their production (high % of PPA returning) keep more of their prior rating. Teams with heavy roster turnover regress more toward the mean. This prevents overvaluing teams that lost key players.

**Transfer Portal Integration:** The returning production metric only counts players who stayed—but what about incoming transfers? JP+ uses a **unit-level approach** that values 100% of portal activity using:

- **Scarcity-based position weights:** OT (0.90) and QB (1.00) are premium; EDGE/IDL (0.75) are anchors; skill positions (WR, RB, CB) are 0.40-0.45 reflecting higher replacement rates
- **Level-up discounts:** G5→P4 trench players get a 25% "physicality tax"; skill players get 10% discount; P4→P4 keeps full value
- **Continuity tax:** Losing incumbents hurts ~11% more than raw value (chemistry, scheme fit)
- **Impact cap:** ±12% maximum team-wide adjustment

This captures portal stars like Carson Beck (4-star QB to Miami) while properly discounting G5 trench players moving up. Blue Bloods hitting the portal cap (Alabama, Georgia at -12%) show minimal final rating impact (~0.3 pts) because their elite talent composite provides the expected offset.

**Coaching Change Regression:** When a talented team underperformed under a bad coach (think Florida under Napier), and a new coach arrives, JP+ "forgets" some of that underperformance. The model shifts weight from prior performance toward talent—essentially saying "this team *should* be better based on their roster."

The adjustment depends on:
- **Talent-performance gap** - Bigger gap = more forgetting (Florida at talent #8 but performing #45 gets a big boost)
- **Coach pedigree** - Proven coaches (Kiffin, Cignetti, Sumrall) get more benefit of the doubt than first-time HCs

*Caveat: Coach pedigree uses career win %, which embeds "opportunity" (better jobs → higher win %), not purely skill. This is a qualitative signal with limited statistical power due to small sample size (~10-15 relevant coaching changes per year).*

This prevents JP+ from being too slow to recognize turnarounds like Indiana 2024 (Cignetti) or projecting improvement at Florida 2025 (Sumrall).

By week 8, the model is 85%+ based on current season data. By week 12, preseason priors are fully phased out.

---

## Why This Approach?

### The Problem with Margin-Based Models

If you just look at "Team A beat Team B by 21 points," you're including:
- Turnover luck (fumble bounces are random)
- Garbage time scores (backups vs backups)
- Weather flukes

These things don't repeat consistently.

### The Solution

By analyzing every play individually:
- We filter out garbage time (blowout situations)
- We measure what teams *do*, not what *happens to them*
- We capture special teams skill (kicking, punting, returns) as a separate component
- We get a more stable, predictive signal

---

## JP+ Performance

### Multi-Year Backtest (2022-2025)

Walk-forward backtest across 4 seasons. Model trained on data available at prediction time—no future leakage.

| Phase | Weeks | Games | MAE | ATS % | 5+ Edge |
|-------|-------|-------|-----|-------|---------|
| Calibration | 1-3 | 597 | 14.75 | 47.1% | 48.7% |
| **Core** | 4-15 | 2,485 | 12.54 | 50.8% | **52.8%** |
| Postseason | 16+ | 176 | 13.41 | 45.1% | 48.7% |

**The profitable zone is Weeks 4-15** (Core season). Early-season predictions rely too heavily on preseason priors, and bowl games have unmodeled factors (opt-outs, motivation, long layoffs).

#### Core Season ATS by Edge (Weeks 4-15)

| Edge | vs Closing Line | vs Opening Line |
|------|-----------------|-----------------|
| All picks | 51.0% (1238-1190) | 53.1% (1277-1130) |
| 3+ pts | 51.8% (727-676) | **54.6%** (783-651) |
| 5+ pts | 53.2% (454-400) | **57.0%** (516-389) |

**Why opening lines are easier to beat:** Opening lines contain more inefficiencies. By closing, sharp money has moved lines toward true value. JP+ captures some of the same information that sharps use, so we see better performance against openers. At 5+ point edge, we hit 57% vs opening lines.

### Accuracy by Game Margin

JP+ is most accurate on close games—exactly where ATS bets are won or lost:

| Actual Margin | Games | MAE |
|---------------|-------|-----|
| 0-7 (close) | 890 (36%) | 7.33 |
| 8-14 | 485 (19%) | 9.31 |
| 15-21 | 397 (16%) | 12.94 |
| 22-28 | 298 (12%) | 18.53 |
| 29+ (blowout) | 419 (17%) | 26.08 |

Blowouts are where JP+ struggles most—but that's expected. Garbage time makes final margins in blowouts essentially random. The 17% of games that are 29+ point blowouts contribute 33% of the overall MAE.

### Reality Check

Vegas lines are set by professionals with decades of experience and access to information we don't have (injury reports, locker room intel, sharp money). Beating them consistently is hard. The goal is to find spots where JP+ has an edge, not to win every bet.

---

## What JP+ Is NOT

- **Not a betting system** - It's a tool for analysis, not guaranteed profits
- **Not magic** - College football is chaotic; upsets happen
- **Not complete** - Still being refined and improved

---

## 2025 JP+ Top 25

End-of-season power ratings including CFP (normalized for direct spread calculation):

| Rank | Team | Overall | Off | Def |
|------|------|---------|-----|-----|
| 1 | Ohio State | +27.5 | +12.9 | +14.6 |
| 2 | **Indiana** ★ | +26.8 | +15.1 | +10.7 |
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

★ **National Champions** - Indiana beat Alabama 38-3, Oregon 56-22, and Miami 27-21 in CFP.

**Spread Calculation:** Team A rating - Team B rating = expected point spread (before adjustments).

Example: Ohio State (+27.5) vs Indiana (+26.8) → Ohio State favored by 0.7 points at neutral site.

Ratings are normalized with mean=0 and std=12 across FBS teams. Higher = better. Ratings include all regular season and postseason (bowl/CFP) play-by-play data.

JP+ exposes separate offensive, defensive, and special teams ratings via `get_ratings_df()` for detailed analysis.

---

## Data Sources

### Play-by-Play and Game Data

All efficiency data comes from the **College Football Data API** (collegefootballdata.com):
- Game scores and locations
- Play-by-play data with EPA values
- Team rosters and recruiting rankings
- Weather data (temperature, wind, precipitation, indoor flag)
- SP+ and FPI ratings (for external comparison/validation)

### Betting Lines

JP+ uses a dual-source approach for betting lines:

**Historical (2022-2025):** CFBD API
- Aggregates lines from DraftKings, ESPN Bet, Bovada, and others
- 91% of FBS games have opening line data
- Used for backtesting ATS performance

**Future (2026+):** The Odds API (the-odds-api.com)
- Captures opening lines Sunday evening after posting
- Captures closing lines Saturday morning before games
- Primary sportsbooks: FanDuel, DraftKings, BetMGM, Caesars, Bovada
- Data stored locally in `data/odds_api_lines.db`

The `src/api/betting_lines.py` module merges both sources, preferring The Odds API data when available for better opening line coverage.

---

## Summary

JP+ rates college football teams by analyzing play-by-play efficiency rather than final scores. It adjusts for opponent strength, home field, and situational factors to predict point spreads. When JP+ significantly disagrees with Vegas, that's worth investigating as a potential betting opportunity.

The philosophy: **Measure the process (efficiency), not just the outcome (points).**
