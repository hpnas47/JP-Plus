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

### Special Teams (Field Goal Efficiency)

JP+ includes a field goal efficiency adjustment based on kicker performance vs expectation. The model calculates Points Above Average Expected (PAAE) for each team's kicking:

- **Expected make rates by distance:** <30 yards (92%), 30-40 (83%), 40-50 (72%), 50-55 (55%), 55+ (30%)
- **PAAE calculation:** Actual points scored (3 if made, 0 if missed) minus expected points (3 × expected rate)
- **Per-game rating:** Total PAAE divided by games played

For example, a kicker who makes a 48-yard field goal earns +0.84 PAAE (got 3 points, expected 2.16). Missing the same kick costs -2.16 PAAE.

This captures the reality that kicker quality varies significantly—elite kickers like Auburn's 2024 squad add real value, while inconsistent kickers cost their teams points.

### Adjustments Layer

After calculating base ratings, JP+ adjusts for:
- **Home field advantage** (1.5 - 4.0 points, team-specific)
- **Travel distance** (cross-country trips hurt)
- **Altitude** (playing at BYU, Air Force, or Colorado is tough)
- **Situational factors** (bye weeks, lookahead spots, letdown games)
- **FCS opponent penalty** (tiered: +18 pts for elite FCS, +32 pts for standard FCS)
- **FG efficiency differential** (kicker PAAE difference between teams)
- **Pace adjustment** (spread compression for triple-option teams)

**Team-Specific HFA:** Not all home fields are equal. LSU at night (4.0 pts) is much tougher than playing at Kent State (1.75 pts). JP+ uses curated HFA values for ~50 teams based on stadium environment, with conference-based defaults for others.

**Trajectory Modifier:** HFA isn't static—it changes as programs rise or fall. A team that's dramatically improved (like Vanderbilt or Indiana in 2024) will have more energized crowds and a stronger home environment. JP+ compares the prior year's win rate to the historical baseline (3 years before) and adjusts HFA by up to ±0.5 points. This is calculated once at the start of each season and locked in. Rising programs get a boost; declining programs get a penalty.

**FCS Penalty (Tiered):** When an FBS team plays an FCS opponent, JP+ applies a tiered penalty based on FCS team quality. **Elite FCS teams** (North Dakota State, Montana State, South Dakota State, Sacramento State, and other FCS playoff regulars) receive an 18-point penalty. **Standard FCS teams** receive a 32-point penalty. This tiered approach recognizes that elite FCS programs routinely compete with lower-tier FBS teams, while standard FCS opponents are dramatically weaker. The tiered system improved 5+ edge ATS from 56.0% to 56.9%.

**Pace Adjustment (Triple-Option):** Triple-option teams (Army, Navy, Air Force, Kennesaw State) run significantly fewer plays per game (~55 vs ~70 for standard offenses). This creates more variance in outcomes—analysis shows 30% worse MAE for triple-option games (16.09 vs 12.36, p=0.001). To account for this reduced game volume, JP+ compresses spreads by 10% toward zero when a triple-option team is involved (15% if both teams run triple-option). This reflects the fundamental uncertainty in games with fewer possessions.

**Triple-Option Rating Boost:** Triple-option teams (especially service academies) are systematically underrated by efficiency metrics like SP+ because EPA calculations don't fully capture their scheme's value. Additionally, service academies have artificially low recruiting rankings due to unique constraints (service commitment, physical requirements) that don't reflect their actual competitiveness. JP+ applies a +6 point boost to raw SP+ ratings for these teams and uses 100% prior rating (no talent blend) to correct this systematic bias.

### Preseason Priors

In weeks 1-4, there's insufficient current-season data. JP+ blends in:
- Previous year's SP+ ratings (60%) - adjusted for returning production
- Recruiting rankings / talent composite (40%)

**Asymmetric Regression:** Standard regression pulls all teams toward the mean uniformly—but this compresses the true spread between elite and terrible teams. A team rated -25 (very bad) shouldn't gain 7 points from regression. JP+ applies asymmetric regression: teams far from the mean (±20 pts) regress only ~10% vs the normal ~30% for mid-tier teams. This preserves the true gap between Ohio State (+27) and Kent State (-25), improving blowout game accuracy.

**Extremity-Weighted Talent Blend:** For extreme teams (20+ pts from mean), the talent weight is reduced from 40% to 20%. Talent composites are more compressed than performance—a terrible team isn't terrible because of talent alone (scheme, coaching, roster construction matter). Trusting proven performance for outliers prevents artificial spread compression.

**Returning Production Adjustment:** Teams returning most of their production (high % of PPA returning) keep more of their prior rating. Teams with heavy roster turnover regress more toward the mean. This prevents overvaluing teams that lost key players.

**Transfer Portal Integration:** The returning production metric only counts players who stayed—but what about incoming transfers? JP+ fetches transfer portal data and calculates net production impact (incoming PPA minus outgoing PPA). This captures portal stars like Riley Leonard transferring to Notre Dame or Carson Beck moving to Miami. Top portal winners get a boost to their effective returning production; portal losers get a penalty.

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
- Special teams chaos

These things don't repeat consistently.

### The Solution

By analyzing every play individually:
- We filter out garbage time (blowout situations)
- We measure what teams *do*, not what *happens to them*
- We get a more stable, predictive signal

---

## JP+ Performance

### 2025 Season Results

| Metric | Value |
|--------|-------|
| MAE | 12.21 points |
| Overall ATS | 51.9% (325-301) |

#### ATS by Edge Threshold (2025)

| Edge | vs Closing Line | vs Opening Line |
|------|-----------------|-----------------|
| 3+ pts | 54.1% (172-146) | **55.3%** (189-153) |
| 5+ pts | 55.4% (98-79) | **58.3%** (119-85) |

### Multi-Year Results (2022-2025)

| Years | MAE | Overall ATS % |
|-------|-----|---------------|
| 2022-2025 | 12.37 | 51.0% |

#### ATS Performance by Edge Threshold (Multi-Year, vs Closing)

| Edge | Record | ATS % |
|------|--------|-------|
| 3+ pts | 612-521 | **54.0%** |
| 5+ pts | 365-272 | **57.3%** |

**Why opening lines are easier to beat:** Opening lines contain more inefficiencies. By closing, sharp money has moved lines toward true value. JP+ captures some of the same information that sharps use, so we see better performance against openers.

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

## Data Source

All data comes from the College Football Data API (collegefootballdata.com), which provides:
- Game scores and locations
- Play-by-play data with EPA values
- Vegas betting lines
- Team rosters and recruiting rankings

---

## Summary

JP+ rates college football teams by analyzing play-by-play efficiency rather than final scores. It adjusts for opponent strength, home field, and situational factors to predict point spreads. When JP+ significantly disagrees with Vegas, that's worth investigating as a potential betting opportunity.

The philosophy: **Measure the process (efficiency), not just the outcome (points).**
