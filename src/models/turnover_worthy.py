"""Turnover-Worthy Plays (TWP) Model.

Proxies turnover-worthy plays using CFBD data to identify teams that have been
lucky or unlucky with turnovers. This measures PROCESS (risky plays) rather
than OUTCOMES (actual turnovers).

Key insight: A QB who throws 10 interceptable passes but only gets picked 2 times
is lucky. Regression is coming. This model identifies that.

Proxy signals from CFBD:
- Pass breakups ("broken up" in play_text) = near-interceptions
- Sacks = fumble-risk situations (~10% fumble rate historically)
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Historical fumble rate on sacks (NFL/CFB average ~10-12%)
SACK_FUMBLE_RATE = 0.10

# Historical INT rate on contested passes (from our data: ~52%)
# But this includes actual INTs, so for breakups only, the "luck" factor is
# that they COULD have been picked. We'll use a lower rate for expected conversion.
CONTESTED_PASS_INT_RATE = 0.40  # Conservative estimate


@dataclass
class TeamTWPMetrics:
    """Turnover-worthy play metrics for a team."""

    team: str

    # Offensive TWP (bad for team)
    pass_breakups_against: int  # Near-INTs thrown
    interceptions_thrown: int   # Actual INTs
    interceptable_passes: int   # Total risky throws (breakups + INTs)
    sacks_taken: int            # Fumble-risk situations
    fumbles_lost: int           # Actual fumbles lost

    # Defensive TWP (good for team)
    pass_breakups_made: int     # Near-INTs forced
    interceptions_caught: int   # Actual INTs caught
    sacks_made: int             # Fumble-risk situations created
    fumbles_recovered: int      # Opponent fumbles recovered

    # Calculated luck metrics
    expected_ints_thrown: float
    int_luck: float  # Negative = lucky (fewer INTs than expected)
    expected_fumbles_lost: float
    fumble_luck: float  # Negative = lucky

    # Combined
    total_turnover_luck: float  # Negative = lucky overall
    luck_adjustment: float  # Points to add to rating (positive = was unlucky, deserves boost)


class TurnoverWorthyModel:
    """Model turnover-worthy plays to identify luck.

    This creates an adjustment that can be applied to team ratings:
    - Teams that were lucky (fewer TOs than expected) get negative adjustment
    - Teams that were unlucky (more TOs than expected) get positive adjustment
    """

    # Points per turnover (standard value)
    POINTS_PER_TURNOVER = 4.5

    def __init__(
        self,
        sack_fumble_rate: float = SACK_FUMBLE_RATE,
        contested_int_rate: float = CONTESTED_PASS_INT_RATE,
        luck_regression_pct: float = 0.5,  # How much to regress (0.5 = half)
    ):
        """Initialize TWP model.

        Args:
            sack_fumble_rate: Expected fumble rate on sacks
            contested_int_rate: Expected INT rate on contested passes (breakups)
            luck_regression_pct: How much of the luck to regress (0.5 = 50%)
        """
        self.sack_fumble_rate = sack_fumble_rate
        self.contested_int_rate = contested_int_rate
        self.luck_regression_pct = luck_regression_pct

        self.team_metrics: dict[str, TeamTWPMetrics] = {}

    def _parse_plays(self, plays_df: pd.DataFrame) -> pd.DataFrame:
        """Parse plays to extract TWP signals.

        Args:
            plays_df: Play-by-play DataFrame with play_type and play_text

        Returns:
            DataFrame with TWP flags added
        """
        df = plays_df.copy()

        # Flag pass breakups (near-INTs)
        df["is_pass_breakup"] = df["play_text"].str.contains(
            r"broken up|broke up",
            case=False,
            na=False,
            regex=True
        )

        # Flag interceptions
        df["is_interception"] = df["play_type"].str.contains(
            "Interception",
            case=False,
            na=False
        )

        # Flag sacks
        df["is_sack"] = df["play_type"].str.lower() == "sack"

        # Flag fumbles lost (opponent recovered)
        df["is_fumble_lost"] = df["play_type"].str.contains(
            r"Fumble Recovery \(Opponent\)|Fumble Return",
            case=False,
            na=False,
            regex=True
        )

        # Flag fumbles recovered (own team)
        df["is_fumble_recovered"] = df["play_type"].str.contains(
            r"Fumble Recovery \(Own\)",
            case=False,
            na=False,
            regex=True
        )

        return df

    def calculate_team_metrics(
        self,
        plays_df: pd.DataFrame,
        games_df: Optional[pd.DataFrame] = None,
    ) -> dict[str, TeamTWPMetrics]:
        """Calculate TWP metrics for all teams.

        Args:
            plays_df: Play-by-play DataFrame
            games_df: Optional games DataFrame (for filtering)

        Returns:
            Dict mapping team name to TWPMetrics
        """
        # Parse plays for TWP signals
        df = self._parse_plays(plays_df)

        # Get all teams
        all_teams = set(df["offense"].dropna()) | set(df["defense"].dropna())

        for team in all_teams:
            # Offensive plays (team on offense)
            off_plays = df[df["offense"] == team]

            # Defensive plays (team on defense)
            def_plays = df[df["defense"] == team]

            # Offensive TWP (bad for team)
            pass_breakups_against = off_plays["is_pass_breakup"].sum()
            interceptions_thrown = off_plays["is_interception"].sum()
            sacks_taken = off_plays["is_sack"].sum()
            fumbles_lost = off_plays["is_fumble_lost"].sum()

            # Defensive TWP (good for team)
            pass_breakups_made = def_plays["is_pass_breakup"].sum()
            interceptions_caught = def_plays["is_interception"].sum()
            sacks_made = def_plays["is_sack"].sum()
            fumbles_recovered = def_plays["is_fumble_lost"].sum()  # Opponent lost = we recovered

            # Calculate expected turnovers
            # For INTs: breakups are near-INTs, so expected INTs = actual + (breakups × conversion rate)
            # But we want to compare to actual, so:
            # Expected INTs given risky throws = (breakups + actual_INTs) × league_avg_rate
            # Actually simpler: expected_additional_INTs = breakups × contested_int_rate
            # Luck = expected_additional - 0 (they got away with breakups)
            # Or: total risky throws = breakups + INTs
            #     expected INTs from risky throws = risky × some_rate
            #     luck = expected - actual

            interceptable_passes = pass_breakups_against + interceptions_thrown

            # Expected INTs if interceptable passes converted at the contested rate
            # For breakups, they "should have" been picked at contested_int_rate
            expected_ints = pass_breakups_against * self.contested_int_rate + interceptions_thrown
            int_luck = interceptions_thrown - expected_ints  # Negative = lucky (fewer than expected)

            # Actually, let me rethink this:
            # - Team threw X interceptable passes (breakups + INTs)
            # - Of those, Y were actually intercepted
            # - League average INT rate on interceptable passes = ~50%
            # - Expected INTs = X × 0.5
            # - Luck = Y - (X × 0.5)
            # If Y > X×0.5, team was unlucky (more INTs than expected)
            # If Y < X×0.5, team was lucky (fewer INTs than expected)

            if interceptable_passes > 0:
                expected_ints_v2 = interceptable_passes * self.contested_int_rate
                int_luck = interceptions_thrown - expected_ints_v2
            else:
                expected_ints_v2 = 0
                int_luck = 0

            # Fumble luck
            # Sacks have ~10% fumble rate
            # Plus some baseline fumble rate on other plays (we'll ignore for simplicity)
            expected_fumbles = sacks_taken * self.sack_fumble_rate
            fumble_luck = fumbles_lost - expected_fumbles  # Negative = lucky

            # Also consider defensive side (team's defense forcing turnovers)
            def_interceptable = pass_breakups_made + interceptions_caught
            if def_interceptable > 0:
                def_expected_ints = def_interceptable * self.contested_int_rate
                def_int_luck = interceptions_caught - def_expected_ints
            else:
                def_int_luck = 0

            def_expected_fumbles = sacks_made * self.sack_fumble_rate
            def_fumble_luck = fumbles_recovered - def_expected_fumbles

            # Total luck (from team's perspective):
            # - Offensive bad luck = more TOs than expected (positive int_luck, fumble_luck)
            # - Defensive good luck = more TOs forced than expected (positive def_int_luck, def_fumble_luck)
            # Net luck = (def_luck) - (off_luck)
            # Positive = lucky overall (defense got more, offense lost fewer)

            off_turnover_luck = int_luck + fumble_luck  # Positive = unlucky offense
            def_turnover_luck = def_int_luck + def_fumble_luck  # Positive = lucky defense

            total_turnover_luck = def_turnover_luck - off_turnover_luck
            # Positive total = net lucky (got more turnovers than gave up, adjusted for process)

            # Luck adjustment: if team was lucky, they should regress (negative adjustment)
            # Points impact = turnovers × points_per_turnover × regression_pct
            luck_adjustment = -total_turnover_luck * self.POINTS_PER_TURNOVER * self.luck_regression_pct

            self.team_metrics[team] = TeamTWPMetrics(
                team=team,
                pass_breakups_against=int(pass_breakups_against),
                interceptions_thrown=int(interceptions_thrown),
                interceptable_passes=int(interceptable_passes),
                sacks_taken=int(sacks_taken),
                fumbles_lost=int(fumbles_lost),
                pass_breakups_made=int(pass_breakups_made),
                interceptions_caught=int(interceptions_caught),
                sacks_made=int(sacks_made),
                fumbles_recovered=int(fumbles_recovered),
                expected_ints_thrown=expected_ints_v2,
                int_luck=int_luck,
                expected_fumbles_lost=expected_fumbles,
                fumble_luck=fumble_luck,
                total_turnover_luck=total_turnover_luck,
                luck_adjustment=luck_adjustment,
            )

        logger.info(f"Calculated TWP metrics for {len(self.team_metrics)} teams")
        return self.team_metrics

    def get_luck_adjustment(self, team: str) -> float:
        """Get the luck adjustment for a team.

        Args:
            team: Team name

        Returns:
            Points adjustment (positive = was unlucky, deserves boost)
        """
        if team in self.team_metrics:
            return self.team_metrics[team].luck_adjustment
        return 0.0

    def get_metrics_df(self) -> pd.DataFrame:
        """Get all team metrics as a DataFrame.

        Returns:
            DataFrame with TWP metrics for all teams
        """
        if not self.team_metrics:
            return pd.DataFrame()

        data = []
        for team, m in self.team_metrics.items():
            data.append({
                "team": team,
                "interceptable_passes": m.interceptable_passes,
                "ints_thrown": m.interceptions_thrown,
                "expected_ints": round(m.expected_ints_thrown, 1),
                "int_luck": round(m.int_luck, 1),
                "sacks_taken": m.sacks_taken,
                "fumbles_lost": m.fumbles_lost,
                "expected_fumbles": round(m.expected_fumbles_lost, 1),
                "fumble_luck": round(m.fumble_luck, 1),
                "total_luck": round(m.total_turnover_luck, 2),
                "luck_adj_pts": round(m.luck_adjustment, 1),
            })

        df = pd.DataFrame(data)
        return df.sort_values("luck_adj_pts", ascending=False)

    def print_luck_report(self, top_n: int = 10) -> None:
        """Print a luck report showing luckiest and unluckiest teams.

        Args:
            top_n: Number of teams to show in each category
        """
        df = self.get_metrics_df()

        if df.empty:
            print("No TWP data calculated yet.")
            return

        print("=" * 70)
        print("TURNOVER-WORTHY PLAYS LUCK REPORT")
        print("=" * 70)
        print("\nPositive luck_adj = team was UNLUCKY (rating boost)")
        print("Negative luck_adj = team was LUCKY (rating penalty)")

        print(f"\n--- UNLUCKIEST TEAMS (deserve rating boost) ---")
        unlucky = df.nlargest(top_n, "luck_adj_pts")
        for _, row in unlucky.iterrows():
            print(
                f"  {row['team']:<20} adj={row['luck_adj_pts']:+5.1f}pts  "
                f"(INT luck={row['int_luck']:+.1f}, fumble luck={row['fumble_luck']:+.1f})"
            )

        print(f"\n--- LUCKIEST TEAMS (deserve rating penalty) ---")
        lucky = df.nsmallest(top_n, "luck_adj_pts")
        for _, row in lucky.iterrows():
            print(
                f"  {row['team']:<20} adj={row['luck_adj_pts']:+5.1f}pts  "
                f"(INT luck={row['int_luck']:+.1f}, fumble luck={row['fumble_luck']:+.1f})"
            )
