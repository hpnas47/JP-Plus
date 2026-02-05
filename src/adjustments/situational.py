"""Situational adjustments for bye weeks, letdown/lookahead spots, and rivalries."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from config.settings import get_settings
from config.teams import is_rivalry_game

logger = logging.getLogger(__name__)


class HistoricalRankings:
    """Week-by-week historical poll rankings.

    CRITICAL: Rankings are volatile in CFB. A team ranked #15 in Week 3 may be
    unranked by Week 8, or vice versa. When evaluating situational spots like
    "letdown after beating a ranked team", we must use the ranking at the TIME
    of the game, not the current ranking.

    This class stores rankings by week and provides lookup methods.
    """

    def __init__(self, poll_name: str = "AP Top 25"):
        """Initialize with preferred poll name.

        Args:
            poll_name: Poll to use ("AP Top 25", "Coaches Poll", "Playoff Committee Rankings")
        """
        self.poll_name = poll_name
        # week -> {team: rank}
        self._rankings_by_week: dict[int, dict[str, int]] = {}

    def load_from_api(self, client, year: int, max_week: int = 16) -> None:
        """Load historical rankings from CFBD API.

        Args:
            client: CFBDClient instance
            year: Season year
            max_week: Maximum week to load
        """
        try:
            all_rankings = client.get_rankings(year)

            for poll_week in all_rankings:
                week = poll_week.week

                for poll in poll_week.polls:
                    if poll.poll == self.poll_name:
                        week_rankings = {}
                        for rank_entry in poll.ranks:
                            week_rankings[rank_entry.school] = rank_entry.rank
                        self._rankings_by_week[week] = week_rankings
                        break

            logger.info(
                f"Loaded {self.poll_name} rankings for {len(self._rankings_by_week)} weeks"
            )
        except Exception as e:
            logger.warning(f"Failed to load historical rankings: {e}")

    def set_week_rankings(self, week: int, rankings: dict[str, int]) -> None:
        """Manually set rankings for a specific week.

        Args:
            week: Week number
            rankings: Dict of team -> rank
        """
        self._rankings_by_week[week] = rankings

    def get_rank(self, team: str, week: int) -> Optional[int]:
        """Get a team's ranking for a specific week.

        Args:
            team: Team name
            week: Week number

        Returns:
            Rank (1-25) or None if unranked/unknown
        """
        week_rankings = self._rankings_by_week.get(week, {})
        return week_rankings.get(team)

    def get_week_rankings(self, week: int) -> dict[str, int]:
        """Get all rankings for a specific week.

        Args:
            week: Week number

        Returns:
            Dict of team -> rank (empty if no data)
        """
        return self._rankings_by_week.get(week, {})

    def has_week(self, week: int) -> bool:
        """Check if rankings exist for a week."""
        return week in self._rankings_by_week

    @property
    def weeks_loaded(self) -> list[int]:
        """List of weeks with rankings data."""
        return sorted(self._rankings_by_week.keys())


@dataclass
class SituationalFactors:
    """Container for situational adjustment factors."""

    team: str
    rest_advantage: float = 0.0  # Days of rest differential (replaces binary bye_week)
    rest_days: int = 7  # Actual days of rest for this team
    letdown_penalty: float = 0.0
    lookahead_penalty: float = 0.0
    sandwich_penalty: float = 0.0  # Extra penalty when BOTH letdown AND lookahead apply
    rivalry_boost: float = 0.0
    total_adjustment: float = 0.0

    def __post_init__(self):
        self.total_adjustment = (
            self.rest_advantage
            + self.letdown_penalty
            + self.lookahead_penalty
            + self.sandwich_penalty
            + self.rivalry_boost
        )

    # Backward compatibility alias
    @property
    def bye_week_advantage(self) -> float:
        return self.rest_advantage


class SituationalAdjuster:
    """
    Calculate situational adjustments for matchups.

    Factors:
    - Rest advantage: Days of rest differential (bye week, short week, mini-bye)
    - Letdown spot: Team coming off big win vs top-15 now facing unranked
    - Look-ahead spot: Team with rival/top-10 opponent next week
    - Sandwich spot: BOTH letdown AND lookahead (compounding penalty)
    - Rivalry: Underdog gets small boost in rivalry games

    Sandwich Spot:
    The most dangerous scheduling spot in CFB is the "sandwich" - when a team
    is coming off a massive emotional win (letdown) AND has a massive game
    on deck next week (lookahead). The unranked team in the middle is the
    "meat" of the sandwich. When both flags trigger, an extra compounding
    penalty is applied beyond the sum of individual penalties.

    Rest Day Categories:
    - Bye week: 14+ days (team didn't play previous week)
    - Mini-bye: 9-13 days (Thursday game → following Saturday)
    - Normal: 6-8 days (Saturday → Saturday)
    - Short week: 4-5 days (Saturday → Thursday)
    """

    # Rest day thresholds
    NORMAL_REST = 7  # Saturday → Saturday baseline
    SHORT_WEEK_THRESHOLD = 5  # Saturday → Thursday
    MINI_BYE_THRESHOLD = 9  # Thursday → Saturday
    BYE_WEEK_THRESHOLD = 14  # Full week off

    def __init__(
        self,
        bye_advantage: Optional[float] = None,
        short_week_penalty: Optional[float] = None,
        letdown_penalty: Optional[float] = None,
        letdown_away_multiplier: Optional[float] = None,
        lookahead_penalty: Optional[float] = None,
        sandwich_extra_penalty: Optional[float] = None,
        rivalry_boost: Optional[float] = None,
    ):
        """Initialize situational adjuster.

        Args:
            bye_advantage: Points for full bye week (14+ days rest)
            short_week_penalty: Points penalty per day below normal rest
            letdown_penalty: Points penalty for letdown spot
            letdown_away_multiplier: Multiplier for letdown when team is away (sleepy road game)
            lookahead_penalty: Points penalty for look-ahead spot
            sandwich_extra_penalty: Extra penalty when BOTH letdown AND lookahead apply
            rivalry_boost: Points boost for underdog in rivalry
        """
        settings = get_settings()

        # Rest advantages/penalties
        self.bye_advantage = (
            bye_advantage if bye_advantage is not None else settings.bye_week_advantage
        )
        # Default: ~0.5 pts per day of rest differential
        self.rest_points_per_day = 0.5

        self.letdown_penalty = (
            letdown_penalty
            if letdown_penalty is not None
            else settings.letdown_penalty
        )
        self.letdown_away_multiplier = (
            letdown_away_multiplier
            if letdown_away_multiplier is not None
            else settings.letdown_away_multiplier
        )
        self.lookahead_penalty = (
            lookahead_penalty
            if lookahead_penalty is not None
            else settings.lookahead_penalty
        )
        self.sandwich_extra_penalty = (
            sandwich_extra_penalty
            if sandwich_extra_penalty is not None
            else settings.sandwich_extra_penalty
        )
        self.rivalry_boost = (
            rivalry_boost
            if rivalry_boost is not None
            else settings.rivalry_underdog_boost
        )

    def calculate_rest_days(
        self,
        team: str,
        game_date: datetime,
        schedule_df: pd.DataFrame,
    ) -> int:
        """Calculate days of rest since team's last game.

        CFB Context:
        - Normal rest: 7 days (Saturday → Saturday)
        - Mini-bye: 9+ days (Thursday → following Saturday)
        - Short week: 5 days (Saturday → Thursday)
        - Bye week: 14+ days (didn't play previous week)

        Args:
            team: Team name
            game_date: Date of current game (datetime)
            schedule_df: DataFrame with schedule (must have start_date, home_team, away_team)

        Returns:
            Days since last game (default 7 if no previous game found)
        """
        # Find team's previous games
        team_games = schedule_df[
            (schedule_df["home_team"] == team) | (schedule_df["away_team"] == team)
        ].copy()

        if team_games.empty:
            return self.NORMAL_REST

        # Parse dates if needed
        if "start_date" not in team_games.columns:
            return self.NORMAL_REST

        # Convert game_date to pandas Timestamp for comparison
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
        elif not isinstance(game_date, pd.Timestamp):
            game_date = pd.Timestamp(game_date)

        # Make game_date timezone-naive for comparison
        if game_date.tzinfo is not None:
            game_date = game_date.tz_localize(None)

        # Convert start_date column to datetime
        team_games["game_datetime"] = pd.to_datetime(team_games["start_date"], utc=True)
        # Make timezone-naive
        team_games["game_datetime"] = team_games["game_datetime"].dt.tz_localize(None)

        # Find most recent game BEFORE current game
        previous_games = team_games[team_games["game_datetime"] < game_date]

        if previous_games.empty:
            return self.NORMAL_REST

        last_game_date = previous_games["game_datetime"].max()
        days_rest = (game_date - last_game_date).days

        return max(1, days_rest)  # Minimum 1 day

    def check_bye_week(
        self,
        team: str,
        current_week: int,
        schedule_df: pd.DataFrame,
    ) -> bool:
        """Check if team is coming off a bye week (DEPRECATED - use calculate_rest_days).

        Maintained for backward compatibility. Use calculate_rest_days for more
        accurate rest differential calculations.

        Args:
            team: Team name
            current_week: Current week number
            schedule_df: DataFrame with schedule (must have week, home_team, away_team)

        Returns:
            True if team had bye last week
        """
        if current_week <= 1:
            return False

        # Check if team played last week
        last_week_games = schedule_df[schedule_df["week"] == current_week - 1]
        team_played = (
            (last_week_games["home_team"] == team)
            | (last_week_games["away_team"] == team)
        ).any()

        return not team_played

    def calculate_rest_advantage(
        self,
        home_rest_days: int,
        away_rest_days: int,
    ) -> float:
        """Calculate rest advantage adjustment.

        Args:
            home_rest_days: Days of rest for home team
            away_rest_days: Days of rest for away team

        Returns:
            Points adjustment (positive = home team advantage)
        """
        rest_diff = home_rest_days - away_rest_days

        # Categories of advantage:
        # - Bye vs Normal (7 days diff): ~1.5 pts (full bye advantage)
        # - Mini-bye vs Normal (2-3 days diff): ~1.0 pts
        # - Normal vs Short week (2 days diff): ~1.0 pts
        # - Short week vs Normal (-2 days): ~-1.0 pts

        if rest_diff == 0:
            return 0.0

        # Scale: ~0.5 pts per day of rest differential, capped
        adjustment = rest_diff * self.rest_points_per_day

        # Cap at bye week advantage (typically 1.5 pts)
        max_advantage = abs(self.bye_advantage)
        adjustment = max(-max_advantage, min(max_advantage, adjustment))

        return adjustment

    def check_letdown_spot(
        self,
        team: str,
        current_week: int,
        opponent: str,
        schedule_df: pd.DataFrame,
        rankings: Optional[dict[str, int]] = None,
        historical_rankings: Optional[HistoricalRankings] = None,
    ) -> bool:
        """Check if team is in a letdown spot.

        Letdown: Coming off a "big win" now facing unranked opponent.

        Big Win criteria (either triggers letdown):
        1. Beat a top-15 ranked team, OR
        2. Beat an arch-rival (regardless of rival's ranking)

        Rivalry Hangover: Beating your arch-rival is an emotional peak even if
        they're unranked. Auburn beating Alabama, Ohio State beating Michigan,
        etc. creates the same letdown risk as beating a top-15 team.

        CRITICAL: Uses historical rankings (rank at time of game) to evaluate
        whether the previous opponent was ranked. CFB rankings are volatile -
        a team ranked #15 in Week 3 may be unranked by Week 8.

        Args:
            team: Team name
            current_week: Current week number
            opponent: Current opponent
            schedule_df: DataFrame with schedule and results
            rankings: Dict of team -> ranking for CURRENT week (used for opponent check)
            historical_rankings: HistoricalRankings object for week-by-week lookup

        Returns:
            True if team is in letdown spot
        """
        if current_week <= 1:
            return False

        # Check last week's game
        last_week_num = current_week - 1
        last_week = schedule_df[schedule_df["week"] == last_week_num]
        team_games = last_week[
            (last_week["home_team"] == team) | (last_week["away_team"] == team)
        ]

        if team_games.empty:
            return False  # Had bye week

        last_game = team_games.iloc[0]

        # Determine if they won
        if last_game["home_team"] == team:
            won = last_game.get("home_points", 0) > last_game.get("away_points", 0)
            last_opponent = last_game["away_team"]
        else:
            won = last_game.get("away_points", 0) > last_game.get("home_points", 0)
            last_opponent = last_game["home_team"]

        if not won:
            return False

        # Check for "Big Win" - either ranked opponent OR rivalry win
        was_big_win = False
        big_win_reason = None

        # Criterion 1: Beat a rival (emotional peak regardless of ranking)
        if is_rivalry_game(team, last_opponent):
            was_big_win = True
            big_win_reason = f"rivalry win vs {last_opponent}"

        # Criterion 2: Beat a top-15 team (use historical rankings)
        if not was_big_win:
            if historical_rankings is not None and historical_rankings.has_week(last_week_num):
                last_opp_rank = historical_rankings.get_rank(last_opponent, last_week_num)
            elif rankings is not None:
                last_opp_rank = rankings.get(last_opponent)
                logger.debug(
                    f"No historical rankings for week {last_week_num}, using current rankings"
                )
            else:
                last_opp_rank = None

            if last_opp_rank is not None and last_opp_rank <= 15:
                was_big_win = True
                big_win_reason = f"beat #{last_opp_rank} {last_opponent}"

        if not was_big_win:
            return False

        # Check if current opponent is unranked (use current week rankings)
        if historical_rankings is not None and historical_rankings.has_week(current_week):
            current_opp_rank = historical_rankings.get_rank(opponent, current_week)
        elif rankings is not None:
            current_opp_rank = rankings.get(opponent)
        else:
            current_opp_rank = None

        is_letdown = current_opp_rank is None
        if is_letdown:
            logger.debug(
                f"Letdown spot detected: {team} {big_win_reason} "
                f"last week, now facing unranked {opponent}"
            )
        return is_letdown

    def check_lookahead_spot(
        self,
        team: str,
        current_week: int,
        schedule_df: pd.DataFrame,
        rankings: Optional[dict[str, int]] = None,
        historical_rankings: Optional[HistoricalRankings] = None,
    ) -> bool:
        """Check if team is in a look-ahead spot.

        Look-ahead: Next week is vs rival or top-10 team.

        Note: For lookahead, we use CURRENT rankings since we're predicting
        whether a team might overlook their current opponent while focusing
        on next week's marquee matchup. The perception NOW matters.

        Args:
            team: Team name
            current_week: Current week number
            schedule_df: DataFrame with schedule
            rankings: Dict of team -> ranking (current week)
            historical_rankings: HistoricalRankings object (optional)

        Returns:
            True if team is in look-ahead spot
        """
        # Check next week's game
        next_week_num = current_week + 1
        next_week = schedule_df[schedule_df["week"] == next_week_num]
        team_games = next_week[
            (next_week["home_team"] == team) | (next_week["away_team"] == team)
        ]

        if team_games.empty:
            return False

        next_game = team_games.iloc[0]

        # Get next opponent
        if next_game["home_team"] == team:
            next_opponent = next_game["away_team"]
        else:
            next_opponent = next_game["home_team"]

        # Check if it's a rivalry game
        if is_rivalry_game(team, next_opponent):
            logger.debug(f"Lookahead spot: {team} has rivalry vs {next_opponent} next week")
            return True

        # Check if next opponent is top-10 (use current rankings - perception matters)
        next_opp_rank = None
        if historical_rankings is not None and historical_rankings.has_week(current_week):
            next_opp_rank = historical_rankings.get_rank(next_opponent, current_week)
        elif rankings:
            next_opp_rank = rankings.get(next_opponent)

        if next_opp_rank is not None and next_opp_rank <= 10:
            logger.debug(
                f"Lookahead spot: {team} has #{next_opp_rank} {next_opponent} next week"
            )
            return True

        return False

    def check_rivalry(
        self,
        team_a: str,
        team_b: str,
    ) -> bool:
        """Check if two teams are rivals.

        Args:
            team_a: First team
            team_b: Second team

        Returns:
            True if teams are rivals
        """
        return is_rivalry_game(team_a, team_b)

    def calculate_factors(
        self,
        team: str,
        opponent: str,
        current_week: int,
        schedule_df: pd.DataFrame,
        rankings: Optional[dict[str, int]] = None,
        team_is_favorite: bool = True,
        team_is_home: bool = True,
        historical_rankings: Optional[HistoricalRankings] = None,
        game_date: Optional[datetime] = None,
    ) -> SituationalFactors:
        """Calculate all situational factors for a team in a matchup.

        Args:
            team: Team to calculate factors for
            opponent: Opponent team
            current_week: Current week number
            schedule_df: DataFrame with schedule and results
            rankings: Dict of team -> ranking (current week snapshot)
            team_is_favorite: Whether team is favored (for rivalry boost)
            team_is_home: Whether team is playing at home (affects letdown severity)
            historical_rankings: HistoricalRankings for week-by-week lookup
            game_date: Date of current game (for rest calculation)

        Returns:
            SituationalFactors for the team
        """
        factors = SituationalFactors(team=team)

        # Calculate rest days (new approach - replaces binary bye week)
        if game_date is not None and "start_date" in schedule_df.columns:
            factors.rest_days = self.calculate_rest_days(team, game_date, schedule_df)
            logger.debug(f"{team} rest days: {factors.rest_days}")
        else:
            # Fallback to binary bye week check if no date available
            if self.check_bye_week(team, current_week, schedule_df):
                factors.rest_days = 14  # Assume full bye
                logger.debug(f"{team} coming off bye (fallback): rest_days=14")
            else:
                factors.rest_days = 7  # Assume normal week

        # Letdown spot (uses historical rankings for previous opponent)
        in_letdown = self.check_letdown_spot(
            team, current_week, opponent, schedule_df, rankings, historical_rankings
        )
        if in_letdown:
            # "Sleepy road game" - letdown is worse when traveling
            # Playing at home keeps the team engaged; noon kickoff on the road = danger
            if team_is_home:
                factors.letdown_penalty = self.letdown_penalty
                logger.debug(f"{team} in letdown spot (home): {self.letdown_penalty}")
            else:
                factors.letdown_penalty = self.letdown_penalty * self.letdown_away_multiplier
                logger.debug(
                    f"{team} in letdown spot (AWAY - sleepy road game): "
                    f"{factors.letdown_penalty:.1f} ({self.letdown_penalty} × {self.letdown_away_multiplier})"
                )

        # Look-ahead spot (uses current rankings - perception matters)
        in_lookahead = self.check_lookahead_spot(
            team, current_week, schedule_df, rankings, historical_rankings
        )
        if in_lookahead:
            factors.lookahead_penalty = self.lookahead_penalty
            logger.debug(f"{team} in look-ahead spot: {self.lookahead_penalty}")

        # SANDWICH SPOT: Team is in BOTH letdown AND lookahead - the most dangerous spot
        # Coming off a big emotional win AND looking ahead to a big game next week
        # The unranked opponent in the middle is the "meat" of the sandwich
        if in_letdown and in_lookahead:
            factors.sandwich_penalty = self.sandwich_extra_penalty
            logger.info(
                f"SANDWICH SPOT: {team} in letdown AND lookahead - "
                f"extra penalty: {self.sandwich_extra_penalty}"
            )

        # Rivalry boost (for underdog only)
        if self.check_rivalry(team, opponent) and not team_is_favorite:
            factors.rivalry_boost = self.rivalry_boost
            logger.debug(f"{team} rivalry underdog boost: +{self.rivalry_boost}")

        # Note: rest_advantage is calculated in get_matchup_adjustment based on differential
        # Total will be recalculated there
        factors.total_adjustment = (
            factors.rest_advantage  # Will be 0 here, set in get_matchup_adjustment
            + factors.letdown_penalty
            + factors.lookahead_penalty
            + factors.sandwich_penalty
            + factors.rivalry_boost
        )

        return factors

    def get_matchup_adjustment(
        self,
        home_team: str,
        away_team: str,
        current_week: int,
        schedule_df: pd.DataFrame,
        rankings: Optional[dict[str, int]] = None,
        home_is_favorite: bool = True,
        historical_rankings: Optional[HistoricalRankings] = None,
        game_date: Optional[datetime] = None,
    ) -> tuple[float, dict]:
        """Get net situational adjustment for a matchup.

        Args:
            home_team: Home team name
            away_team: Away team name
            current_week: Current week number
            schedule_df: Schedule DataFrame
            rankings: Team rankings (current week snapshot)
            home_is_favorite: Whether home team is favored
            historical_rankings: HistoricalRankings for week-by-week lookup
            game_date: Date of current game (for rest day calculation)

        Returns:
            Tuple of (net adjustment favoring home, breakdown dict)
        """
        home_factors = self.calculate_factors(
            team=home_team,
            opponent=away_team,
            current_week=current_week,
            schedule_df=schedule_df,
            rankings=rankings,
            team_is_favorite=home_is_favorite,
            team_is_home=True,
            historical_rankings=historical_rankings,
            game_date=game_date,
        )

        away_factors = self.calculate_factors(
            team=away_team,
            opponent=home_team,
            current_week=current_week,
            schedule_df=schedule_df,
            rankings=rankings,
            team_is_favorite=not home_is_favorite,
            team_is_home=False,  # Away team - sleepy road game multiplier applies
            historical_rankings=historical_rankings,
            game_date=game_date,
        )

        # Calculate rest differential (replaces binary bye week)
        rest_advantage = self.calculate_rest_advantage(
            home_rest_days=home_factors.rest_days,
            away_rest_days=away_factors.rest_days,
        )

        # Update factors with rest advantage (assigned to home team by convention)
        home_factors.rest_advantage = rest_advantage

        # Recalculate totals with rest advantage included
        home_factors.total_adjustment = (
            home_factors.rest_advantage
            + home_factors.letdown_penalty
            + home_factors.lookahead_penalty
            + home_factors.sandwich_penalty
            + home_factors.rivalry_boost
        )
        away_factors.total_adjustment = (
            away_factors.rest_advantage  # 0 - rest differential is on home side
            + away_factors.letdown_penalty
            + away_factors.lookahead_penalty
            + away_factors.sandwich_penalty
            + away_factors.rivalry_boost
        )

        net_adjustment = home_factors.total_adjustment - away_factors.total_adjustment

        breakdown = {
            # Rest days and differential (replaces bye week)
            "home_rest_days": home_factors.rest_days,
            "away_rest_days": away_factors.rest_days,
            "rest_advantage": rest_advantage,
            # Legacy fields for backward compatibility
            "home_bye": home_factors.bye_week_advantage,  # Now reflects rest_advantage
            "away_bye": away_factors.bye_week_advantage,  # Always 0 (differential on home)
            # Other factors
            "home_letdown": home_factors.letdown_penalty,
            "away_letdown": away_factors.letdown_penalty,
            "home_lookahead": home_factors.lookahead_penalty,
            "away_lookahead": away_factors.lookahead_penalty,
            "home_sandwich": home_factors.sandwich_penalty,
            "away_sandwich": away_factors.sandwich_penalty,
            "home_rivalry": home_factors.rivalry_boost,
            "away_rivalry": away_factors.rivalry_boost,
            "net": net_adjustment,
        }

        return net_adjustment, breakdown
