"""QB Injury Adjustment Module.

Calculates starter-to-backup drop-off for QB injuries and applies
adjustments to spread predictions when a starter is flagged as out.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cfbd

logger = logging.getLogger(__name__)


@dataclass
class QBDepthChart:
    """QB depth chart with PPA ratings."""

    team: str
    starter_name: str
    starter_ppa: float  # PPA per pass play
    backup_name: Optional[str]
    backup_ppa: Optional[float]
    ppa_drop: float  # Starter PPA - Backup PPA (positive = losing quality)
    point_adjustment: float  # Estimated point impact when starter is out

    def __str__(self) -> str:
        backup_str = f"{self.backup_name} ({self.backup_ppa:.3f})" if self.backup_name else "Unknown"
        return (
            f"{self.team}: {self.starter_name} ({self.starter_ppa:.3f}) -> "
            f"{backup_str}, adj: {self.point_adjustment:+.1f} pts"
        )


# Estimated pass plays per game for point conversion
PASS_PLAYS_PER_GAME = 30

# Default adjustment when we can't calculate (conservative)
DEFAULT_QB_OUT_ADJUSTMENT = -4.0


def calculate_qb_depth_chart(
    metrics_api: cfbd.MetricsApi,
    games_api: cfbd.GamesApi,
    team: str,
    year: int,
    through_week: Optional[int] = None,
) -> Optional[QBDepthChart]:
    """Calculate QB depth chart and drop-off for a team.

    Args:
        metrics_api: CFBD MetricsApi instance
        games_api: CFBD GamesApi instance
        team: Team name
        year: Season year
        through_week: Only use data through this week (for walk-forward)

    Returns:
        QBDepthChart with starter/backup info, or None if insufficient data
    """
    try:
        # Get QB PPA for the season
        ppa_data = metrics_api.get_predicted_points_added_by_player_season(
            year=year, team=team, position="QB"
        )

        if not ppa_data:
            logger.debug(f"No QB PPA data for {team}")
            return None

        # Extract PPA per pass for each QB
        qb_ratings = []
        for p in ppa_data:
            d = p.to_dict()
            name = d.get("name", "Unknown")
            avg_ppa = d.get("averagePPA", {})
            pass_ppa = avg_ppa.get("pass") if avg_ppa else None

            # Get total passing PPA to estimate volume
            total_ppa = d.get("totalPPA", {})
            total_pass_ppa = total_ppa.get("pass", 0) if total_ppa else 0

            if pass_ppa is not None and total_pass_ppa != 0:
                # Estimate attempts from total/avg
                est_attempts = abs(total_pass_ppa / pass_ppa) if pass_ppa != 0 else 0
                qb_ratings.append({
                    "name": name,
                    "pass_ppa": pass_ppa,
                    "est_attempts": est_attempts,
                })

        if len(qb_ratings) < 1:
            logger.debug(f"Insufficient QB data for {team}")
            return None

        # Sort by estimated attempts (starter has most)
        qb_ratings.sort(key=lambda x: x["est_attempts"], reverse=True)

        starter = qb_ratings[0]
        backup = qb_ratings[1] if len(qb_ratings) > 1 else None

        # Calculate drop-off
        if backup:
            ppa_drop = starter["pass_ppa"] - backup["pass_ppa"]
            backup_name = backup["name"]
            backup_ppa = backup["pass_ppa"]
        else:
            # No backup data - use conservative estimate
            # Assume backup is ~0.2 PPA worse (roughly league average backup drop)
            ppa_drop = 0.2
            backup_name = None
            backup_ppa = None

        # Convert PPA drop to point adjustment
        # PPA drop * pass plays per game = total PPA lost = ~points
        point_adjustment = -ppa_drop * PASS_PLAYS_PER_GAME

        return QBDepthChart(
            team=team,
            starter_name=starter["name"],
            starter_ppa=starter["pass_ppa"],
            backup_name=backup_name,
            backup_ppa=backup_ppa,
            ppa_drop=ppa_drop,
            point_adjustment=point_adjustment,
        )

    except Exception as e:
        logger.warning(f"Error calculating QB depth for {team}: {e}")
        return None


def build_qb_depth_charts(
    api_key: str,
    year: int,
    teams: Optional[list[str]] = None,
) -> dict[str, QBDepthChart]:
    """Build QB depth charts for multiple teams.

    Args:
        api_key: CFBD API key
        year: Season year
        teams: List of teams (if None, uses top 25 + common contenders)

    Returns:
        Dict mapping team name to QBDepthChart
    """
    config = cfbd.Configuration()
    config.access_token = api_key

    metrics_api = cfbd.MetricsApi(cfbd.ApiClient(config))
    games_api = cfbd.GamesApi(cfbd.ApiClient(config))

    # Default to major teams if none specified
    if teams is None:
        teams = [
            # Traditional powers
            "Ohio State", "Alabama", "Georgia", "Michigan", "Notre Dame",
            "Texas", "Oregon", "Penn State", "USC", "Oklahoma",
            "Clemson", "Florida State", "Miami", "LSU", "Tennessee",
            "Ole Miss", "Texas A&M", "Utah", "Washington", "Wisconsin",
            # 2024-25 contenders
            "Indiana", "Iowa State", "BYU", "Colorado", "Boise State",
            "Army", "Navy", "SMU", "Arizona State", "Louisville",
        ]

    depth_charts = {}

    for team in teams:
        chart = calculate_qb_depth_chart(metrics_api, games_api, team, year)
        if chart:
            depth_charts[team] = chart
            logger.info(f"QB depth: {chart}")

    return depth_charts


def get_qb_adjustment(
    team: str,
    depth_charts: dict[str, QBDepthChart],
) -> float:
    """Get point adjustment for a team with QB out.

    Args:
        team: Team name
        depth_charts: Pre-computed depth charts

    Returns:
        Point adjustment (negative = team is worse without starter)
    """
    if team in depth_charts:
        return depth_charts[team].point_adjustment

    # Team not in depth charts - use default
    logger.warning(f"No QB depth chart for {team}, using default {DEFAULT_QB_OUT_ADJUSTMENT}")
    return DEFAULT_QB_OUT_ADJUSTMENT


class QBInjuryAdjuster:
    """Manages QB injury adjustments for spread predictions."""

    def __init__(
        self,
        api_key: str,
        year: int,
        teams: Optional[list[str]] = None,
    ):
        """Initialize with pre-computed depth charts.

        Args:
            api_key: CFBD API key
            year: Season year
            teams: Teams to compute depth charts for
        """
        self.year = year
        self.depth_charts = build_qb_depth_charts(api_key, year, teams)
        self.qb_out_teams: set[str] = set()

    def flag_qb_out(self, team: str) -> None:
        """Flag a team's starting QB as out."""
        self.qb_out_teams.add(team)
        if team in self.depth_charts:
            chart = self.depth_charts[team]
            logger.info(
                f"QB OUT: {team} - {chart.starter_name} out, "
                f"adjustment: {chart.point_adjustment:+.1f} pts"
            )
        else:
            logger.info(f"QB OUT: {team} - using default adjustment {DEFAULT_QB_OUT_ADJUSTMENT}")

    def clear_qb_out(self, team: Optional[str] = None) -> None:
        """Clear QB out flag(s)."""
        if team:
            self.qb_out_teams.discard(team)
        else:
            self.qb_out_teams.clear()

    def get_adjustment(self, home_team: str, away_team: str) -> float:
        """Get net QB adjustment for a matchup.

        Args:
            home_team: Home team name
            away_team: Away team name

        Returns:
            Net point adjustment (positive = favors home)
        """
        adjustment = 0.0

        # Home team QB out hurts home
        if home_team in self.qb_out_teams:
            adjustment += get_qb_adjustment(home_team, self.depth_charts)

        # Away team QB out helps home
        if away_team in self.qb_out_teams:
            adjustment -= get_qb_adjustment(away_team, self.depth_charts)

        return adjustment

    def get_depth_chart(self, team: str) -> Optional[QBDepthChart]:
        """Get depth chart for a team."""
        return self.depth_charts.get(team)

    def print_depth_charts(self) -> None:
        """Print all depth charts.

        P3.9: Converted from print() to logger.info() for consistent logging.
        """
        logger.info("QB Depth Charts:")
        for team in sorted(self.depth_charts.keys()):
            chart = self.depth_charts[team]
            logger.info(f"  {chart}")
