"""Preseason feature engineering for win total projections.

Builds ~17 features per FBS team using CFBD API data that is available
before the season starts.

Leakage audit status per feature:
- SAFE: Timing independently verified (e.g., SP+ published post-season)
- ASSUMED: Timing assumed preseason-safe but not independently verified
  against CFBD publication dates (e.g., returning production)
- LEAKAGE_RISK: Known or suspected in-season data contamination
- EXCLUDED: Deliberately excluded from V1 (e.g., portal features)

Portal features are EXCLUDED in V1. CFBD's transfer portal endpoint does
NOT provide reliable transfer dates for filtering to pre-August transfers.
Without date filtering, portal data may include transfers that occur after
preseason projections would be published, creating leakage risk.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.teams import normalize_team_name
from src.api.cfbd_client import CFBDClient
from src.data.priors_cache import PriorsDataCache

logger = logging.getLogger(__name__)

# Leakage audit statuses
SAFE = 'SAFE'           # Timing verified
ASSUMED = 'ASSUMED'      # Timing assumed safe, not independently verified
LEAKAGE_RISK = 'LEAKAGE_RISK'  # Known or suspected leakage
EXCLUDED = 'EXCLUDED'   # Deliberately excluded from model

# Feature metadata for leakage auditing.
# Status values: SAFE, ASSUMED, LEAKAGE_RISK, EXCLUDED.
FEATURE_METADATA = {
    # Prior SP+ — Published by ESPN/Connelly post-season (~January). SAFE.
    'prior_sp_overall': {
        'description': 'End-of-prior-season SP+ overall rating',
        'timing': 'Post prior season (published ~Jan by ESPN)',
        'status': SAFE,
        'leakage_risk': False,
    },
    'prior_sp_offense': {
        'description': 'End-of-prior-season SP+ offensive rating',
        'timing': 'Post prior season (published ~Jan by ESPN)',
        'status': SAFE,
        'leakage_risk': False,
    },
    'prior_sp_defense': {
        'description': 'End-of-prior-season SP+ defensive rating',
        'timing': 'Post prior season (published ~Jan by ESPN)',
        'status': SAFE,
        'leakage_risk': False,
    },
    'prior_sp_2yr_avg': {
        'description': '2-year average of SP+ overall rating',
        'timing': 'Derived from prior SP+ (SAFE inputs)',
        'status': SAFE,
        'leakage_risk': False,
    },
    'prior_sp_st': {
        'description': 'End-of-prior-season SP+ special teams rating',
        'timing': 'Post prior season (published ~Jan by ESPN)',
        'status': SAFE,
        'leakage_risk': False,
    },
    # Returning production — CFBD publication date not independently verified.
    # Data represents roster composition which can change through August
    # (late transfers, medical retirements). ASSUMED safe as CFBD typically
    # publishes a snapshot, but exact snapshot date is unverified.
    'ret_ppa_total': {
        'description': 'Returning production: total percent PPA returning',
        'timing': 'Preseason — CFBD snapshot date unverified',
        'status': ASSUMED,
        'leakage_risk': False,
    },
    'ret_ppa_offense': {
        'description': 'Returning production: offensive percent PPA',
        'timing': 'Preseason — CFBD snapshot date unverified',
        'status': ASSUMED,
        'leakage_risk': False,
    },
    'ret_ppa_defense': {
        'description': 'Returning production: defensive percent PPA',
        'timing': 'Preseason — CFBD snapshot date unverified',
        'status': ASSUMED,
        'leakage_risk': False,
    },
    # Talent composite — Based on recruiting rankings (signing days in Dec/Feb).
    # SAFE: recruiting class data is finalized well before the season.
    'talent_composite': {
        'description': 'Team talent composite score',
        'timing': 'Preseason (NSD Feb, data finalized by spring)',
        'status': SAFE,
        'leakage_risk': False,
    },
    'talent_3yr_avg': {
        'description': '3-year rolling average talent composite',
        'timing': 'Derived from talent composite (SAFE inputs)',
        'status': SAFE,
        'leakage_risk': False,
    },
    'talent_trend': {
        'description': 'Year-over-year change in talent composite',
        'timing': 'Derived from talent composite (SAFE inputs)',
        'status': SAFE,
        'leakage_risk': False,
    },
    # Coaching — Coaching changes happen Dec-Jan, finalized before spring.
    # ASSUMED because CFBD's coaching endpoint timing is not independently
    # verified for mid-season firing edge cases (rare but possible).
    'coaching_years': {
        'description': 'Head coach tenure at current school (years)',
        'timing': 'Preseason — coaching changes Dec-Jan, CFBD timing assumed',
        'status': ASSUMED,
        'leakage_risk': False,
    },
    'coaching_was_imputed': {
        'description': 'Binary: 1.0 if coaching data was missing and median-imputed',
        'timing': 'Derived from coaching_years (ASSUMED input)',
        'status': ASSUMED,
        'leakage_risk': False,
    },
    'is_new_coach': {
        'description': 'Binary flag: new head coach this season',
        'timing': 'Derived from coaching_years (ASSUMED input)',
        'status': ASSUMED,
        'leakage_risk': False,
    },
    # Conference strength — Derived from prior SP+ (SAFE) + conference
    # membership. Conference realignment happens well before season.
    'conf_strength': {
        'description': 'Average prior SP+ of conference members',
        'timing': 'Derived from prior SP+ (SAFE) + preseason conference lists',
        'status': SAFE,
        'leakage_risk': False,
    },
    # Prior record — Game results from completed prior season.
    'prior_win_pct': {
        'description': 'Prior season win percentage (regular season only)',
        'timing': 'Post prior season (completed games)',
        'status': SAFE,
        'leakage_risk': False,
    },
    'prior_sos': {
        'description': 'Prior season strength of schedule (avg opp prior SP+)',
        'timing': 'Post prior season (derived from SAFE inputs)',
        'status': SAFE,
        'leakage_risk': False,
    },
}

# EXCLUDED features — documented for transparency
EXCLUDED_FEATURES = {
    'portal_impact': {
        'description': 'Transfer portal net talent change',
        'reason': 'CFBD portal endpoint lacks reliable transfer dates. '
                  'Cannot filter to pre-August transfers without date field. '
                  'Risk of including transfers that occur after preseason projections.',
        'status': EXCLUDED,
        'reactivation_condition': 'CFBD adds transfer_date field with reliable dates',
    },
    'portal_ppa_gained': {
        'description': 'PPA gained from incoming transfers',
        'reason': 'Same as portal_impact — no date filtering available',
        'status': EXCLUDED,
        'reactivation_condition': 'CFBD adds transfer_date field with reliable dates',
    },
    'portal_ppa_lost': {
        'description': 'PPA lost to outgoing transfers',
        'reason': 'Same as portal_impact — no date filtering available',
        'status': EXCLUDED,
        'reactivation_condition': 'CFBD adds transfer_date field with reliable dates',
    },
}


def feature_status_counts() -> dict[str, int]:
    """Count features by leakage audit status."""
    counts: dict[str, int] = {}
    for meta in FEATURE_METADATA.values():
        status = meta['status']
        counts[status] = counts.get(status, 0) + 1
    return counts


class PreseasonFeatureBuilder:
    """Builds preseason features from CFBD API data."""

    def __init__(
        self,
        client: CFBDClient | None = None,
        cache: PriorsDataCache | None = None,
    ):
        self.client = client or CFBDClient()
        self.cache = cache or PriorsDataCache()

    def _get_sp_ratings(self, year: int) -> dict[str, dict]:
        """Get SP+ ratings dict {team: {overall, offense, defense, st}}."""
        raw = self.client.get_sp_ratings(year)
        result = {}
        for r in raw:
            team = normalize_team_name(r.team)
            result[team] = {
                'overall': getattr(r, 'rating', None) or 0.0,
                'offense': getattr(r, 'offense', None) and getattr(r.offense, 'rating', 0.0) or 0.0,
                'defense': getattr(r, 'defense', None) and getattr(r.defense, 'rating', 0.0) or 0.0,
                'st': getattr(r, 'special_teams', None) and getattr(r.special_teams, 'rating', 0.0) or 0.0,
            }
        return result

    def _get_talent(self, year: int) -> dict[str, float]:
        """Get talent composite {team: score}."""
        cached = self.cache.load_talent(year)
        if cached is not None:
            return {normalize_team_name(t): v for t, v in cached.items()}

        raw = self.client.get_team_talent(year)
        result = {}
        for r in raw:
            team = normalize_team_name(r.school)
            result[team] = float(r.talent) if r.talent else 0.0

        self.cache.save_talent(year, result)
        return result

    def _get_returning_production(self, year: int) -> dict[str, dict]:
        """Get returning production {team: {total, offense, defense}}."""
        raw = self.client.get_returning_production(year)
        result = {}
        for r in raw:
            team = normalize_team_name(r.team)
            result[team] = {
                'total': float(r.percent_ppa) if r.percent_ppa else 0.5,
                'offense': float(r.percent_ppa_offense) if hasattr(r, 'percent_ppa_offense') and r.percent_ppa_offense else 0.5,
                'defense': float(r.percent_ppa_defense) if hasattr(r, 'percent_ppa_defense') and r.percent_ppa_defense else 0.5,
            }
        return result

    def _get_coaching(self, year: int) -> dict[str, dict]:
        """Get coaching info {team: {years: int, is_new: bool}}.

        Counts consecutive years at the same school by iterating backwards
        from the query year and breaking at the first gap.
        """
        raw = self.client.get_coaches(year=year)
        result = {}
        for coach in raw:
            for season in getattr(coach, 'seasons', []):
                if getattr(season, 'year', None) == year:
                    team = normalize_team_name(season.school)
                    # Build set of years this coach was at this school
                    years_at_school = set()
                    for s in getattr(coach, 'seasons', []):
                        if getattr(s, 'school', '') == season.school:
                            years_at_school.add(getattr(s, 'year', 0))
                    # Count backwards from query year
                    tenure = 0
                    for y in range(year, year - 50, -1):
                        if y in years_at_school:
                            tenure += 1
                        else:
                            break
                    result[team] = {
                        'years': tenure,
                        'is_new': tenure <= 1,
                    }
        return result

    def _get_prior_record_from_games(self, games: list) -> dict[str, dict]:
        """Get prior season win-loss records from pre-fetched games.

        On-field result convention:
        - Win: home_points > away_points OR away_points > home_points
        - Regular season only: week <= 15
        - CCG and bowls are EXCLUDED
        """
        records: dict[str, dict] = {}

        for g in games:
            if g.home_points is None or g.away_points is None:
                continue
            # Safeguard: exclude CCG (week > 15) even if API returns them
            week = getattr(g, 'week', None)
            if week is not None and week > 15:
                continue

            home = normalize_team_name(g.home_team)
            away = normalize_team_name(g.away_team)

            for team in [home, away]:
                if team not in records:
                    records[team] = {'wins': 0, 'losses': 0}

            if g.home_points > g.away_points:
                records[home]['wins'] += 1
                records[away]['losses'] += 1
            elif g.away_points > g.home_points:
                records[away]['wins'] += 1
                records[home]['losses'] += 1

        for team in records:
            total = records[team]['wins'] + records[team]['losses']
            records[team]['win_pct'] = records[team]['wins'] / total if total > 0 else 0.5

        return records

    def _get_fbs_teams(self, year: int) -> set[str]:
        """Get set of FBS teams for a given year."""
        raw = self.client.get_fbs_teams(year=year)
        return {normalize_team_name(t.school) for t in raw}

    def build_features(self, year: int) -> pd.DataFrame:
        """Build preseason features for all FBS teams in a given year.

        Args:
            year: Season year to build features for

        Returns:
            DataFrame with one row per team and ~17 feature columns
        """
        logger.info(f"Building preseason features for {year}")

        fbs_teams = self._get_fbs_teams(year)
        sp_prior = self._get_sp_ratings(year - 1)
        sp_2yr_ago = self._get_sp_ratings(year - 2) if year >= 2016 else {}
        talent_cur = self._get_talent(year)
        talent_1yr = self._get_talent(year - 1) if year >= 2016 else {}
        talent_2yr = self._get_talent(year - 2) if year >= 2017 else {}
        ret_prod = self._get_returning_production(year)
        coaching = self._get_coaching(year)
        prior_games = self.client.get_games(year=year - 1, season_type='regular')
        prior_record = self._get_prior_record_from_games(prior_games)

        fbs_raw = self.client.get_fbs_teams(year=year)
        team_conf = {}
        for t in fbs_raw:
            team_conf[normalize_team_name(t.school)] = getattr(t, 'conference', 'Independent')

        conf_sp: dict[str, list[float]] = {}
        for team, conf in team_conf.items():
            if conf and team in sp_prior:
                conf_sp.setdefault(conf, []).append(sp_prior[team]['overall'])
        conf_avg = {c: np.mean(vals) for c, vals in conf_sp.items()}

        prior_sos = self._compute_prior_sos_from_games(prior_games, sp_prior)

        coaching_years_list = [v['years'] for v in coaching.values()]
        median_tenure = int(np.median(coaching_years_list)) if coaching_years_list else 3

        rows = []
        for team in sorted(fbs_teams):
            sp = sp_prior.get(team, {})
            sp2 = sp_2yr_ago.get(team, {})
            tc = talent_cur.get(team, 0.0)
            t1 = talent_1yr.get(team, tc)
            t2 = talent_2yr.get(team, t1)
            rp = ret_prod.get(team, {'total': 0.5, 'offense': 0.5, 'defense': 0.5})
            coaching_was_imputed = team not in coaching
            coach = coaching.get(team, {'years': median_tenure, 'is_new': False})
            rec = prior_record.get(team, {'win_pct': 0.5})
            conf = team_conf.get(team, 'Independent')

            prior_overall = sp.get('overall', 0.0)
            prior_2yr = sp2.get('overall', prior_overall)

            row = {
                'team': team,
                'year': year,
                'prior_sp_overall': prior_overall,
                'prior_sp_offense': sp.get('offense', 0.0),
                'prior_sp_defense': sp.get('defense', 0.0),
                'prior_sp_st': sp.get('st', 0.0),
                'prior_sp_2yr_avg': (prior_overall + prior_2yr) / 2.0,
                'ret_ppa_total': rp['total'],
                'ret_ppa_offense': rp['offense'],
                'ret_ppa_defense': rp['defense'],
                'talent_composite': tc,
                'talent_3yr_avg': (tc + t1 + t2) / 3.0,
                'talent_trend': tc - t1,
                'coaching_years': coach['years'],
                'coaching_was_imputed': float(coaching_was_imputed),
                'is_new_coach': float(coach['is_new']),
                'conf_strength': conf_avg.get(conf, 0.0),
                'prior_win_pct': rec['win_pct'],
                'prior_sos': prior_sos.get(team, 0.0),
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Log feature status summary
        counts = feature_status_counts()
        logger.info(f"Built {len(df)} team feature rows for {year} "
                     f"({len([c for c in df.columns if c not in ['team', 'year']])} features). "
                     f"Status: {counts}")

        return df

    def _compute_prior_sos_from_games(
        self, games: list, sp_prior: dict[str, dict]
    ) -> dict[str, float]:
        """Compute prior season SOS as average opponent SP+ rating.

        Regular season only (week <= 15 safeguard).
        """
        opp_ratings: dict[str, list[float]] = {}

        for g in games:
            if g.home_points is None:
                continue
            week = getattr(g, 'week', None)
            if week is not None and week > 15:
                continue

            home = normalize_team_name(g.home_team)
            away = normalize_team_name(g.away_team)

            home_sp = sp_prior.get(home, {}).get('overall', 0.0)
            away_sp = sp_prior.get(away, {}).get('overall', 0.0)

            opp_ratings.setdefault(home, []).append(away_sp)
            opp_ratings.setdefault(away, []).append(home_sp)

        return {team: float(np.mean(ratings)) for team, ratings in opp_ratings.items()}

    def build_training_dataset(
        self,
        start_year: int,
        end_year: int,
        exclude_years: set[int] | None = None,
    ) -> pd.DataFrame:
        """Build multi-year training dataset with target variable.

        Target: End-of-season SP+ overall rating for the prediction year.
        """
        exclude = exclude_years or set()
        frames = []

        for year in range(start_year, end_year + 1):
            if year in exclude:
                logger.info(f"Skipping {year} (excluded)")
                continue

            try:
                features = self.build_features(year)
                sp_current = self._get_sp_ratings(year)
                targets = {team: data['overall'] for team, data in sp_current.items()}
                features['target_sp'] = features['team'].map(targets)

                before = len(features)
                features = features.dropna(subset=['target_sp'])
                dropped = before - len(features)
                if dropped > 0:
                    logger.warning(f"{year}: Dropped {dropped} teams without SP+ target")

                frames.append(features)
                logger.info(f"{year}: {len(features)} teams with features + target")

            except Exception as e:
                logger.error(f"Failed to build features for {year}: {e}")
                continue

        if not frames:
            raise ValueError(f"No training data built for {start_year}-{end_year}")

        dataset = pd.concat(frames, ignore_index=True)
        logger.info(f"Training dataset: {len(dataset)} team-seasons, "
                     f"{start_year}-{end_year}")
        return dataset

    @staticmethod
    def feature_columns() -> list[str]:
        """Return ordered list of active feature column names (excludes EXCLUDED)."""
        return sorted(FEATURE_METADATA.keys())
