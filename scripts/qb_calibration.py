#!/usr/bin/env python3
"""
QB Continuous Rating System Calibration Script.

Generates:
1. Distribution of qb_value_shrunk across team-weeks by week bucket
2. Percentiles (10/25/50/75/90) for calibrating QB_SCALE
3. Recommended QB_SCALE based on 90th percentile targeting
4. Frequency of cap binding by week bucket
5. Starter projection accuracy diagnostic

Usage:
    python scripts/qb_calibration.py --year 2024
    python scripts/qb_calibration.py --years 2022 2023 2024 --qb-scale 4.0
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.adjustments.qb_continuous import (
    QBContinuousAdjuster,
    DEFAULT_SHRINKAGE_K,
    DEFAULT_QB_CAP,
    DEFAULT_QB_SCALE,
    DEFAULT_PRIOR_DECAY,
)
from src.api.cfbd_client import CFBDClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_calibration(
    years: list[int],
    shrinkage_k: float = DEFAULT_SHRINKAGE_K,
    qb_cap: float = DEFAULT_QB_CAP,
    qb_scale: float = DEFAULT_QB_SCALE,
    prior_decay: float = DEFAULT_PRIOR_DECAY,
    use_prior_season: bool = True,
) -> pd.DataFrame:
    """Run QB calibration across specified years.

    Returns DataFrame with all computed QB qualities.
    """
    all_data = []

    for year in years:
        logger.info(f"Processing {year}...")

        adjuster = QBContinuousAdjuster(
            year=year,
            shrinkage_k=shrinkage_k,
            qb_cap=qb_cap,
            qb_scale=qb_scale,
            prior_decay=prior_decay,
            use_prior_season=use_prior_season,
        )

        # Get FBS teams for this year
        client = CFBDClient()
        teams = client.get_fbs_teams(year=year)
        team_names = [t.school for t in teams]
        logger.info(f"  Found {len(team_names)} FBS teams")

        # Build data and compute qualities for each week
        for pred_week in range(1, 16):
            try:
                adjuster.build_qb_data(through_week=pred_week - 1 if pred_week > 1 else 0)
            except Exception as e:
                logger.warning(f"Error building data for {year} week {pred_week}: {e}")
                continue

            # Compute qualities for all teams by calling get_adjustment
            for team in team_names:
                try:
                    # Call get_adjustment to populate _qualities cache
                    adjuster.get_adjustment(team, "Opponent", pred_week=pred_week)
                except Exception:
                    pass  # Some teams may not have data

        # Get calibration data
        cal_df = adjuster.get_calibration_data()
        if len(cal_df) > 0:
            all_data.append(cal_df)

        logger.info(f"  {len(cal_df)} team-week records")

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def analyze_calibration(df: pd.DataFrame, qb_cap: float, qb_scale: float) -> dict:
    """Analyze calibration data and generate recommendations."""
    if df.empty:
        return {}

    results = {}

    # Week buckets
    df['week_bucket'] = pd.cut(
        df['pred_week'],
        bins=[0, 3, 8, 16],
        labels=['Weeks 1-3', 'Weeks 4-8', 'Weeks 9-15']
    )

    # 1. Distribution of qb_value_shrunk by week bucket
    print("\n" + "=" * 70)
    print("QB VALUE (SHRUNK) DISTRIBUTION BY WEEK BUCKET")
    print("=" * 70)

    percentiles = [10, 25, 50, 75, 90]
    bucket_stats = []

    for bucket in ['Weeks 1-3', 'Weeks 4-8', 'Weeks 9-15']:
        bucket_df = df[df['week_bucket'] == bucket]
        if len(bucket_df) == 0:
            continue

        values = bucket_df['qb_value_shrunk'].dropna()
        stats = {
            'Bucket': bucket,
            'N': len(values),
            'Mean': values.mean(),
            'Std': values.std(),
        }
        for p in percentiles:
            stats[f'P{p}'] = np.percentile(values, p)

        bucket_stats.append(stats)

    stats_df = pd.DataFrame(bucket_stats)
    print(stats_df.to_string(index=False))
    results['distribution'] = stats_df

    # 2. Calculate recommended QB_SCALE
    print("\n" + "=" * 70)
    print("QB_SCALE CALIBRATION")
    print("=" * 70)

    # Target: 90th percentile reaches ~65% of cap by Week 8
    midseason = df[df['week_bucket'] == 'Weeks 4-8']
    if len(midseason) > 0:
        p90_shrunk = np.percentile(midseason['qb_value_shrunk'].dropna(), 90)
        target_points = 0.65 * qb_cap  # ~65% of cap
        recommended_scale = target_points / p90_shrunk if p90_shrunk > 0 else qb_scale

        print(f"Current QB_SCALE: {qb_scale}")
        print(f"90th percentile qb_value_shrunk (Weeks 4-8): {p90_shrunk:.4f}")
        print(f"Target qb_points (65% of cap): {target_points:.2f}")
        print(f"Recommended QB_SCALE: {recommended_scale:.2f}")

        results['recommended_scale'] = recommended_scale
        results['p90_midseason'] = p90_shrunk

    # 3. Cap binding frequency
    print("\n" + "=" * 70)
    print("CAP BINDING FREQUENCY")
    print("=" * 70)

    for bucket in ['Weeks 1-3', 'Weeks 4-8', 'Weeks 9-15']:
        bucket_df = df[df['week_bucket'] == bucket]
        if len(bucket_df) == 0:
            continue

        # Count where |qb_points| == qb_cap (within tolerance)
        cap_hit = (bucket_df['qb_points'].abs() >= qb_cap * 0.99).sum()
        total = len(bucket_df)
        pct = 100 * cap_hit / total if total > 0 else 0

        print(f"{bucket}: {cap_hit}/{total} ({pct:.1f}%) team-weeks hit cap")

    # 4. Uncertainty distribution
    print("\n" + "=" * 70)
    print("UNCERTAINTY DISTRIBUTION BY WEEK BUCKET")
    print("=" * 70)

    for bucket in ['Weeks 1-3', 'Weeks 4-8', 'Weeks 9-15']:
        bucket_df = df[df['week_bucket'] == bucket]
        if len(bucket_df) == 0:
            continue

        unc = bucket_df['qb_uncertainty'].dropna()
        print(f"{bucket}: mean={unc.mean():.3f}, median={unc.median():.3f}, "
              f"min={unc.min():.3f}, max={unc.max():.3f}")

    # 5. Unknown starters and starter changes
    print("\n" + "=" * 70)
    print("STARTER IDENTIFICATION QUALITY")
    print("=" * 70)

    for bucket in ['Weeks 1-3', 'Weeks 4-8', 'Weeks 9-15']:
        bucket_df = df[df['week_bucket'] == bucket]
        if len(bucket_df) == 0:
            continue

        unknown_pct = 100 * bucket_df['unknown_starter'].mean()
        changed_pct = 100 * bucket_df['starter_changed'].mean()
        prior_pct = 100 * bucket_df['prior_used'].mean()

        print(f"{bucket}: {unknown_pct:.1f}% unknown starter, {changed_pct:.1f}% starter changed, "
              f"{prior_pct:.1f}% using prior season")

    # 6. QB points distribution
    print("\n" + "=" * 70)
    print("QB POINTS (FINAL) DISTRIBUTION BY WEEK BUCKET")
    print("=" * 70)

    for bucket in ['Weeks 1-3', 'Weeks 4-8', 'Weeks 9-15']:
        bucket_df = df[df['week_bucket'] == bucket]
        if len(bucket_df) == 0:
            continue

        pts = bucket_df['qb_points'].dropna()
        eff_pts = bucket_df['qb_points_effective'].dropna()

        print(f"\n{bucket}:")
        print(f"  qb_points: min={pts.min():.2f}, mean={pts.mean():.2f}, max={pts.max():.2f}")
        print(f"  qb_points_effective: min={eff_pts.min():.2f}, mean={eff_pts.mean():.2f}, max={eff_pts.max():.2f}")
        print(f"  |qb_points_effective| > 1.0: {(eff_pts.abs() > 1.0).sum()} / {len(eff_pts)} ({100*(eff_pts.abs() > 1.0).mean():.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="QB Continuous Rating Calibration"
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2023, 2024],
        help="Years to analyze. Default: 2023 2024",
    )
    parser.add_argument(
        "--qb-shrinkage-k",
        type=float,
        default=DEFAULT_SHRINKAGE_K,
        help=f"Shrinkage parameter. Default: {DEFAULT_SHRINKAGE_K}",
    )
    parser.add_argument(
        "--qb-cap",
        type=float,
        default=DEFAULT_QB_CAP,
        help=f"Point adjustment cap. Default: {DEFAULT_QB_CAP}",
    )
    parser.add_argument(
        "--qb-scale",
        type=float,
        default=DEFAULT_QB_SCALE,
        help=f"PPA-to-points scale. Default: {DEFAULT_QB_SCALE}",
    )
    parser.add_argument(
        "--qb-prior-decay",
        type=float,
        default=DEFAULT_PRIOR_DECAY,
        help=f"Prior season decay. Default: {DEFAULT_PRIOR_DECAY}",
    )
    parser.add_argument(
        "--no-prior-season",
        action="store_true",
        help="Disable prior season data",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file for full calibration data",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("QB CONTINUOUS RATING CALIBRATION")
    print("=" * 70)
    print(f"Years: {args.years}")
    print(f"Shrinkage K: {args.qb_shrinkage_k}")
    print(f"Cap: ±{args.qb_cap}")
    print(f"Scale: {args.qb_scale}")
    print(f"Prior decay: {args.qb_prior_decay}")
    print(f"Use prior season: {not args.no_prior_season}")

    # Run calibration
    df = run_calibration(
        years=args.years,
        shrinkage_k=args.qb_shrinkage_k,
        qb_cap=args.qb_cap,
        qb_scale=args.qb_scale,
        prior_decay=args.qb_prior_decay,
        use_prior_season=not args.no_prior_season,
    )

    if df.empty:
        print("\nNo data collected. Check API connectivity.")
        return

    # Analyze
    results = analyze_calibration(df, args.qb_cap, args.qb_scale)

    # Save output
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nFull calibration data saved to {args.output}")


if __name__ == "__main__":
    main()
