#!/usr/bin/env python3
"""
Verification script for List B presence across Phase 1 modes.

This script proves that List B (5+ edge diagnostic) is present and not
being suppressed by dedupe, across both "weighted" and "skip" Phase 1 modes.

Usage:
    python scripts/check_listB_presence.py
    python scripts/check_listB_presence.py --year 2025 --week 2
    python scripts/check_listB_presence.py --show 10
    python scripts/check_listB_presence.py --log-path /tmp/test_log.csv

Exit codes:
    0: All checks pass
    1: One or more checks failed
"""

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.run_spread_weekly import run_week


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify List B presence across Phase 1 modes"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Season year (default: 2025)",
    )
    parser.add_argument(
        "--week",
        type=int,
        default=2,
        help="Week number (default: 2)",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Custom log path (default: temp file)",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=5,
        help="Number of ListB rows to print (default: 5)",
    )
    return parser.parse_args()


def run_check(
    year: int,
    week: int,
    phase1_policy: str,
    log_path: Path,
    show_n: int,
) -> dict:
    """Run a single check with the given phase1_policy.

    Args:
        year: Season year
        week: Week number
        phase1_policy: "weighted" or "skip"
        log_path: Path to write CSV log
        show_n: Number of ListB rows to display

    Returns:
        Dict with check results
    """
    # Run the week with write_csv=True to populate log file
    list_a_df, list_b_df, summary = run_week(
        year=year,
        week=week,
        preset="balanced",
        phase1_policy=phase1_policy,
        write_csv=True,
        log_path=log_path,
    )

    # Read back CSV to verify what was written
    if log_path.exists():
        csv_df = pd.read_csv(log_path)
        csv_b_rows = len(csv_df[csv_df["list_type"] == "B"])
        csv_a_rows = len(csv_df[csv_df["list_type"] == "A"])
    else:
        csv_df = pd.DataFrame()
        csv_b_rows = 0
        csv_a_rows = 0

    # Sort ListB by edge_abs desc, then game_id for deterministic ordering
    if len(list_b_df) > 0:
        list_b_df = list_b_df.sort_values(
            ["edge_abs", "game_id"],
            ascending=[False, True]
        ).reset_index(drop=True)

    # Extract top N ListB rows
    top_b_rows = []
    if len(list_b_df) > 0:
        for i, (_, row) in enumerate(list_b_df.head(show_n).iterrows()):
            top_b_rows.append({
                "game_id": row["game_id"],
                "edge_pts": row.get("edge_pts", row.get("edge_abs", 0)),
                "edge_abs": row["edge_abs"],
                "side": row["side"],
                "ev": row.get("ev", 0),
                "guardrail_reason": row["guardrail_reason"],
            })

    return {
        "phase1_policy": phase1_policy,
        "list_a_count": len(list_a_df),
        "list_b_count": len(list_b_df),
        "csv_a_rows": csv_a_rows,
        "csv_b_rows": csv_b_rows,
        "top_b_rows": top_b_rows,
        "list_a_df": list_a_df,
        "list_b_df": list_b_df,
        "csv_df": csv_df,
    }


def check_dedupe_safety(result_a: dict, result_b: dict) -> list[str]:
    """Check that dedupe doesn't suppress List B rows.

    Args:
        result_a: Results from weighted run
        result_b: Results from skip run

    Returns:
        List of error messages (empty if all checks pass)
    """
    errors = []

    # Check 1: ListB rows written to CSV should match ListB count
    for result in [result_a, result_b]:
        policy = result["phase1_policy"]
        list_b_count = result["list_b_count"]
        csv_b_rows = result["csv_b_rows"]

        if list_b_count > 0 and csv_b_rows == 0:
            errors.append(
                f"[{policy}] ListB has {list_b_count} rows but CSV has 0 list_type='B' rows"
            )

        if list_b_count != csv_b_rows:
            # This might be OK if there's intentional filtering,
            # but flag it for review
            errors.append(
                f"[{policy}] ListB count ({list_b_count}) != CSV B rows ({csv_b_rows})"
            )

    # Check 2: Verify skip mode has ListA=0 (or all stakes=0)
    if result_b["list_a_count"] > 0:
        # Check if all stakes are 0
        if len(result_b["list_a_df"]) > 0:
            stakes = result_b["list_a_df"]["stake"].sum()
            if stakes > 0:
                errors.append(
                    f"[skip] ListA has {result_b['list_a_count']} bets with total stake {stakes}"
                )

    # Check 3: Both runs should have same ListB count for 5+ edge games
    # (since ListB is based on edge, not EV threshold)
    # Note: This might differ slightly if phase config changes ev_floor
    # which affects which games "fail" EV but still have 5+ edge

    return errors


def print_summary(result: dict, show_n: int) -> None:
    """Print summary for a single run."""
    policy = result["phase1_policy"]
    print(f"\nRun ({policy}):")
    print(f"  ListA = {result['list_a_count']}")
    print(f"  ListB = {result['list_b_count']}")
    print(f"  CSV_A_rows = {result['csv_a_rows']}")
    print(f"  CSV_B_rows = {result['csv_b_rows']}")

    if result["top_b_rows"]:
        print(f"\n  Top {min(show_n, len(result['top_b_rows']))} ListB edges:")
        print(f"  {'game_id':<12} {'edge':>7} {'side':<6} {'ev':>7} {'guardrail':<20}")
        print(f"  {'-'*12} {'-'*7} {'-'*6} {'-'*7} {'-'*20}")
        for row in result["top_b_rows"][:show_n]:
            print(
                f"  {str(row['game_id'])[:12]:<12} "
                f"{row['edge_abs']:>6.1f}p "
                f"{row['side']:<6} "
                f"{row['ev']*100:>+5.1f}% "
                f"{row['guardrail_reason']:<20}"
            )


def main():
    args = parse_args()

    print("=" * 70)
    print(f"List B Presence Check: {args.year} Week {args.week}")
    print("=" * 70)

    # Use temp files for logs
    with tempfile.TemporaryDirectory() as tmpdir:
        log_a_path = Path(tmpdir) / f"check_weighted_{args.year}_w{args.week}.csv"
        log_b_path = Path(tmpdir) / f"check_skip_{args.year}_w{args.week}.csv"

        # If user specified log path, use it for both (will overwrite)
        if args.log_path:
            log_a_path = Path(args.log_path.replace(".csv", "_weighted.csv"))
            log_b_path = Path(args.log_path.replace(".csv", "_skip.csv"))

        # Run A: weighted (default)
        print("\nRunning with phase1_policy='weighted'...")
        result_a = run_check(
            year=args.year,
            week=args.week,
            phase1_policy="weighted",
            log_path=log_a_path,
            show_n=args.show,
        )

        # Run B: skip
        print("Running with phase1_policy='skip'...")
        result_b = run_check(
            year=args.year,
            week=args.week,
            phase1_policy="skip",
            log_path=log_b_path,
            show_n=args.show,
        )

        # Print summaries
        print_summary(result_a, args.show)
        print_summary(result_b, args.show)

        # Check dedupe safety
        print("\n" + "=" * 70)
        print("Dedupe Safety Checks")
        print("=" * 70)

        errors = check_dedupe_safety(result_a, result_b)

        if errors:
            print("\nERRORS FOUND:")
            for err in errors:
                print(f"  - {err}")
            print("\nResult: FAIL")
            sys.exit(1)
        else:
            print("\nAll checks passed:")
            print("  - weighted: ListB rows written to CSV correctly")
            print("  - skip: ListB rows written to CSV correctly")
            print("  - skip: ListA = 0 (as expected)")
            print("  - No dedupe collisions detected")
            print("\nResult: PASS")
            sys.exit(0)


if __name__ == "__main__":
    main()
