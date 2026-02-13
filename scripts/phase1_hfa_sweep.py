#!/usr/bin/env python3
"""Phase 1 HFA Offset Sweep - Find optimal offset for weeks 1-3.

Runs backtest with different HFA offset values to find the optimal
setting for Phase 1 (weeks 1-3) without affecting Core performance.
"""

import subprocess
import sys
import re
from dataclasses import dataclass


@dataclass
class SweepResult:
    offset: float
    games: int
    mae: float
    mse: float  # Mean signed error (predicted - actual)
    ats_close: float
    ats_close_record: str
    edge_3_close: float
    edge_3_close_record: str
    edge_5_close: float
    edge_5_close_record: str


def run_backtest(hfa_offset: float) -> str:
    """Run backtest with given HFA offset for Phase 1 only."""
    cmd = [
        "python3", "scripts/backtest.py",
        "--start-week", "1",
        "--end-week", "3",
        "--years", "2022", "2023", "2024", "2025",
        "--hfa-offset", str(hfa_offset),
        "--qb-continuous",
        "--qb-scale", "5.0",
        "--qb-phase1-only",
    ]

    print(f"\n{'='*60}")
    print(f"Running HFA offset = {hfa_offset}")
    print(f"{'='*60}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd="/Users/jason/Documents/CFB Power Ratings Model"
    )

    return result.stdout + result.stderr


def parse_results(output: str, offset: float) -> SweepResult:
    """Parse backtest output to extract metrics."""

    # Extract games count: "Total games predicted: 992"
    games_match = re.search(r"Total games predicted:\s*(\d+)", output)
    games = int(games_match.group(1)) if games_match else 0

    # Extract MAE: "| MAE (vs actual) | 13.95 pts |"
    mae_match = re.search(r"MAE \(vs actual\)\s*\|\s*([\d.]+)", output)
    mae = float(mae_match.group(1)) if mae_match else 0.0

    # Extract Mean Error (MSE - signed): "Mean error: +0.90 pts"
    mse_match = re.search(r"Mean error:\s*([-+]?[\d.]+)", output)
    mse = float(mse_match.group(1)) if mse_match else 0.0

    # Extract ATS Close: "| All | 463-488-9 | 48.7% |"
    ats_match = re.search(r"\|\s*All\s*\|\s*(\d+)-(\d+)(?:-\d+)?\s*\|\s*([\d.]+)%", output)
    if ats_match:
        ats_close = float(ats_match.group(3))
        ats_close_record = f"{ats_match.group(1)}-{ats_match.group(2)}"
    else:
        ats_close = 0.0
        ats_close_record = "0-0"

    # Extract 3+ Edge: "| 3+ pts | 300-315 (48.8%) |"
    edge_3_match = re.search(r"\|\s*3\+\s*pts\s*\|\s*(\d+)-(\d+)\s*\(([\d.]+)%\)", output)
    if edge_3_match:
        edge_3_close = float(edge_3_match.group(3))
        edge_3_close_record = f"{edge_3_match.group(1)}-{edge_3_match.group(2)}"
    else:
        edge_3_close = 0.0
        edge_3_close_record = "0-0"

    # Extract 5+ Edge: "| 5+ pts | 225-221 (50.4%) |"
    edge_5_match = re.search(r"\|\s*5\+\s*pts\s*\|\s*(\d+)-(\d+)\s*\(([\d.]+)%\)", output)
    if edge_5_match:
        edge_5_close = float(edge_5_match.group(3))
        edge_5_close_record = f"{edge_5_match.group(1)}-{edge_5_match.group(2)}"
    else:
        edge_5_close = 0.0
        edge_5_close_record = "0-0"

    return SweepResult(
        offset=offset,
        games=games,
        mae=mae,
        mse=mse,
        ats_close=ats_close,
        ats_close_record=ats_close_record,
        edge_3_close=edge_3_close,
        edge_3_close_record=edge_3_close_record,
        edge_5_close=edge_5_close,
        edge_5_close_record=edge_5_close_record,
    )


def main():
    # Sweep values: current baseline to aggressive
    offsets = [0.50, 0.75, 1.00, 1.25, 1.50, 1.60]

    results = []

    for offset in offsets:
        output = run_backtest(offset)
        result = parse_results(output, offset)
        results.append(result)

        # Print intermediate result
        print(f"\nOffset {offset}: MAE={result.mae:.2f}, MSE={result.mse:+.2f}, "
              f"ATS Close={result.ats_close:.1f}%, 5+ Edge={result.edge_5_close:.1f}%")

    # Print summary table
    print("\n" + "=" * 95)
    print("PHASE 1 HFA OFFSET SWEEP RESULTS (Weeks 1-3, 2022-2025)")
    print("=" * 95)
    print(f"{'Offset':>8} | {'Games':>6} | {'MAE':>6} | {'MSE':>7} | "
          f"{'ATS':>7} | {'ATS Rec':>10} | {'3+ Edge':>7} | {'3+Rec':>10} | "
          f"{'5+ Edge':>7} | {'5+Rec':>10}")
    print("-" * 95)

    baseline = results[0]

    for r in results:
        # Calculate deltas
        ats_delta = r.ats_close - baseline.ats_close
        edge_5_delta = r.edge_5_close - baseline.edge_5_close
        mse_delta = r.mse - baseline.mse

        # Mark improvements
        ats_mark = "↑" if ats_delta > 0.5 else ("↓" if ats_delta < -0.5 else " ")
        edge_5_mark = "↑" if edge_5_delta > 0.5 else ("↓" if edge_5_delta < -0.5 else " ")

        print(f"{r.offset:>8.2f} | {r.games:>6} | {r.mae:>6.2f} | {r.mse:>+7.2f} | "
              f"{r.ats_close:>6.1f}%{ats_mark}| {r.ats_close_record:>10} | {r.edge_3_close:>6.1f}% | {r.edge_3_close_record:>10} | "
              f"{r.edge_5_close:>6.1f}%{edge_5_mark}| {r.edge_5_close_record:>10}")

    print("-" * 95)

    # Find best offset
    best_5_edge = max(results, key=lambda r: r.edge_5_close)
    best_ats = max(results, key=lambda r: r.ats_close)
    lowest_mse = min(results, key=lambda r: abs(r.mse))

    print("\nBEST SETTINGS:")
    print(f"  Best 5+ Edge:   offset={best_5_edge.offset} → {best_5_edge.edge_5_close:.1f}% ({best_5_edge.edge_5_close_record})")
    print(f"  Best ATS:       offset={best_ats.offset} → {best_ats.ats_close:.1f}% ({best_ats.ats_close_record})")
    print(f"  Lowest |MSE|:   offset={lowest_mse.offset} → MSE={lowest_mse.mse:+.2f}")

    # Delta from baseline
    print("\nDELTA FROM BASELINE (offset=0.50):")
    print(f"{'Offset':>8} | {'5+ Edge Δ':>10} | {'ATS Δ':>8} | {'MSE Δ':>8}")
    print("-" * 40)
    for r in results:
        print(f"{r.offset:>8.2f} | {r.edge_5_close - baseline.edge_5_close:>+10.1f}% | "
              f"{r.ats_close - baseline.ats_close:>+8.1f}% | {r.mse - baseline.mse:>+8.2f}")

    # Now run open line test for the best offset if different from baseline
    if best_5_edge.offset != 0.50:
        print(f"\n{'='*60}")
        print(f"OPEN LINE VERIFICATION for offset={best_5_edge.offset}")
        print(f"{'='*60}")

        cmd = [
            "python3", "scripts/backtest.py",
            "--start-week", "1",
            "--end-week", "3",
            "--years", "2022", "2023", "2024", "2025",
            "--hfa-offset", str(best_5_edge.offset),
            "--qb-continuous",
            "--qb-scale", "5.0",
            "--qb-phase1-only",
            "--opening-line",  # Use opening line
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd="/Users/jason/Documents/CFB Power Ratings Model"
        )
        open_output = result.stdout + result.stderr
        open_result = parse_results(open_output, best_5_edge.offset)
        print(f"  Open line ATS: {open_result.ats_close:.1f}% ({open_result.ats_close_record})")
        print(f"  Open line 5+ Edge: {open_result.edge_5_close:.1f}% ({open_result.edge_5_close_record})")


if __name__ == "__main__":
    main()
