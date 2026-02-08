#!/usr/bin/env python3
"""
Generate and update documentation with current backtest metrics.

Runs the full backtest (or reads from cache), extracts performance metrics,
and updates CLAUDE.md and docs/MODEL_ARCHITECTURE.md with current values.

Usage:
    python3 scripts/generate_docs.py                 # Run backtest + update docs
    python3 scripts/generate_docs.py --skip-backtest  # Use cached metrics
    python3 scripts/generate_docs.py --dry-run        # Show changes without writing
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CACHE_FILE = project_root / "data" / "last_backtest_metrics.json"
CLAUDE_MD = project_root / "CLAUDE.md"
MODEL_ARCH_MD = project_root / "docs" / "MODEL_ARCHITECTURE.md"
TIMESTAMP_COMMENT = "<!-- Last validated: {date} by generate_docs.py -->"


def run_backtest_and_extract() -> dict:
    """Run the full backtest and extract structured metrics.

    Returns:
        Dictionary of metrics suitable for JSON serialization.
    """
    # Import backtest machinery
    from scripts.backtest import (
        fetch_all_season_data,
        run_backtest,
        get_phase,
        calculate_phase_metrics,
    )

    years = [2022, 2023, 2024, 2025]
    start_week = 1

    logger.info("Fetching season data...")
    season_data = fetch_all_season_data(
        years,
        use_priors=True,
        use_portal=True,
        portal_scale=0.15,
    )

    logger.info("Running backtest (start_week=1, closing lines)...")
    results = run_backtest(
        years=years,
        start_week=start_week,
        ridge_alpha=50.0,
        use_priors=True,
        hfa_value=2.5,
        prior_weight=8,
        season_data=season_data,
        efficiency_weight=0.45,
        explosiveness_weight=0.45,
        turnover_weight=0.10,
        asymmetric_garbage=True,
        fcs_penalty_elite=18.0,
        fcs_penalty_standard=32.0,
        use_portal=True,
        portal_scale=0.15,
        use_opening_line=False,  # Closing line is default for ATS
        hfa_global_offset=0.50,  # Calibrated Feb 2026
    )

    predictions_df = results["predictions"]
    ats_df = results["ats_results"]

    metrics = extract_metrics(predictions_df, ats_df)
    return metrics


def extract_metrics(predictions_df: pd.DataFrame, ats_df: pd.DataFrame) -> dict:
    """Extract all metrics needed for documentation from backtest results.

    Computes ATS against both closing and opening lines from the same
    ATS DataFrame (which contains both spread_open and spread_close).
    """
    from scripts.backtest import get_phase

    metrics = {}

    # --- Full season metrics ---
    total_games = len(predictions_df)
    full_mae = predictions_df["abs_error"].mean()
    full_rmse = np.sqrt((predictions_df["error"] ** 2).mean())

    metrics["full"] = {
        "games": int(total_games),
        "mae": round(float(full_mae), 2),
        "rmse": round(float(full_rmse), 2),
    }

    # --- Phase-level metrics ---
    # Add phase to predictions
    pred_with_phase = predictions_df.copy()
    pred_with_phase["phase"] = pred_with_phase["week"].apply(get_phase)

    for phase_name, phase_label, week_range in [
        ("Phase 1 (Calibration)", "phase1", "1-3"),
        ("Phase 2 (Core)", "phase2", "4-15"),
        ("Phase 3 (Postseason)", "phase3", "16+"),
    ]:
        phase_pred = pred_with_phase[pred_with_phase["phase"] == phase_name]
        if len(phase_pred) == 0:
            continue

        phase_mae = phase_pred["abs_error"].mean()
        phase_rmse = np.sqrt((phase_pred["error"] ** 2).mean())

        metrics[phase_label] = {
            "games": int(len(phase_pred)),
            "mae": round(float(phase_mae), 2),
            "rmse": round(float(phase_rmse), 2),
            "weeks": week_range,
        }

    # --- ATS metrics (both closing and opening) ---
    if ats_df is not None and not ats_df.empty:
        ats_with_phase = ats_df.copy()
        ats_with_phase["phase"] = ats_with_phase["week"].apply(get_phase)

        # Merge abs_error from predictions
        if "abs_error" not in ats_with_phase.columns and "game_id" in ats_with_phase.columns:
            error_cols = predictions_df[["game_id", "abs_error", "error"]].drop_duplicates()
            ats_with_phase = ats_with_phase.merge(error_cols, on="game_id", how="left")

        for line_type, spread_col in [("close", "spread_close"), ("open", "spread_open")]:
            _compute_ats_metrics(ats_with_phase, spread_col, line_type, metrics)

    # Add metadata
    metrics["_meta"] = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "years": [2022, 2023, 2024, 2025],
        "start_week": 1,
    }

    return metrics


def _compute_ats_metrics(ats_df: pd.DataFrame, spread_col: str, line_type: str, metrics: dict):
    """Compute ATS metrics for a given spread column (close or open).

    Recalculates ATS wins/losses against the specified spread column
    and stores results in metrics dict under keys like 'full_ats_close',
    'phase2_ats_close', etc.
    """
    from scripts.backtest import get_phase

    df = ats_df.copy()

    # Filter to games that have this spread type
    valid = df[df[spread_col].notna()].copy()
    if len(valid) == 0:
        return

    # Recalculate ATS against this specific spread
    actual_margin = valid["actual_margin"].values
    model_spread = valid["predicted_spread"].values
    vegas_spread = valid[spread_col].values

    model_spread_vegas = -model_spread
    edge = model_spread_vegas - vegas_spread
    model_pick_home = edge < 0

    home_cover = actual_margin + vegas_spread
    ats_win = np.where(model_pick_home, home_cover > 0, home_cover < 0)
    ats_push = home_cover == 0
    abs_edge = np.abs(edge)

    valid["_ats_win"] = ats_win
    valid["_ats_push"] = ats_push
    valid["_abs_edge"] = abs_edge

    # Compute for full + each phase
    phase_slices = [
        ("full", valid),
        ("phase1", valid[valid["phase"] == "Phase 1 (Calibration)"]),
        ("phase2", valid[valid["phase"] == "Phase 2 (Core)"]),
        ("phase3", valid[valid["phase"] == "Phase 3 (Postseason)"]),
    ]

    for key, slice_df in phase_slices:
        if len(slice_df) == 0:
            continue

        wins = int(slice_df["_ats_win"].sum())
        pushes = int(slice_df["_ats_push"].sum())
        losses = int(len(slice_df) - wins - pushes)
        total = wins + losses
        pct = round(wins / total * 100, 1) if total > 0 else 0.0

        ats_key = f"{key}_ats_{line_type}"
        metrics[ats_key] = {
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "pct": pct,
            "record": f"{wins}-{losses}",
        }

        # 3+ and 5+ edge for this slice
        for threshold in [3, 5]:
            edge_df = slice_df[slice_df["_abs_edge"] >= threshold]
            if len(edge_df) == 0:
                continue
            e_wins = int(edge_df["_ats_win"].sum())
            e_pushes = int(edge_df["_ats_push"].sum())
            e_losses = int(len(edge_df) - e_wins - e_pushes)
            e_total = e_wins + e_losses
            e_pct = round(e_wins / e_total * 100, 1) if e_total > 0 else 0.0

            edge_key = f"{key}_{threshold}plus_ats_{line_type}"
            metrics[edge_key] = {
                "wins": e_wins,
                "losses": e_losses,
                "games": int(len(edge_df)),
                "pct": e_pct,
                "record": f"{e_wins}-{e_losses}",
            }


def save_metrics_cache(metrics: dict):
    """Save metrics to JSON cache file."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics cached to {CACHE_FILE}")


def load_metrics_cache() -> dict:
    """Load metrics from JSON cache file."""
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"No cached metrics found at {CACHE_FILE}. "
            "Run without --skip-backtest first."
        )
    with open(CACHE_FILE, "r") as f:
        metrics = json.load(f)
    logger.info(f"Loaded cached metrics from {CACHE_FILE} (generated: {metrics.get('_meta', {}).get('generated', 'unknown')})")
    return metrics


def _get(metrics: dict, key: str, subkey: str, default="--"):
    """Safely get a nested metric value."""
    entry = metrics.get(key, {})
    if isinstance(entry, dict):
        return entry.get(subkey, default)
    return default


def _fmt_pct(val):
    """Format a percentage value."""
    if isinstance(val, (int, float)):
        return f"{val:.1f}%"
    return str(val)


def _fmt_num(val, decimals=2):
    """Format a numeric value to fixed decimal places."""
    if isinstance(val, (int, float)):
        return f"{val:.{decimals}f}"
    return str(val)


def build_claude_md_table(metrics: dict) -> str:
    """Build the CLAUDE.md baseline table from metrics."""
    # Full season
    full_games = _get(metrics, "full", "games")
    full_mae = _fmt_num(_get(metrics, "full", "mae"))
    full_rmse = _fmt_num(_get(metrics, "full", "rmse"))
    full_ats_close_pct = _fmt_pct(_get(metrics, "full_ats_close", "pct"))
    full_ats_open_pct = _fmt_pct(_get(metrics, "full_ats_open", "pct"))

    # Phase 1
    p1_games = _get(metrics, "phase1", "games")
    p1_mae = _fmt_num(_get(metrics, "phase1", "mae"))
    p1_ats_close = _fmt_pct(_get(metrics, "phase1_ats_close", "pct"))
    p1_ats_open = _fmt_pct(_get(metrics, "phase1_ats_open", "pct"))

    # Phase 2 (Core)
    p2_games = _get(metrics, "phase2", "games")
    p2_mae = _fmt_num(_get(metrics, "phase2", "mae"))
    p2_rmse = _fmt_num(_get(metrics, "phase2", "rmse"))
    p2_ats_close = _fmt_pct(_get(metrics, "phase2_ats_close", "pct"))
    p2_ats_open = _fmt_pct(_get(metrics, "phase2_ats_open", "pct"))

    # Phase 3
    p3_games = _get(metrics, "phase3", "games")
    p3_mae = _fmt_num(_get(metrics, "phase3", "mae"))
    p3_ats_close = _fmt_pct(_get(metrics, "phase3_ats_close", "pct"))
    p3_ats_open = _fmt_pct(_get(metrics, "phase3_ats_open", "pct"))

    # 3+ Edge (Core, close)
    e3_close = metrics.get("phase2_3plus_ats_close", {})
    e3_close_games = e3_close.get("games", "--")
    e3_close_record = e3_close.get("record", "--")
    e3_close_pct = _fmt_pct(e3_close.get("pct", "--"))
    e3_close_wins = e3_close.get("wins", "")
    e3_close_losses = e3_close.get("losses", "")
    e3_close_str = f"{e3_close_pct} ({e3_close_wins}-{e3_close_losses})" if e3_close_wins != "" else "--"

    # 3+ Edge (Core, open)
    e3_open = metrics.get("phase2_3plus_ats_open", {})
    e3_open_wins = e3_open.get("wins", "")
    e3_open_losses = e3_open.get("losses", "")
    e3_open_pct = _fmt_pct(e3_open.get("pct", "--"))
    e3_open_str = f"{e3_open_pct} ({e3_open_wins}-{e3_open_losses})" if e3_open_wins != "" else "--"

    # 5+ Edge (Core, close)
    e5_close = metrics.get("phase2_5plus_ats_close", {})
    e5_close_wins = e5_close.get("wins", "")
    e5_close_losses = e5_close.get("losses", "")
    e5_close_pct = _fmt_pct(e5_close.get("pct", "--"))
    e5_close_str = f"{e5_close_pct} ({e5_close_wins}-{e5_close_losses})" if e5_close_wins != "" else "--"

    # 5+ Edge (Core, open)
    e5_open = metrics.get("phase2_5plus_ats_open", {})
    e5_open_wins = e5_open.get("wins", "")
    e5_open_losses = e5_open.get("losses", "")
    e5_open_pct = _fmt_pct(e5_open.get("pct", "--"))
    e5_open_str = f"{e5_open_pct} ({e5_open_wins}-{e5_open_losses})" if e5_open_wins != "" else "--"

    # Build the table
    lines = [
        "| Slice | Weeks | Games | MAE | RMSE | ATS (Close) | ATS (Open) |",
        "|-------|-------|-------|-----|------|-------------|------------|",
        f"| **Full (`--start-week 1`)** | 1–Post | {full_games:,} | {full_mae} | {full_rmse} | {full_ats_close_pct} | {full_ats_open_pct} |",
        f"| Phase 1 (Calibration) | 1–3 | {p1_games:,} | {p1_mae} | — | {p1_ats_close} | {p1_ats_open} |",
        f"| **Phase 2 (Core)** | **4–15** | **{p2_games:,}** | **{p2_mae}** | **{p2_rmse}** | **{p2_ats_close}** | **{p2_ats_open}** |",
        f"| Phase 3 (Postseason) | 16+ | {p3_games:,} | {p3_mae} | — | {p3_ats_close} | {p3_ats_open} |",
        f"| 3+ Edge (Core) | 4–15 | {e3_close.get('games', '--'):,} | — | — | {e3_close_str} | {e3_open_str} |",
        f"| 5+ Edge (Core) | 4–15 | {e5_close.get('games', '--'):,} | — | — | {e5_close_str} | {e5_open_str} |",
    ]

    return "\n".join(lines)


def update_claude_md(metrics: dict, dry_run: bool = False) -> bool:
    """Update the baseline table in CLAUDE.md.

    Returns True if changes were made (or would be made in dry_run).
    """
    if not CLAUDE_MD.exists():
        logger.error(f"CLAUDE.md not found at {CLAUDE_MD}")
        return False

    content = CLAUDE_MD.read_text()

    # Update the date in the section header
    today = datetime.now().strftime("%Y-%m-%d")
    content = re.sub(
        r"(## ✅ Current Production Baseline \(2022-2025 backtest, as of )\d{4}-\d{2}-\d{2}\)",
        rf"\g<1>{today})",
        content,
    )

    # Find and replace the baseline table
    # Pattern: starts with "| Slice |" header, ends before a blank line or non-table line
    table_pattern = re.compile(
        r"(\| Slice \| Weeks \| Games \|.*?\n"  # Header
        r"\|[-| ]+\n"  # Separator
        r"(?:\|.*\n)+)",  # Data rows (greedy, multiple lines)
        re.MULTILINE,
    )

    new_table = build_claude_md_table(metrics) + "\n"
    match = table_pattern.search(content)

    if not match:
        logger.error("Could not find baseline table in CLAUDE.md")
        return False

    old_table = match.group(0)

    if old_table.strip() == new_table.strip():
        logger.info("CLAUDE.md baseline table is already up to date")
        return False

    new_content = content[:match.start()] + new_table + content[match.end():]

    # Also update the Quant Auditor MAE baseline if it references a specific value
    p2_mae = _get(metrics, "phase2", "mae")
    if isinstance(p2_mae, (int, float)):
        new_content = re.sub(
            r"(\*\*MAE Baseline:\*\* )\d+\.\d+",
            rf"\g<1>{p2_mae}",
            new_content,
        )

    # Update/add timestamp
    new_content = _update_timestamp(new_content, today)

    if dry_run:
        logger.info(f"[DRY RUN] Would update CLAUDE.md baseline table")
        print(f"\n--- CLAUDE.md table (new) ---")
        print(new_table)
        print("---")
        return True

    CLAUDE_MD.write_text(new_content)
    logger.info(f"Updated CLAUDE.md baseline table")
    return True


def update_model_architecture_md(metrics: dict, dry_run: bool = False) -> bool:
    """Update hardcoded metrics in docs/MODEL_ARCHITECTURE.md.

    Returns True if changes were made.
    """
    if not MODEL_ARCH_MD.exists():
        logger.warning(f"MODEL_ARCHITECTURE.md not found at {MODEL_ARCH_MD}")
        return False

    content = MODEL_ARCH_MD.read_text()
    original = content
    today = datetime.now().strftime("%Y-%m-%d")

    # Update the "Performance by Season Phase" table
    phase_table_pattern = re.compile(
        r"(\| Phase \| Weeks \| Games \| MAE \|.*?\n"
        r"\|[-| ]+\n"
        r"(?:\|.*\n)+)",
        re.MULTILINE,
    )

    match = phase_table_pattern.search(content)
    if match:
        new_phase_table = _build_arch_phase_table(metrics)
        content = content[:match.start()] + new_phase_table + "\n" + content[match.end():]

    # Update the "Against The Spread" table in Core detail
    ats_table_pattern = re.compile(
        r"(\| Edge Filter \| vs Closing Line \| vs Opening Line \|\n"
        r"\|[-| ]+\n"
        r"(?:\|.*\n)+)",
        re.MULTILINE,
    )

    match = ats_table_pattern.search(content)
    if match:
        new_ats_table = _build_arch_ats_table(metrics)
        content = content[:match.start()] + new_ats_table + "\n" + content[match.end():]

    # Update the summary line about total games
    full_games = _get(metrics, "full", "games")
    if isinstance(full_games, int):
        content = re.sub(
            r"Walk-forward backtest across 4 seasons covering the full CFB calendar \([\d,]+ games\)",
            f"Walk-forward backtest across 4 seasons covering the full CFB calendar ({full_games:,} games)",
            content,
        )

    # Update "All metrics verified" date
    content = re.sub(
        r"\*All metrics verified \d{4}-\d{2}-\d{2}\.\*",
        f"*All metrics verified {today}.*",
        content,
    )

    # Update "Last Updated" header
    content = re.sub(
        r"\*\*Last Updated:\*\* .*",
        f"**Last Updated:** {datetime.now().strftime('%B %d, %Y').replace(' 0', ' ')} (auto-generated by generate_docs.py)",
        content,
        count=1,
    )

    # Update timestamp
    content = _update_timestamp(content, today)

    if content == original:
        logger.info("MODEL_ARCHITECTURE.md is already up to date")
        return False

    if dry_run:
        logger.info("[DRY RUN] Would update MODEL_ARCHITECTURE.md")
        return True

    MODEL_ARCH_MD.write_text(content)
    logger.info("Updated MODEL_ARCHITECTURE.md")
    return True


def _build_arch_phase_table(metrics: dict) -> str:
    """Build the performance by season phase table for MODEL_ARCHITECTURE.md."""
    # This table has more columns than CLAUDE.md
    def _phase_row(phase_key, phase_name, bold=False):
        m = metrics.get(phase_key, {})
        games = m.get("games", "--")
        mae = _fmt_num(m.get("mae", "--"))
        rmse = _fmt_num(m.get("rmse", "—"))

        ats_close = metrics.get(f"{phase_key}_ats_close", {})
        ats_open = metrics.get(f"{phase_key}_ats_open", {})
        ats_close_pct = _fmt_pct(ats_close.get("pct", "--"))
        ats_open_pct = _fmt_pct(ats_open.get("pct", "--"))

        e3_close = metrics.get(f"{phase_key}_3plus_ats_close", {})
        e5_close = metrics.get(f"{phase_key}_5plus_ats_close", {})
        e5_open = metrics.get(f"{phase_key}_5plus_ats_open", {})

        e3_pct = _fmt_pct(e3_close.get("pct", "--"))
        e5_close_pct = _fmt_pct(e5_close.get("pct", "--"))
        e5_open_pct = _fmt_pct(e5_open.get("pct", "--"))

        if bold:
            return (
                f"| **{phase_name}** | **{m.get('weeks', '--')}** | **{games:,}** | "
                f"**{mae}** | **{rmse}** | **{ats_close_pct}** | **{ats_open_pct}** | "
                f"**{e3_pct}** | **{e5_close_pct}** | **{e5_open_pct}** |"
            )
        return (
            f"| {phase_name} | {m.get('weeks', '--')} | {games:,} | "
            f"{mae} | {rmse} | {ats_close_pct} | {ats_open_pct} | "
            f"{e3_pct} | {e5_close_pct} | {e5_open_pct} |"
        )

    # Full row
    full = metrics.get("full", {})
    full_ats_c = _fmt_pct(_get(metrics, "full_ats_close", "pct"))
    full_ats_o = _fmt_pct(_get(metrics, "full_ats_open", "pct"))
    full_e3 = _fmt_pct(_get(metrics, "full_3plus_ats_close", "pct"))
    full_e5c = _fmt_pct(_get(metrics, "full_5plus_ats_close", "pct"))
    full_e5o = _fmt_pct(_get(metrics, "full_5plus_ats_open", "pct"))

    lines = [
        "| Phase | Weeks | Games | MAE | RMSE | ATS % (Close) | ATS % (Open) | 3+ Edge (Close) | 5+ Edge (Close) | 5+ Edge (Open) |",
        "|-------|-------|-------|-----|------|---------------|--------------|-----------------|-----------------|----------------|",
        _phase_row("phase1", "Calibration"),
        _phase_row("phase2", "Core", bold=True),
        _phase_row("phase3", "Postseason"),
        f"| **Full Season** | All | {full.get('games', '--'):,} | {_fmt_num(full.get('mae', '--'))} | {_fmt_num(full.get('rmse', '--'))} | {full_ats_c} | {full_ats_o} | {full_e3} | {full_e5c} | {full_e5o} |",
    ]
    return "\n".join(lines)


def _build_arch_ats_table(metrics: dict) -> str:
    """Build the ATS detail table for Core season in MODEL_ARCHITECTURE.md."""
    # All picks
    p2_close = metrics.get("phase2_ats_close", {})
    p2_open = metrics.get("phase2_ats_open", {})

    close_record = f"{p2_close.get('wins', '--')}-{p2_close.get('losses', '--')}"
    close_pct = _fmt_pct(p2_close.get("pct", "--"))
    open_record = f"{p2_open.get('wins', '--')}-{p2_open.get('losses', '--')}"
    open_pct = _fmt_pct(p2_open.get("pct", "--"))

    # 3+ edge
    e3c = metrics.get("phase2_3plus_ats_close", {})
    e3o = metrics.get("phase2_3plus_ats_open", {})
    e3c_str = f"{e3c.get('wins', '--')}-{e3c.get('losses', '--')} ({_fmt_pct(e3c.get('pct', '--'))})"
    e3o_str = f"{e3o.get('wins', '--')}-{e3o.get('losses', '--')} ({_fmt_pct(e3o.get('pct', '--'))})"

    # 5+ edge
    e5c = metrics.get("phase2_5plus_ats_close", {})
    e5o = metrics.get("phase2_5plus_ats_open", {})
    e5c_str = f"{e5c.get('wins', '--')}-{e5c.get('losses', '--')} ({_fmt_pct(e5c.get('pct', '--'))})"
    e5o_str = f"{e5o.get('wins', '--')}-{e5o.get('losses', '--')} ({_fmt_pct(e5o.get('pct', '--'))})"

    lines = [
        "| Edge Filter | vs Closing Line | vs Opening Line |",
        "|-------------|-----------------|-----------------|",
        f"| **All picks** | {close_record} ({close_pct}) | {open_record} ({open_pct}) |",
        f"| **3+ pt edge** | {e3c_str} | {e3o_str} |",
        f"| **5+ pt edge** | {e5c_str} | {e5o_str} |",
    ]
    return "\n".join(lines)


def _update_timestamp(content: str, today: str) -> str:
    """Update or add the validation timestamp at the end of a file."""
    timestamp = TIMESTAMP_COMMENT.format(date=today)

    # Replace existing timestamp
    if "<!-- Last validated:" in content:
        content = re.sub(
            r"<!-- Last validated:.*?-->",
            timestamp,
            content,
        )
    else:
        # Add at end
        content = content.rstrip() + "\n\n" + timestamp + "\n"

    return content


def main():
    parser = argparse.ArgumentParser(
        description="Generate/update documentation with current backtest metrics"
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Use cached metrics instead of running backtest",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without writing files",
    )
    args = parser.parse_args()

    # Ensure we're running from project root
    os.chdir(project_root)

    if args.skip_backtest:
        metrics = load_metrics_cache()
    else:
        logger.info("Running full backtest to extract metrics...")
        metrics = run_backtest_and_extract()
        save_metrics_cache(metrics)

    # Print summary
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    print(f"  Full:  {_get(metrics, 'full', 'games'):,} games, MAE={_fmt_num(_get(metrics, 'full', 'mae'))}, RMSE={_fmt_num(_get(metrics, 'full', 'rmse'))}")
    print(f"  Core:  {_get(metrics, 'phase2', 'games'):,} games, MAE={_fmt_num(_get(metrics, 'phase2', 'mae'))}, RMSE={_fmt_num(_get(metrics, 'phase2', 'rmse'))}")
    print(f"  ATS Close (Core): {_fmt_pct(_get(metrics, 'phase2_ats_close', 'pct'))}")
    print(f"  ATS Open  (Core): {_fmt_pct(_get(metrics, 'phase2_ats_open', 'pct'))}")
    e5c = metrics.get("phase2_5plus_ats_close", {})
    e5o = metrics.get("phase2_5plus_ats_open", {})
    print(f"  5+ Edge Close (Core): {_fmt_pct(e5c.get('pct', '--'))} ({e5c.get('wins', '--')}-{e5c.get('losses', '--')})")
    print(f"  5+ Edge Open  (Core): {_fmt_pct(e5o.get('pct', '--'))} ({e5o.get('wins', '--')}-{e5o.get('losses', '--')})")
    print("=" * 60)

    # Update docs
    updated = []
    if update_claude_md(metrics, dry_run=args.dry_run):
        updated.append("CLAUDE.md")
    if update_model_architecture_md(metrics, dry_run=args.dry_run):
        updated.append("docs/MODEL_ARCHITECTURE.md")

    if updated:
        action = "Would update" if args.dry_run else "Updated"
        print(f"\n{action}: {', '.join(updated)}")
    else:
        print("\nNo documentation changes needed.")


if __name__ == "__main__":
    main()
