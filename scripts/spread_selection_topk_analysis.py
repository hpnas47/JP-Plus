#!/usr/bin/env python3
"""Top-K by EV analysis and rolling-window calibration comparison.

This script compares calibration variants at fixed bet volumes (Top-K by EV)
to resolve the "tiny volume / huge ROI" vs "realistic volume / modest ROI" tradeoff.

Variants compared:
- INCLUDE_2022: Current default (all prior years in training)
- EXCLUDE_2022: exclude_years_from_training=[2022]
- ROLLING_3: training_window_seasons=3 (most recent 3 seasons)
- ROLLING_2: training_window_seasons=2 (most recent 2 seasons)

Output:
- data/outputs/spread_selection_topk_comparison.md
- data/outputs/spread_selection_topk_comparison.json
"""

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import expit

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.spread_selection.calibration import (
    load_and_normalize_game_data,
    walk_forward_validate,
    breakeven_prob,
    calculate_ev_vectorized,
)


@dataclass
class TopKResult:
    """Results for a top-K selection."""
    k: int
    n_actual: int  # May be < k if not enough games
    wins: int
    losses: int
    pushes: int
    ats_pct: float
    roi_pct: float
    avg_ev: float
    avg_edge_abs: float
    avg_p_cover: float


@dataclass
class ThresholdResult:
    """Results for a threshold-based strategy."""
    threshold_type: str  # "EV3", "EV5", "5pt"
    threshold_value: float
    n_bets: int
    wins: int
    losses: int
    pushes: int
    ats_pct: float
    roi_pct: float
    avg_ev: float
    avg_edge_abs: float
    avg_p_cover: float
    is_small_sample: bool  # True if N < 20


@dataclass
class FoldDiagnostics:
    """Diagnostics for a single fold."""
    eval_year: int
    n_train: int
    train_years: list[int]
    slope: float
    intercept: float
    p_cover_at_zero: float
    p_cover_at_5: float
    p_cover_at_10: float
    breakeven_edge: float


@dataclass
class VariantResults:
    """Complete results for a calibration variant."""
    name: str
    description: str
    fold_diagnostics: list[FoldDiagnostics]
    topk_by_season: dict[str, list[TopKResult]]  # season -> [K=25, K=50, K=100]
    threshold_by_season: dict[str, list[ThresholdResult]]  # season -> [EV3, EV5, 5pt]
    brier_score: float
    n_eval_games: int


def compute_topk_results(
    eval_df: pd.DataFrame,
    k_values: list[int] = [25, 50, 100],
) -> list[TopKResult]:
    """Compute Top-K results for a set of games.

    Args:
        eval_df: DataFrame with EV, jp_side_covered, push, edge_abs, p_cover_no_push
        k_values: List of K values to evaluate

    Returns:
        List of TopKResult for each K
    """
    # Filter to eligible games (non-push for ranking, but we'll track pushes separately)
    eligible = eval_df[eval_df["p_cover_no_push"].notna() & (eval_df["edge_abs"] > 0)].copy()

    # Sort by EV descending
    eligible = eligible.sort_values("ev", ascending=False)

    results = []
    for k in k_values:
        top_k = eligible.head(k)
        n_actual = len(top_k)

        if n_actual == 0:
            results.append(TopKResult(
                k=k, n_actual=0, wins=0, losses=0, pushes=0,
                ats_pct=0, roi_pct=0, avg_ev=0, avg_edge_abs=0, avg_p_cover=0
            ))
            continue

        # Separate pushes from outcomes
        non_push = top_k[~top_k["push"]]
        pushes = top_k["push"].sum()

        wins = non_push["jp_side_covered"].sum()
        losses = len(non_push) - wins

        # ATS % excludes pushes from denominator
        ats_pct = wins / len(non_push) * 100 if len(non_push) > 0 else 0

        # ROI: win pays 100/110, loss costs 1.0, push returns stake
        roi_pct = (wins * (100/110) - losses) / n_actual * 100 if n_actual > 0 else 0

        results.append(TopKResult(
            k=k,
            n_actual=n_actual,
            wins=int(wins),
            losses=int(losses),
            pushes=int(pushes),
            ats_pct=ats_pct,
            roi_pct=roi_pct,
            avg_ev=top_k["ev"].mean(),
            avg_edge_abs=top_k["edge_abs"].mean(),
            avg_p_cover=top_k["p_cover_no_push"].mean(),
        ))

    return results


def compute_threshold_results(
    eval_df: pd.DataFrame,
) -> list[ThresholdResult]:
    """Compute threshold-based strategy results.

    Args:
        eval_df: DataFrame with EV, jp_side_covered, push, edge_abs, p_cover_no_push

    Returns:
        List of ThresholdResult for each strategy
    """
    eligible = eval_df[eval_df["p_cover_no_push"].notna() & (eval_df["edge_abs"] > 0)].copy()

    strategies = [
        ("5pt", "edge_abs", 5.0),
        ("EV3", "ev", 0.03),
        ("EV5", "ev", 0.05),
    ]

    results = []
    for name, col, threshold in strategies:
        subset = eligible[eligible[col] >= threshold]
        n_bets = len(subset)

        if n_bets == 0:
            results.append(ThresholdResult(
                threshold_type=name, threshold_value=threshold,
                n_bets=0, wins=0, losses=0, pushes=0,
                ats_pct=0, roi_pct=0, avg_ev=0, avg_edge_abs=0, avg_p_cover=0,
                is_small_sample=True
            ))
            continue

        non_push = subset[~subset["push"]]
        pushes = subset["push"].sum()
        wins = non_push["jp_side_covered"].sum()
        losses = len(non_push) - wins

        ats_pct = wins / len(non_push) * 100 if len(non_push) > 0 else 0
        roi_pct = (wins * (100/110) - losses) / n_bets * 100 if n_bets > 0 else 0

        results.append(ThresholdResult(
            threshold_type=name,
            threshold_value=threshold,
            n_bets=n_bets,
            wins=int(wins),
            losses=int(losses),
            pushes=int(pushes),
            ats_pct=ats_pct,
            roi_pct=roi_pct,
            avg_ev=subset["ev"].mean(),
            avg_edge_abs=subset["edge_abs"].mean(),
            avg_p_cover=subset["p_cover_no_push"].mean(),
            is_small_sample=n_bets < 20,
        ))

    return results


def compute_overlap(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    k: int,
) -> dict:
    """Compute overlap between two variants' top-K sets.

    Args:
        df1: First variant's eval DataFrame (with EV computed)
        df2: Second variant's eval DataFrame (with EV computed)
        k: Number of top games to compare

    Returns:
        Dict with overlap statistics
    """
    # Get top K game_ids from each
    top1 = set(df1.nlargest(k, "ev")["game_id"].tolist())
    top2 = set(df2.nlargest(k, "ev")["game_id"].tolist())

    overlap = top1 & top2
    only_1 = top1 - top2
    only_2 = top2 - top1

    return {
        "k": k,
        "overlap_count": len(overlap),
        "overlap_pct": len(overlap) / k * 100 if k > 0 else 0,
        "only_variant1": len(only_1),
        "only_variant2": len(only_2),
    }


def run_variant(
    normalized_df: pd.DataFrame,
    name: str,
    description: str,
    exclude_years: list[int] | None = None,
    training_window: int | None = None,
    eval_years: list[int] = [2024, 2025],
) -> tuple[VariantResults, pd.DataFrame]:
    """Run analysis for a single calibration variant.

    Args:
        normalized_df: Normalized game data
        name: Variant name
        description: Variant description
        exclude_years: Years to exclude from training
        training_window: Rolling window size (None = all prior years)
        eval_years: Years to evaluate on

    Returns:
        Tuple of (VariantResults, eval_df with EV computed)
    """
    wf_result = walk_forward_validate(
        normalized_df,
        min_train_seasons=2,
        exclude_covid=True,
        exclude_years_from_training=exclude_years,
        training_window_seasons=training_window,
    )

    # Get evaluation games
    eval_df = wf_result.game_results[
        wf_result.game_results["year"].isin(eval_years)
    ].copy()

    # Compute EV
    mask = eval_df["p_cover_no_push"].notna()
    eval_df.loc[mask, "ev"] = calculate_ev_vectorized(
        eval_df.loc[mask, "p_cover_no_push"].values
    )

    # Fold diagnostics
    fold_diagnostics = []
    for fold in wf_result.fold_summaries:
        if fold["eval_year"] in eval_years:
            p_at_5 = expit(fold["intercept"] + fold["slope"] * 5)
            p_at_10 = expit(fold["intercept"] + fold["slope"] * 10)
            fold_diagnostics.append(FoldDiagnostics(
                eval_year=int(fold["eval_year"]),
                n_train=int(fold["n_train"]),
                train_years=[int(y) for y in fold["years_trained"]],
                slope=float(fold["slope"]),
                intercept=float(fold["intercept"]),
                p_cover_at_zero=float(fold["p_cover_at_zero"]),
                p_cover_at_5=float(p_at_5),
                p_cover_at_10=float(p_at_10),
                breakeven_edge=float(fold["breakeven_edge"]),
            ))

    # Top-K results by season
    topk_by_season = {}
    threshold_by_season = {}

    for year in eval_years + ["pooled"]:
        if year == "pooled":
            season_df = eval_df
        else:
            season_df = eval_df[eval_df["year"] == year]

        topk_by_season[str(year)] = compute_topk_results(season_df)
        threshold_by_season[str(year)] = compute_threshold_results(season_df)

    return VariantResults(
        name=name,
        description=description,
        fold_diagnostics=fold_diagnostics,
        topk_by_season=topk_by_season,
        threshold_by_season=threshold_by_season,
        brier_score=float(wf_result.overall_brier),
        n_eval_games=len(eval_df),
    ), eval_df


def generate_markdown_report(
    variants: dict[str, VariantResults],
    overlaps: dict[str, dict],
    eval_years: list[int],
) -> str:
    """Generate markdown report."""
    lines = [
        "# Spread Selection Calibration Comparison Report",
        "",
        "## Executive Summary",
        "",
        "This report compares calibration variants at fixed bet volumes (Top-K by EV) ",
        "to assess which produces the best risk-adjusted returns without overfitting to EV thresholds.",
        "",
        f"**Evaluation Years:** {eval_years}",
        "",
        "### Variants Compared",
        "",
    ]

    for name, v in variants.items():
        lines.append(f"- **{name}**: {v.description}")
    lines.append("")

    # Fold diagnostics comparison
    lines.extend([
        "## Part 1: Fold Diagnostics Comparison",
        "",
        "| Variant | Eval Year | Train Years | N Train | Slope | P(c)@5 | P(c)@10 | BE Edge |",
        "|---------|-----------|-------------|---------|-------|--------|---------|---------|",
    ])

    for name, v in variants.items():
        for fold in v.fold_diagnostics:
            train_str = ",".join(str(y) for y in fold.train_years)
            lines.append(
                f"| {name} | {fold.eval_year} | {train_str} | {fold.n_train} | "
                f"{fold.slope:.5f} | {fold.p_cover_at_5*100:.1f}% | {fold.p_cover_at_10*100:.1f}% | "
                f"{fold.breakeven_edge:.1f} |"
            )
    lines.append("")

    # Top-K comparison (pooled)
    lines.extend([
        "## Part 2: Top-K by EV Comparison (Pooled Across Seasons)",
        "",
        "Volume-controlled comparison: same number of bets selected, compare outcomes.",
        "",
    ])

    for k in [25, 50, 100]:
        lines.extend([
            f"### Top-{k} Bets by EV",
            "",
            "| Variant | N | W-L-P | ATS% | ROI% | Avg EV | Avg Edge | Avg P(c) |",
            "|---------|---|-------|------|------|--------|----------|----------|",
        ])

        for name, v in variants.items():
            result = [r for r in v.topk_by_season["pooled"] if r.k == k][0]
            wlp = f"{result.wins}-{result.losses}-{result.pushes}"
            lines.append(
                f"| {name} | {result.n_actual} | {wlp} | {result.ats_pct:.1f}% | "
                f"{result.roi_pct:+.1f}% | {result.avg_ev:.3f} | {result.avg_edge_abs:.1f} | "
                f"{result.avg_p_cover:.3f} |"
            )
        lines.append("")

    # Overlap analysis
    lines.extend([
        "## Part 3: Top-K Overlap Analysis",
        "",
        "How much do the top-K sets differ between variants?",
        "",
        "| Comparison | K | Overlap | Overlap% | Only V1 | Only V2 |",
        "|------------|---|---------|----------|---------|---------|",
    ])

    for key, overlap_data in overlaps.items():
        for k_data in overlap_data:
            lines.append(
                f"| {key} | {k_data['k']} | {k_data['overlap_count']} | "
                f"{k_data['overlap_pct']:.1f}% | {k_data['only_variant1']} | {k_data['only_variant2']} |"
            )
    lines.append("")

    # Season-by-season threshold results
    lines.extend([
        "## Part 4: Threshold Strategy Results (Season-by-Season)",
        "",
        "Traditional threshold-based strategies for comparison.",
        "",
    ])

    for year in eval_years + ["pooled"]:
        lines.extend([
            f"### {year}",
            "",
            "| Variant | Strategy | N | W-L-P | ATS% | ROI% | Small Sample? |",
            "|---------|----------|---|-------|------|------|---------------|",
        ])

        for name, v in variants.items():
            for result in v.threshold_by_season[str(year)]:
                wlp = f"{result.wins}-{result.losses}-{result.pushes}"
                small = "YES" if result.is_small_sample else ""
                lines.append(
                    f"| {name} | {result.threshold_type} | {result.n_bets} | {wlp} | "
                    f"{result.ats_pct:.1f}% | {result.roi_pct:+.1f}% | {small} |"
                )
        lines.append("")

    # Recommendation
    lines.extend([
        "## Recommendation",
        "",
    ])

    # Find best variant at each K
    best_at_k = {}
    for k in [25, 50, 100]:
        best_roi = -999
        best_name = None
        for name, v in variants.items():
            result = [r for r in v.topk_by_season["pooled"] if r.k == k][0]
            if result.roi_pct > best_roi:
                best_roi = result.roi_pct
                best_name = name
        best_at_k[k] = (best_name, best_roi)

    lines.extend([
        "### Best Variant at Fixed Volume (Top-K)",
        "",
    ])
    for k, (name, roi) in best_at_k.items():
        lines.append(f"- **Top-{k}**: {name} ({roi:+.1f}% ROI)")
    lines.append("")

    # Compare rolling vs exclude
    include = variants.get("INCLUDE_2022")
    exclude = variants.get("EXCLUDE_2022")
    rolling3 = variants.get("ROLLING_3")

    if rolling3 and exclude:
        r3_roi = [r for r in rolling3.topk_by_season["pooled"] if r.k == 50][0].roi_pct
        ex_roi = [r for r in exclude.topk_by_season["pooled"] if r.k == 50][0].roi_pct

        lines.extend([
            "### Rolling Window vs Explicit Exclusion",
            "",
            f"At Top-50 volume:",
            f"- ROLLING_3 ROI: {r3_roi:+.1f}%",
            f"- EXCLUDE_2022 ROI: {ex_roi:+.1f}%",
            "",
        ])

        if abs(r3_roi - ex_roi) < 2:
            lines.append("Rolling-window produces similar results to explicit 2022 exclusion, "
                        "suggesting it's a principled alternative that doesn't require manual year selection.")
        elif r3_roi > ex_roi:
            lines.append("Rolling-window outperforms explicit exclusion, suggesting recency "
                        "adaptation is more important than specific year exclusion.")
        else:
            lines.append("Explicit 2022 exclusion outperforms rolling-window, suggesting "
                        "2022's anomalous behavior is the primary issue rather than general recency.")
        lines.append("")

    # Final recommendation
    if include:
        inc_roi_50 = [r for r in include.topk_by_season["pooled"] if r.k == 50][0].roi_pct
        inc_n_ev5 = [r for r in include.threshold_by_season["pooled"] if r.threshold_type == "EV5"][0].n_bets

        lines.extend([
            "### INCLUDE_2022 as Ultra-Conservative Mode",
            "",
            f"INCLUDE_2022 produces only {inc_n_ev5} EV5 bets but at high hit rate. "
            f"At Top-50 volume: {inc_roi_50:+.1f}% ROI.",
            "",
            "This should be considered an **ultra-conservative mode** that:",
            "- Naturally filters to only extreme-edge bets",
            "- Accepts low volume for high confidence",
            "- May be appropriate for users prioritizing win rate over volume",
            "",
        ])

    lines.extend([
        "### Summary",
        "",
        "Based on volume-controlled comparison:",
        "",
    ])

    # Determine overall recommendation
    best_overall = max(variants.items(), key=lambda x: [r for r in x[1].topk_by_season["pooled"] if r.k == 50][0].roi_pct)
    lines.append(f"**Recommended default:** {best_overall[0]}")
    lines.append("")
    lines.append("---")
    lines.append("*Generated by spread_selection_topk_analysis.py*")

    return "\n".join(lines)


def main():
    """Run full Top-K and rolling-window analysis."""
    print("=" * 80)
    print("SPREAD SELECTION TOP-K ANALYSIS")
    print("=" * 80)

    # Load data
    csv_path = "data/spread_selection/ats_export.csv"
    print(f"\nLoading data from {csv_path}...")
    raw_df = pd.read_csv(csv_path)
    normalized_df = load_and_normalize_game_data(raw_df, jp_convention="pos_home_favored")
    print(f"Loaded {len(normalized_df)} games")

    eval_years = [2024, 2025]

    # Define variants
    variant_configs = [
        ("INCLUDE_2022", "All prior years in training (current default)", None, None),
        ("EXCLUDE_2022", "Exclude 2022 from training", [2022], None),
        ("ROLLING_3", "Most recent 3 seasons before eval year", None, 3),
        ("ROLLING_2", "Most recent 2 seasons before eval year", None, 2),
    ]

    # Run all variants
    variants = {}
    eval_dfs = {}

    for name, desc, exclude, window in variant_configs:
        print(f"\nRunning {name}...")
        try:
            result, eval_df = run_variant(
                normalized_df,
                name=name,
                description=desc,
                exclude_years=exclude,
                training_window=window,
                eval_years=eval_years,
            )
            variants[name] = result
            eval_dfs[name] = eval_df
            print(f"  Completed: {result.n_eval_games} eval games, Brier={result.brier_score:.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")

    # Compute overlaps
    print("\nComputing top-K overlaps...")
    overlaps = {}
    variant_names = list(variants.keys())
    for i, v1 in enumerate(variant_names):
        for v2 in variant_names[i+1:]:
            key = f"{v1} vs {v2}"
            overlaps[key] = [
                compute_overlap(eval_dfs[v1], eval_dfs[v2], k)
                for k in [25, 50, 100]
            ]

    # Generate reports
    print("\nGenerating reports...")

    # Ensure output directory exists
    output_dir = Path("data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Markdown report
    md_report = generate_markdown_report(variants, overlaps, eval_years)
    md_path = output_dir / "spread_selection_topk_comparison.md"
    md_path.write_text(md_report)
    print(f"  Markdown report: {md_path}")

    # JSON report
    json_data = {
        "eval_years": eval_years,
        "variants": {},
        "overlaps": overlaps,
    }

    for name, v in variants.items():
        json_data["variants"][name] = {
            "description": v.description,
            "brier_score": v.brier_score,
            "n_eval_games": v.n_eval_games,
            "fold_diagnostics": [asdict(f) for f in v.fold_diagnostics],
            "topk_by_season": {
                k: [asdict(r) for r in results]
                for k, results in v.topk_by_season.items()
            },
            "threshold_by_season": {
                k: [asdict(r) for r in results]
                for k, results in v.threshold_by_season.items()
            },
        }

    json_path = output_dir / "spread_selection_topk_comparison.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  JSON report: {json_path}")

    # Print summary to console
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nTop-50 by EV (Pooled):")
    print(f"{'Variant':<15} | {'W-L-P':>10} | {'ATS%':>7} | {'ROI%':>8} | {'Avg EV':>8}")
    print("-" * 60)

    for name, v in variants.items():
        result = [r for r in v.topk_by_season["pooled"] if r.k == 50][0]
        wlp = f"{result.wins}-{result.losses}-{result.pushes}"
        print(f"{name:<15} | {wlp:>10} | {result.ats_pct:>6.1f}% | {result.roi_pct:>+7.1f}% | {result.avg_ev:>8.3f}")

    print("\n" + "=" * 80)
    print(f"Full reports saved to {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
