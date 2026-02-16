#!/usr/bin/env python3
"""Format weather watchlist data into a Discord-ready alert table.

Reads the watchlist from the Thursday weather capture output (JSON)
and formats it for Discord posting.

Usage:
    # Run weather capture first, then format for Discord
    python3 scripts/show_weather_alert.py <year> <week>

    # Or pipe from capture (standalone mode — reads from DB)
    python3 scripts/show_weather_alert.py 2025 12
"""

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)


def format_weather_alert(stats: dict) -> str | None:
    """Format weather watchlist into Discord-ready markdown table.

    Returns None if no games exceed thresholds.
    """
    watchlist = stats.get("watchlist", [])
    if not watchlist:
        return None

    year = stats["year"]
    week = stats["week"]
    n_outdoor = stats.get("outdoor_games", 0)
    n_flagged = len(watchlist)
    capture_time = stats.get("capture_time", "")

    # Parse capture time for display
    try:
        from datetime import datetime
        ct = datetime.fromisoformat(capture_time)
        time_str = ct.strftime("%A %I:%M %p ET")
    except Exception:
        time_str = capture_time[:16] if capture_time else "Unknown"

    lines = []
    lines.append(f"## 🌧️ Weather Alert — {year} Week {week}\n")
    lines.append(f"*{n_outdoor} outdoor games checked, {n_flagged} flagged • {time_str}*\n")

    # Table header
    lines.append("| # | Matchup | 🌬️ Wind | 🌡️ Temp | ⬇️ Wx Adj | 📊 JP+ | 🎰 Vegas | 💰 Edge | Signal |")
    lines.append("|---|---------|---------|---------|----------|--------|---------|---------|--------|")

    # Sort by edge (most negative = strongest UNDER first)
    sorted_wl = sorted(
        watchlist,
        key=lambda x: (
            (0, x["edge"]) if x.get("edge") is not None
            else (1, x.get("weather_adjustment", 0))
        )
    )

    for i, entry in enumerate(sorted_wl, 1):
        matchup = entry["matchup"]
        wind = entry.get("wind_speed", 0)
        gust = entry.get("wind_gust")
        temp = entry.get("temperature", 0)
        wx_adj = entry.get("weather_adjustment", 0)
        jp_total = entry.get("jp_weather_adjusted")
        vegas = entry.get("vegas_total")
        edge = entry.get("edge")
        high_var = entry.get("high_variance", False)

        # Wind display
        if gust and gust > wind + 5:
            wind_str = f"{wind:.0f}g{gust:.0f}"
        else:
            wind_str = f"{wind:.0f} mph"

        # Temp display
        temp_str = f"{temp:.0f}°F"

        # Weather adjustment
        wx_str = f"{wx_adj:+.1f}"

        # JP+ total (weather-adjusted)
        jp_str = f"{jp_total:.1f}" if jp_total is not None else "—"

        # Vegas total
        vegas_str = f"{vegas:.1f}" if vegas is not None else "—"

        # Edge and signal
        if edge is not None:
            edge_str = f"{edge:+.1f}"
            if high_var:
                signal = "⚠️ HIGH VAR"
            elif edge < -5:
                signal = "🚨 BET UNDER"
            elif edge < -3:
                signal = "📉 STRONG UNDER"
            elif edge < 0:
                signal = "📉 Lean Under"
            else:
                signal = "➖ Neutral"
        else:
            edge_str = "—"
            if abs(wx_adj) >= 3:
                signal = "📉 Weather UNDER"
            else:
                signal = "📉 Monitor"

        lines.append(
            f"| {i} | {matchup} | {wind_str} | {temp_str} | {wx_str} | {jp_str} | {vegas_str} | {edge_str} | {signal} |"
        )

    lines.append("")

    # Footer with confidence note
    confidences = [e.get("confidence", 0) for e in watchlist]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    hours = [e.get("hours_until_game", 0) for e in watchlist]
    avg_hours = sum(hours) / len(hours) if hours else 0

    lines.append(f"*Forecast confidence: {avg_conf:.0%} avg ({avg_hours:.0f}h until kickoff)*")
    lines.append("")
    lines.append("💡 **Action:** Consider betting UNDER before the market adjusts.")
    lines.append("Saturday AM confirmation will re-check with higher-confidence forecasts.")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/show_weather_alert.py <year> <week> [--saturday]")
        sys.exit(1)

    year = int(sys.argv[1])
    week = int(sys.argv[2])
    saturday = "--saturday" in sys.argv

    # Check for cached watchlist JSON first
    cache_dir = Path(__file__).parent.parent / "data" / "weather"
    cache_file = cache_dir / f"watchlist_{year}_w{week}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            stats = json.load(f)
    else:
        print(f"No cached watchlist found at {cache_file}", file=sys.stderr)
        print("Run weather_thursday_capture.py first.", file=sys.stderr)
        sys.exit(1)

    output = format_weather_alert(stats)
    if output:
        print(output)
    else:
        print(f"✅ No weather-impacted games for {year} Week {week}.")


if __name__ == "__main__":
    main()
