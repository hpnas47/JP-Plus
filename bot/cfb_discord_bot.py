#!/usr/bin/env python3
"""CFB Power Ratings Discord Bot.

Slash commands for display scripts, pipeline orchestration, and status.
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import discord
from discord import app_commands

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.config import (
    CHANNEL_MONEYLINE,
    CHANNEL_RATINGS,
    CHANNEL_SPREAD,
    CHANNEL_TOTALS,
    CHANNEL_WIN_TOTALS,
    DEFAULT_CHANNEL_ID,
    DISCORD_BOT_TOKEN,
    DISCORD_GUILD_ID,
    PIPELINE_CHECK_INTERVAL_MINUTES,
    PIPELINE_SCHEDULE_HOUR,
)
from bot.formatters import format_pipeline_results, send_long_message, send_to_channel
from bot.state_manager import get_season_record, get_status, pipeline_completed
from bot.sunday_pipeline import run_pipeline
from bot.task_runner import PYTHON, run_display_script

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("cfb_bot")

ET = ZoneInfo("America/New_York")

# ── Bot setup ──────────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True

bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)

guild_obj = discord.Object(id=DISCORD_GUILD_ID) if DISCORD_GUILD_ID else None


# ── Display commands ───────────────────────────────────────────────────

@tree.command(name="spread-bets", description="Show spread betting recommendations", guild=guild_obj)
@app_commands.describe(year="Season year", week="Week number")
async def spread_bets(interaction: discord.Interaction, year: int, week: int):
    await interaction.response.defer()
    result = run_display_script("show_spread_bets.py", [str(year), str(week)])
    output = result.stdout if result.returncode == 0 else f"Error:\n{result.stderr}"
    await send_long_message(interaction, output, f"spread_bets_{year}_w{week}.md")


@tree.command(name="totals-bets", description="Show totals (O/U) betting recommendations", guild=guild_obj)
@app_commands.describe(year="Season year", week="Week number")
async def totals_bets(interaction: discord.Interaction, year: int, week: int):
    await interaction.response.defer()
    result = run_display_script("show_totals_bets.py", [str(year), str(week)])
    output = result.stdout if result.returncode == 0 else f"Error:\n{result.stderr}"
    await send_long_message(interaction, output, f"totals_bets_{year}_w{week}.md")


@tree.command(name="moneyline-bets", description="Show moneyline betting recommendations", guild=guild_obj)
@app_commands.describe(year="Season year", week="Week number")
async def moneyline_bets(interaction: discord.Interaction, year: int, week: int):
    await interaction.response.defer()
    result = run_display_script("show_moneyline_bets.py", [str(year), str(week)])
    output = result.stdout if result.returncode == 0 else f"Error:\n{result.stderr}"
    await send_long_message(interaction, output, f"moneyline_bets_{year}_w{week}.md")


@tree.command(name="win-totals", description="Show preseason win total projections", guild=guild_obj)
@app_commands.describe(year="Season year", top_n="Number of teams (default 25, or 'all')", conference="Conference filter")
async def win_totals(
    interaction: discord.Interaction,
    year: int,
    top_n: str = "25",
    conference: str | None = None,
):
    await interaction.response.defer()
    args = [str(year), top_n]
    if conference:
        args.append(conference)
    result = run_display_script("show_win_totals.py", args)
    output = result.stdout if result.returncode == 0 else f"Error:\n{result.stderr}"
    await send_long_message(interaction, output, f"win_totals_{year}.md")


@tree.command(name="ratings", description="Show JP+ power ratings", guild=guild_obj)
@app_commands.describe(year="Season year", top_n="Number of teams (default 25)", week="Through week N (omit for full season)", refresh="Recompute from API (slow, ~60s)")
async def ratings(interaction: discord.Interaction, year: int, top_n: int = 25, week: int | None = None, refresh: bool = False):
    await interaction.response.defer()
    args = [PYTHON, "scripts/show_ratings.py", str(year), str(top_n)]
    if week is not None:
        args.extend(["--week", str(week)])
    if refresh:
        args.append("--refresh")
    from bot.task_runner import run_command_async
    result = await run_command_async(args, timeout=180)
    output = result.stdout if result.returncode == 0 else f"Error:\n{result.stderr}"
    await send_long_message(interaction, output, f"ratings_{year}.md")


# ── Pipeline commands ──────────────────────────────────────────────────

@tree.command(name="run-pipeline", description="Run full Sunday pipeline (odds + spreads + totals + ML)", guild=guild_obj)
@app_commands.describe(year="Season year (auto-detect if omitted)", week="Week number (auto-detect if omitted)", force="Re-run even if already completed this week")
async def run_pipeline_cmd(
    interaction: discord.Interaction,
    year: int | None = None,
    week: int | None = None,
    force: bool = False,
):
    await interaction.response.defer()

    # Auto-detect year/week if not provided
    if year is None or week is None:
        try:
            from src.api.cfbd_client import CFBDClient
            client = CFBDClient()
            cal = client.get_calendar(year=year or datetime.now(ET).year)
            now = datetime.now(ET)
            detected_year = year or now.year
            detected_week = 1
            for entry in cal:
                start = datetime.fromisoformat(str(entry.first_game_start).replace("Z", "+00:00"))
                if now >= start:
                    detected_week = entry.week
            year = detected_year
            week = detected_week
        except Exception as e:
            await interaction.followup.send(f"Could not auto-detect week: {e}\nPlease provide year and week.")
            return

    await interaction.followup.send(f"Starting pipeline for **{year} Week {week}**... (this may take 5-10 minutes)")

    # Run in background task so we don't block
    async def _run_and_report():
        results = await run_pipeline(year, week, force=force)
        summary = format_pipeline_results(results)

        # Post summary to the channel where command was issued
        channel = bot.get_channel(interaction.channel_id)
        if channel:
            await channel.send(f"**Pipeline complete — {year} Week {week}**\n\n{summary}")

        # Auto-post each display to its designated channel
        await _post_to_channels(year, week)

    asyncio.create_task(_run_and_report())


async def _post_to_channels(year: int, week: int):
    """After pipeline completes, post each display output to its designated channel."""
    channel_map = [
        (CHANNEL_SPREAD, "show_spread_bets.py", [str(year), str(week)], f"spread_bets_{year}_w{week}.md"),
        (CHANNEL_TOTALS, "show_totals_bets.py", [str(year), str(week)], f"totals_bets_{year}_w{week}.md"),
        (CHANNEL_MONEYLINE, "show_moneyline_bets.py", [str(year), str(week)], f"moneyline_bets_{year}_w{week}.md"),
    ]
    for channel_id, script, args, filename in channel_map:
        if not channel_id:
            continue
        channel = bot.get_channel(channel_id)
        if not channel:
            continue
        try:
            result = run_display_script(script, args)
            output = result.stdout if result.returncode == 0 else f"Error:\n{result.stderr}"
            if not output.strip():
                continue
            await send_to_channel(channel, output, filename)
        except Exception as e:
            logger.error(f"Failed to post to {channel_id}: {e}")


@tree.command(name="capture-odds", description="Capture opening or closing odds", guild=guild_obj)
@app_commands.describe(timing="'opening' or 'closing'", year="Season year", week="Week number")
@app_commands.choices(timing=[
    app_commands.Choice(name="opening", value="opening"),
    app_commands.Choice(name="closing", value="closing"),
])
async def capture_odds(interaction: discord.Interaction, timing: str, year: int, week: int):
    await interaction.response.defer()
    from bot.task_runner import run_command_async
    result = await run_command_async(
        [PYTHON, "scripts/weekly_odds_capture.py", f"--{timing}", "--year", str(year), "--week", str(week)],
        timeout=120,
    )
    output = result.stdout if result.returncode == 0 else f"Error:\n{result.stderr}"
    await send_long_message(interaction, output[:1900], f"odds_{timing}_{year}_w{week}.txt")


@tree.command(name="settle-bets", description="Settle spread and ML bets for a completed week", guild=guild_obj)
@app_commands.describe(year="Season year", week="Week number")
async def settle_bets(interaction: discord.Interaction, year: int, week: int):
    await interaction.response.defer()
    from bot.task_runner import run_command_async
    # Settle spreads
    r1 = await run_command_async(
        [PYTHON, "scripts/run_spread_weekly.py", "--settle", "--year", str(year), "--week", str(week)],
        timeout=120,
    )
    # Settle moneylines
    r2 = await run_command_async(
        [PYTHON, "scripts/run_moneyline_weekly.py", "--settle", "--year", str(year), "--week", str(week)],
        timeout=120,
    )
    lines = [f"**Settle {year} Week {week}:**\n"]
    lines.append(f"Spreads: {'✅' if r1.returncode == 0 else '❌'} ({r1.duration:.1f}s)")
    if r1.returncode != 0:
        lines.append(f"```{r1.stderr[:300]}```")
    lines.append(f"Moneylines: {'✅' if r2.returncode == 0 else '❌'} ({r2.duration:.1f}s)")
    if r2.returncode != 0:
        lines.append(f"```{r2.stderr[:300]}```")
    await interaction.followup.send("\n".join(lines))


# ── Status commands ────────────────────────────────────────────────────

@tree.command(name="status", description="Show bot and pipeline status", guild=guild_obj)
async def status(interaction: discord.Interaction):
    state = get_status()
    last = state.get("last_pipeline", {})
    errors = state.get("recent_errors", [])

    lines = ["**JP+ Bot Status**\n"]
    if last:
        lines.append(f"Last pipeline: **{last.get('year')} Week {last.get('week')}** at {last.get('timestamp', '?')}")
    else:
        lines.append("No pipeline runs recorded yet.")

    if errors:
        lines.append(f"\nRecent errors ({len(errors)}):")
        for e in errors[:3]:
            lines.append(f"  • `{e['error'][:100]}` ({e['timestamp'][:16]})")

    await interaction.response.send_message("\n".join(lines))


@tree.command(name="record", description="Show season betting record", guild=guild_obj)
@app_commands.describe(year="Season year")
async def record(interaction: discord.Interaction, year: int):
    records = get_season_record(year)
    if not records:
        await interaction.response.send_message(f"No betting records found for {year}.")
        return

    lines = [f"**{year} Season Record**\n"]
    for market, info in records.items():
        w, l = info["wins"], info["losses"]
        pct = w / (w + l) * 100 if (w + l) > 0 else 0
        lines.append(f"**{market.title()}:** {w}-{l} ({pct:.1f}%)")

    await interaction.response.send_message("\n".join(lines))


@tree.command(name="next-week", description="Show the next CFB week to process", guild=guild_obj)
async def next_week(interaction: discord.Interaction):
    try:
        from src.api.cfbd_client import CFBDClient
        client = CFBDClient()
        now = datetime.now(ET)
        cal = client.get_calendar(year=now.year)
        current_week = 1
        for entry in cal:
            start = datetime.fromisoformat(str(entry.first_game_start).replace("Z", "+00:00"))
            if now >= start:
                current_week = entry.week
        await interaction.response.send_message(f"Current: **{now.year} Week {current_week}**\nNext to process: **Week {current_week + 1}**")
    except Exception as e:
        await interaction.response.send_message(f"Error detecting week: {e}")


# ── Auto-scheduler ─────────────────────────────────────────────────────

async def auto_schedule_loop():
    """Check every N minutes if it's Sunday 10 AM ET and auto-trigger pipeline."""
    await bot.wait_until_ready()
    logger.info("Auto-scheduler started")

    while not bot.is_closed():
        try:
            now = datetime.now(ET)
            if (
                now.weekday() == 6  # Sunday
                and now.hour == PIPELINE_SCHEDULE_HOUR
                and now.minute < PIPELINE_CHECK_INTERVAL_MINUTES
            ):
                year = now.year
                # Auto-detect week
                from src.api.cfbd_client import CFBDClient
                client = CFBDClient()
                cal = client.get_calendar(year=year)
                week = 1
                for entry in cal:
                    start = datetime.fromisoformat(str(entry.first_game_start).replace("Z", "+00:00"))
                    if now >= start:
                        week = entry.week

                if not pipeline_completed(year, week):
                    logger.info(f"Auto-triggering pipeline for {year} Week {week}")
                    results = await run_pipeline(year, week)
                    summary = format_pipeline_results(results)

                    channel = bot.get_channel(DEFAULT_CHANNEL_ID)
                    if channel:
                        await channel.send(
                            f"**Auto Pipeline — {year} Week {week}**\n\n{summary}"
                        )

                    # Post displays to designated channels
                    await _post_to_channels(year, week)
        except Exception as e:
            logger.error(f"Auto-scheduler error: {e}")

        await asyncio.sleep(PIPELINE_CHECK_INTERVAL_MINUTES * 60)


# ── Bot events ─────────────────────────────────────────────────────────

@bot.event
async def on_ready():
    if guild_obj:
        tree.copy_global_to(guild=guild_obj)
        await tree.sync(guild=guild_obj)
    else:
        await tree.sync()
    logger.info(f"Bot ready as {bot.user} — synced {len(tree.get_commands(guild=guild_obj))} commands")
    bot.loop.create_task(auto_schedule_loop())


# ── Main ───────────────────────────────────────────────────────────────

def main():
    if not DISCORD_BOT_TOKEN:
        print("Error: DISCORD_BOT_TOKEN not set in .env")
        sys.exit(1)
    bot.run(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    main()
