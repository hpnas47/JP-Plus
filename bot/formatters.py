"""Discord message formatting utilities."""

from __future__ import annotations

import io
import re
from typing import TYPE_CHECKING

import discord

if TYPE_CHECKING:
    from discord import Interaction


def _md_table_to_fixed(text: str) -> str:
    """Convert markdown pipe tables to aligned fixed-width text for Discord code blocks.

    Also converts markdown headers (## Title) to plain decorated headers.
    """
    output_lines: list[str] = []

    for block in _split_into_blocks(text):
        if isinstance(block, list):
            # It's a table block (list of row-lists)
            output_lines.extend(_format_table(block))
        else:
            # Non-table line — clean up markdown
            line = block
            # Convert ## headers to prominent decorated text
            m = re.match(r'^#{1,3}\s+(.+)$', line)
            if m:
                title = m.group(1)
                bar = "━" * len(title)
                line = f"\n{'━' * (len(title) + 6)}\n━━ {title} ━━\n{'━' * (len(title) + 6)}"
            # Convert **bold** to CAPS-ish (just strip the **)
            line = re.sub(r'\*\*(.+?)\*\*', r'\1', line)
            # Strip italic markers
            line = re.sub(r'\*(.+?)\*', r'\1', line)
            # Skip footnote separator and lines after it
            if line.strip() == '---':
                # Drop everything from --- onward
                break
            output_lines.append(line)

    return "\n".join(output_lines)


def _split_into_blocks(text: str) -> list:
    """Split text into table blocks (list[list[str]]) and non-table strings."""
    lines = text.split("\n")
    blocks: list = []
    current_table: list[list[str]] = []

    for line in lines:
        if line.startswith("|") and "|" in line[1:]:
            # Parse cells
            cells = [c.strip() for c in line.split("|")]
            # Remove empty first/last from leading/trailing |
            if cells and cells[0] == "":
                cells = cells[1:]
            if cells and cells[-1] == "":
                cells = cells[:-1]

            # Skip separator rows (|---|---|)
            if all(re.match(r'^[-:]+$', c) for c in cells):
                continue

            current_table.append(cells)
        else:
            if current_table:
                blocks.append(current_table)
                current_table = []
            blocks.append(line)

    if current_table:
        blocks.append(current_table)

    return blocks


def _abbreviate_teams(cell: str) -> str:
    """Replace full team names with abbreviations in a cell value."""
    from src.utils.display_helpers import ABBREV
    # Sort by length descending so "Oklahoma State" matches before "Oklahoma"
    for full, abbr in sorted(ABBREV.items(), key=lambda x: -len(x[0])):
        if full in cell:
            cell = cell.replace(full, abbr)
    return cell


def _format_table(rows: list[list[str]]) -> list[str]:
    """Format parsed table rows into aligned fixed-width lines.

    Uses team abbreviations and drops low-value columns to fit
    Discord's code block width.
    """
    if not rows:
        return []

    n_cols = max(len(r) for r in rows)
    header = [rows[0][i] if i < len(rows[0]) else "" for i in range(n_cols)]
    header_lower = [h.lower() for h in header]

    # Drop columns that add clutter for Discord
    drop_cols: set[int] = set()
    drop_names = {"date", "score", "final", "#"}
    for i, h in enumerate(header_lower):
        if h in drop_names:
            drop_cols.add(i)

    kept = [i for i in range(n_cols) if i not in drop_cols]

    # Keep full team names — Discord desktop has enough width

    # Compute column widths (add 1 char padding for breathing room)
    widths = {i: 0 for i in kept}
    for row in rows:
        for i in kept:
            cell = row[i] if i < len(row) else ""
            widths[i] = max(widths[i], len(cell))
    for i in kept:
        widths[i] += 1  # extra padding per column

    SEP = "  "  # 2-space column separator for readability

    # Build output
    lines: list[str] = []
    for row_idx, row in enumerate(rows):
        parts: list[str] = []
        for i in kept:
            cell = row[i] if i < len(row) else ""
            parts.append(cell.ljust(widths[i]))
        line = SEP.join(parts).rstrip()
        lines.append(line)

        if row_idx == 0:
            sep_parts = ["─" * widths[i] for i in kept]
            lines.append(SEP.join(sep_parts))

    return lines


def split_for_discord(text: str, max_len: int = 1990) -> list[str]:
    """Split formatted text into chunks that fit Discord's 2000-char limit.

    Preserves table headers by repeating them at the start of each chunk.
    """
    if len(text) <= max_len:
        return [text]

    lines = text.split("\n")

    # Detect table header (first content line + separator)
    header_lines: list[str] = []
    for i, line in enumerate(lines):
        if i < 2:
            header_lines.append(line)
        elif "─" in line and i == len(header_lines):
            header_lines.append(line)
            break
        else:
            break

    chunks: list[str] = []
    current_lines: list[str] = []
    current_len = 0

    # Track if we're in a section that has its own header
    section_header: list[str] = []

    for line in lines:
        # Detect section headers (━━━ ... ━━━)
        if "━━━" in line:
            section_header = []  # reset — new section coming

        line_len = len(line) + 1
        if current_len + line_len > max_len and current_lines:
            chunks.append("\n".join(current_lines))
            current_lines = list(section_header) if section_header else []
            current_len = sum(len(h) + 1 for h in current_lines)

        current_lines.append(line)
        current_len += line_len

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


async def send_long_message(
    interaction: Interaction,
    content: str,
    filename: str = "output.md",
) -> None:
    """Send content that might exceed Discord's message limit.

    Converts markdown tables to fixed-width for readable Discord display.
    """
    formatted = _md_table_to_fixed(content)

    if len(formatted) <= 1990:
        await interaction.followup.send(f"```\n{formatted}\n```")
        return

    if len(formatted) <= 8000:
        chunks = split_for_discord(formatted)
        for chunk in chunks:
            await interaction.followup.send(f"```\n{chunk}\n```")
        return

    # Too long — send as file
    file = discord.File(
        io.BytesIO(formatted.encode("utf-8")),
        filename=filename,
    )
    await interaction.followup.send("Output too long for chat, attached as file:", file=file)


async def send_to_channel(channel, content: str, filename: str = "output.md") -> None:
    """Send formatted content directly to a channel (for auto-posts)."""
    formatted = _md_table_to_fixed(content)

    if len(formatted) <= 1990:
        await channel.send(f"```\n{formatted}\n```")
    elif len(formatted) <= 8000:
        for chunk in split_for_discord(formatted):
            await channel.send(f"```\n{chunk}\n```")
    else:
        file = discord.File(
            io.BytesIO(formatted.encode("utf-8")),
            filename=filename,
        )
        await channel.send(f"Results attached:", file=file)


# ── Embed formatting for bet recommendations ─────────────────────────

# Color palette for embed sidebars
EMBED_COLORS = {
    "spread_primary": 0x2ECC71,   # Green
    "spread_edge": 0x3498DB,      # Blue
    "totals_primary": 0xF39C12,   # Orange
    "moneyline_action": 0x2ECC71, # Green
    "moneyline_watchlist": 0xF1C40F,  # Yellow
}


def parse_bet_sections(text: str) -> list[dict]:
    """Parse markdown output into sections with title, header, rows, and record.

    Returns list of dicts: {title, headers, rows: list[dict], record}
    """
    sections: list[dict] = []
    current_section: dict | None = None

    for block in _split_into_blocks(text):
        if isinstance(block, str):
            # Check for section header
            m = re.match(r'^#{1,3}\s+(.+)$', block)
            if m:
                if current_section:
                    sections.append(current_section)
                current_section = {"title": m.group(1), "headers": [], "rows": [], "record": ""}
                continue

            # Check for record line
            rec = re.match(r'^\*\*(.+)\*\*$', block.strip())
            if rec and current_section:
                # Append record lines (there may be multiple)
                if current_section["record"]:
                    current_section["record"] += "\n" + rec.group(1)
                else:
                    current_section["record"] = rec.group(1)
                continue

            # Stop at footnote separator
            if block.strip() == '---':
                break

        elif isinstance(block, list) and current_section is not None:
            # Table block — first row is headers, rest are data
            if len(block) < 1:
                continue
            current_section["headers"] = block[0]
            for row_cells in block[1:]:
                row_dict = {}
                for i, h in enumerate(block[0]):
                    row_dict[h.strip()] = row_cells[i].strip() if i < len(row_cells) else ""
                current_section["rows"].append(row_dict)

    if current_section:
        sections.append(current_section)

    return sections


def _detect_section_color(title: str, bet_type: str) -> int:
    """Pick embed color based on section title and bet type."""
    title_lower = title.lower()
    if bet_type == "spread":
        if "primary" in title_lower or "ev engine" in title_lower:
            return EMBED_COLORS["spread_primary"]
        return EMBED_COLORS["spread_edge"]
    elif bet_type == "totals":
        return EMBED_COLORS["totals_primary"]
    elif bet_type == "moneyline":
        if "watchlist" in title_lower:
            return EMBED_COLORS["moneyline_watchlist"]
        return EMBED_COLORS["moneyline_action"]
    return 0x95A5A6  # Gray fallback


def _build_spread_field(row: dict, has_ev: bool) -> tuple[str, str]:
    """Build embed field name/value for a spread bet row."""
    matchup = row.get("Matchup", "")
    bet = row.get("Bet (Open)", "")
    edge = row.get("Edge", "")
    jp_line = row.get("JP+ Line", "")
    ev = row.get("~EV", "")
    result = row.get("Result", "")

    name = matchup if matchup else "—"
    parts = [f"Pick: {bet}", f"Edge: {edge}"]
    if has_ev and ev:
        parts.append(f"EV: {ev}")
    parts.append(f"JP+: {_abbreviate_teams(jp_line)}")
    if result and result != "—":
        parts.append(f"Result: {result}")

    return name, " | ".join(parts)


def _build_totals_field(row: dict) -> tuple[str, str]:
    """Build embed field name/value for a totals bet row."""
    matchup = row.get("Matchup", "")
    side = row.get("Side", "")
    edge = row.get("Edge", "")
    vegas = row.get("Vegas (Open)", "")
    jp_total = row.get("JP+ Total", "")
    ev = row.get("~EV", "")
    result = row.get("Result", "")

    name = matchup if matchup else "—"
    parts = [f"Pick: {side} {vegas}"]
    parts.append(f"Edge: {edge}")
    if jp_total:
        parts.append(f"JP+: {jp_total}")
    if ev:
        parts.append(f"EV: {ev}")
    if result and result != "—":
        parts.append(f"Result: {result}")

    return name, " | ".join(parts)


def _build_moneyline_field(row: dict) -> tuple[str, str]:
    """Build embed field name/value for a moneyline bet row."""
    matchup = row.get("Matchup", "")
    bet = row.get("Bet", row.get("Side", ""))
    odds = row.get("Odds", "")
    ev = row.get("EV", "")
    conf = row.get("Conf", "")
    win_prob = row.get("Win Prob", "")
    result = row.get("Result", "")

    name = matchup if matchup else "—"
    parts = [f"Pick: {bet} ({odds})"]
    if win_prob:
        parts.append(f"Win Prob: {win_prob}")
    if ev:
        parts.append(f"EV: {ev}")
    if conf:
        parts.append(f"Conf: {conf}")
    if result and result != "—":
        parts.append(f"Result: {result}")

    return name, " | ".join(parts)


def section_to_embed(section: dict, color: int, bet_type: str) -> discord.Embed:
    """Convert a parsed section to a Discord embed."""
    embed = discord.Embed(title=section["title"], color=color)

    has_ev = any("~EV" in h or "EV" in h for h in section.get("headers", []) if "~EV" in h)

    for row in section["rows"][:25]:  # Discord max 25 fields
        if bet_type == "spread":
            name, value = _build_spread_field(row, has_ev="~EV" in section.get("headers", []))
        elif bet_type == "totals":
            name, value = _build_totals_field(row)
        elif bet_type == "moneyline":
            name, value = _build_moneyline_field(row)
        else:
            # Generic: combine all values
            name = row.get(section["headers"][1], "") if len(section["headers"]) > 1 else "—"
            value = " | ".join(f"{h}: {row.get(h, '')}" for h in section["headers"][2:] if row.get(h))

        # Discord field limits: name 256 chars, value 1024 chars
        name = (name or "—")[:256]
        value = (value or "—")[:1024]
        embed.add_field(name=name, value=value, inline=False)

    if section.get("record"):
        embed.set_footer(text=section["record"])

    if not section["rows"]:
        embed.description = "*No qualifying bets this section.*"

    return embed


def bets_to_embeds(text: str, bet_type: str) -> list[discord.Embed]:
    """Parse markdown bet output and return list of Discord embeds.

    Args:
        text: Raw markdown output from a display script
        bet_type: "spread", "totals", or "moneyline"

    Returns:
        List of discord.Embed objects (one per section)
    """
    sections = parse_bet_sections(text)
    if not sections:
        return []

    embeds = []
    for section in sections:
        color = _detect_section_color(section["title"], bet_type)
        embeds.append(section_to_embed(section, color, bet_type))

    return embeds


async def send_as_embeds(
    interaction: Interaction,
    content: str,
    bet_type: str,
    filename: str = "output.md",
) -> None:
    """Parse bet output and send as Discord embeds. Falls back to code blocks on failure."""
    try:
        embeds = bets_to_embeds(content, bet_type)
        if embeds:
            for embed in embeds:
                await interaction.followup.send(embed=embed)
            return
    except Exception:
        pass  # Fall back to code block

    await send_long_message(interaction, content, filename)


async def send_embeds_to_channel(
    channel,
    content: str,
    bet_type: str,
    filename: str = "output.md",
) -> None:
    """Parse bet output and send as embeds to a channel. Falls back to code blocks."""
    try:
        embeds = bets_to_embeds(content, bet_type)
        if embeds:
            for embed in embeds:
                await channel.send(embed=embed)
            return
    except Exception:
        pass  # Fall back to code block

    await send_to_channel(channel, content, filename)


def format_pipeline_results(results: dict[str, dict]) -> str:
    """Format pipeline step results into a summary string."""
    lines = ["**Pipeline Results:**\n"]
    for step_name, info in results.items():
        status = info.get("status", "unknown")
        duration = info.get("duration", 0)
        if status == "success":
            icon = "✅"
        elif status == "skipped":
            icon = "⏭️"
        elif status == "error":
            icon = "❌"
        else:
            icon = "⬜"
        line = f"{icon} **{step_name}** — {duration:.1f}s"
        if status == "error":
            err = info.get("error", "")
            if err:
                line += f"\n   `{err[:200]}`"
        lines.append(line)
    return "\n".join(lines)
