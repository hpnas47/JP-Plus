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

    # Compute column widths
    widths = {i: 0 for i in kept}
    for row in rows:
        for i in kept:
            cell = row[i] if i < len(row) else ""
            widths[i] = max(widths[i], len(cell))

    # Build output
    lines: list[str] = []
    for row_idx, row in enumerate(rows):
        parts: list[str] = []
        for i in kept:
            cell = row[i] if i < len(row) else ""
            parts.append(cell.ljust(widths[i]))
        line = " ".join(parts).rstrip()
        lines.append(line)

        if row_idx == 0:
            sep_parts = ["─" * widths[i] for i in kept]
            lines.append(" ".join(sep_parts))

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
