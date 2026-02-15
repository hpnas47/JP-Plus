#!/bin/bash
# Start the CFB Discord bot from project root
cd "$(dirname "$0")/.." || exit 1
source .env 2>/dev/null
exec python3 -m bot.cfb_discord_bot
