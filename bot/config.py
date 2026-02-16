"""Bot configuration — loads secrets from .env and defines constants."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Discord secrets
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
DISCORD_GUILD_ID = int(os.getenv("DISCORD_GUILD_ID", "0"))

# Per-command channel routing
# #weekly-spread-bets, #weekly-totals-bets, #weekly-moneyline-sprinkles,
# #regular-season-wins, #weekly-power-ratings
CHANNEL_SPREAD = int(os.getenv("CHANNEL_SPREAD", "0"))
CHANNEL_TOTALS = int(os.getenv("CHANNEL_TOTALS", "0"))
CHANNEL_MONEYLINE = int(os.getenv("CHANNEL_MONEYLINE", "0"))
CHANNEL_WIN_TOTALS = int(os.getenv("CHANNEL_WIN_TOTALS", "0"))
CHANNEL_RATINGS = int(os.getenv("CHANNEL_RATINGS", "0"))

# Fallback default channel (pipeline status posts go here)
DEFAULT_CHANNEL_ID = int(os.getenv("DEFAULT_CHANNEL_ID", "0")) or CHANNEL_SPREAD

# Owner mention for auto-notifications
OWNER_ID = int(os.getenv("DISCORD_OWNER_ID", "389424176387325965"))

# Pipeline timing
PIPELINE_SCHEDULE_HOUR = 10  # 10 AM ET on Sundays
PIPELINE_CHECK_INTERVAL_MINUTES = 5

# Timeouts (seconds)
DISPLAY_SCRIPT_TIMEOUT = 60
PIPELINE_STEP_TIMEOUT = 300  # 5 min per step
PIPELINE_TOTAL_TIMEOUT = 900  # 15 min total

# State file
STATE_FILE = PROJECT_ROOT / "data" / "bot_state.json"
