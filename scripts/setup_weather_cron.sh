#!/bin/bash
# Setup cron jobs for weather capture (Thursday + Saturday)
#
# This installs cron jobs for the two-stage weather capture workflow:
# 1. Thursday 6:00 AM local time - Early capture (72h out, lower confidence)
# 2. Saturday 8:00 AM ET - Confirmation run (6-12h out, high confidence)
#
# Usage:
#   ./scripts/setup_weather_cron.sh install   # Install both cron jobs
#   ./scripts/setup_weather_cron.sh remove    # Remove all cron jobs
#   ./scripts/setup_weather_cron.sh status    # Check if installed

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_PATH="/usr/bin/python3"
SCRIPT_PATH="$PROJECT_DIR/scripts/weather_thursday_capture.py"
LOG_DIR="$PROJECT_DIR/logs"

# Ensure logs directory exists
mkdir -p "$LOG_DIR"

install_cron() {
    echo "Installing weather capture cron jobs..."

    # Check if already installed
    if crontab -l 2>/dev/null | grep -q "weather_thursday_capture"; then
        echo "Cron jobs already installed. Use 'remove' first to reinstall."
        exit 1
    fi

    # Thursday entry: Every Thursday (day 4) at 6:00 AM local time
    THURSDAY_ENTRY="0 6 * * 4 cd $PROJECT_DIR && $PYTHON_PATH $SCRIPT_PATH >> $LOG_DIR/weather_thursday.log 2>&1 # JP+ Thursday Weather"

    # Saturday entry: Every Saturday (day 6) at 8:00 AM ET (adjust for your timezone)
    # Using --saturday flag for confirmation mode
    SATURDAY_ENTRY="0 8 * * 6 cd $PROJECT_DIR && $PYTHON_PATH $SCRIPT_PATH --saturday >> $LOG_DIR/weather_saturday.log 2>&1 # JP+ Saturday Confirmation"

    # Add both to crontab
    (crontab -l 2>/dev/null; echo "$THURSDAY_ENTRY"; echo "$SATURDAY_ENTRY") | crontab -

    echo "✅ Cron jobs installed!"
    echo ""
    echo "Schedule:"
    echo "  📅 Thursday 6:00 AM - Early capture (72h out)"
    echo "  📅 Saturday 8:00 AM - Confirmation run (6-12h out)"
    echo ""
    echo "Logs:"
    echo "  $LOG_DIR/weather_thursday.log"
    echo "  $LOG_DIR/weather_saturday.log"
    echo ""
    echo "To test manually:"
    echo "  python3 $SCRIPT_PATH --dry-run           # Thursday mode"
    echo "  python3 $SCRIPT_PATH --saturday --dry-run # Saturday mode"
}

remove_cron() {
    echo "Removing weather capture cron jobs..."

    # Remove all weather-related cron entries
    crontab -l 2>/dev/null | grep -v "weather_thursday_capture" | crontab -

    echo "✅ Cron jobs removed!"
}

status_cron() {
    echo "Checking cron job status..."
    echo ""

    if crontab -l 2>/dev/null | grep -q "weather_thursday_capture"; then
        echo "✅ Cron jobs are INSTALLED:"
        echo ""
        crontab -l | grep "JP+ Thursday\|JP+ Saturday"
    else
        echo "❌ Cron jobs are NOT installed."
        echo ""
        echo "Run './scripts/setup_weather_cron.sh install' to install."
    fi
}

case "$1" in
    install)
        install_cron
        ;;
    remove)
        remove_cron
        ;;
    status)
        status_cron
        ;;
    *)
        echo "Usage: $0 {install|remove|status}"
        echo ""
        echo "Commands:"
        echo "  install  - Install Thursday + Saturday cron jobs"
        echo "  remove   - Remove all weather cron jobs"
        echo "  status   - Check if cron jobs are installed"
        echo ""
        echo "Two-Stage Workflow:"
        echo "  1. Thursday 6 AM - Early capture, identify weather concerns"
        echo "  2. Saturday 8 AM - Confirmation with accurate 6-12h forecasts"
        exit 1
        ;;
esac
