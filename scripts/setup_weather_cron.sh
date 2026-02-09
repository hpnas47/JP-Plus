#!/bin/bash
# Setup cron job for Thursday weather capture
#
# This installs a cron job that runs every Thursday at 6:00 AM local time
# during CFB season (August - January).
#
# Usage:
#   ./scripts/setup_weather_cron.sh install   # Install the cron job
#   ./scripts/setup_weather_cron.sh remove    # Remove the cron job
#   ./scripts/setup_weather_cron.sh status    # Check if installed

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_PATH="/usr/bin/python3"
SCRIPT_PATH="$PROJECT_DIR/scripts/weather_thursday_capture.py"
LOG_DIR="$PROJECT_DIR/logs"
CRON_COMMENT="# JP+ Thursday Weather Capture"

# Ensure logs directory exists
mkdir -p "$LOG_DIR"

install_cron() {
    echo "Installing Thursday weather capture cron job..."

    # Check if already installed
    if crontab -l 2>/dev/null | grep -q "weather_thursday_capture"; then
        echo "Cron job already installed. Use 'remove' first to reinstall."
        exit 1
    fi

    # Create the cron entry
    # Runs every Thursday (day 4) at 6:00 AM
    CRON_ENTRY="0 6 * * 4 cd $PROJECT_DIR && $PYTHON_PATH $SCRIPT_PATH >> $LOG_DIR/weather_cron.log 2>&1 $CRON_COMMENT"

    # Add to crontab
    (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

    echo "✅ Cron job installed!"
    echo ""
    echo "Schedule: Every Thursday at 6:00 AM"
    echo "Script: $SCRIPT_PATH"
    echo "Log: $LOG_DIR/weather_cron.log"
    echo ""
    echo "To test manually:"
    echo "  python3 $SCRIPT_PATH --dry-run"
}

remove_cron() {
    echo "Removing Thursday weather capture cron job..."

    # Remove the line from crontab
    crontab -l 2>/dev/null | grep -v "weather_thursday_capture" | crontab -

    echo "✅ Cron job removed!"
}

status_cron() {
    echo "Checking cron job status..."
    echo ""

    if crontab -l 2>/dev/null | grep -q "weather_thursday_capture"; then
        echo "✅ Cron job is INSTALLED:"
        echo ""
        crontab -l | grep "weather_thursday"
    else
        echo "❌ Cron job is NOT installed."
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
        echo "  install  - Install the Thursday 6 AM cron job"
        echo "  remove   - Remove the cron job"
        echo "  status   - Check if cron job is installed"
        exit 1
        ;;
esac
