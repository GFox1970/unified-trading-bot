
#!/bin/bash
# Render Start Script for Unified Trading Bot
# ===========================================
# This script is used by Render to start the unified trading bot service

set -e  # Exit on any error

echo "Starting Unified Trading Bot on Render..."

# Set environment variables for production
export PYTHONPATH="${PYTHONPATH}:/opt/render/project/src"
export TRADING_BOT_LOG_LEVEL="${LOG_LEVEL:-INFO}"
export TRADING_BOT_BOT_UPDATE_INTERVAL="${UPDATE_INTERVAL:-60}"

# Set trading mode based on environment variable
if [ "${LIVE_TRADING}" = "true" ]; then
    echo "Starting in LIVE TRADING mode"
    TRADING_MODE="--live"
else
    echo "Starting in DRY RUN mode (default)"
    TRADING_MODE="--dry-run"
fi

# Set signal strength for testing/production
if [ "${TEST_MODE}" = "true" ]; then
    echo "Test mode enabled - using relaxed signal thresholds"
    export TRADING_BOT_SIGNALS_MIN_STRENGTH="0.3"
    export TRADING_BOT_RISK_MAX_RISK_SCORE="0.8"
else
    echo "Production mode - using standard signal thresholds"
    export TRADING_BOT_SIGNALS_MIN_STRENGTH="0.6"
    export TRADING_BOT_RISK_MAX_RISK_SCORE="0.6"
fi

# Create logs directory
mkdir -p logs

# Start the unified trading bot
echo "Executing: python unified_trading_bot.py ${TRADING_MODE} --log-level ${TRADING_BOT_LOG_LEVEL}"
exec python unified_trading_bot.py ${TRADING_MODE} --log-level "${TRADING_BOT_LOG_LEVEL}"
