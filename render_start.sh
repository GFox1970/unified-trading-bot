#!/bin/bash
set -e
echo "Starting Render deployment..."
pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:/opt/render/project/src"
export PYTHONUNBUFFERED=1
mkdir -p logs
mkdir -p data
echo "Starting the trading bot..."
python unified_trading_bot.py
