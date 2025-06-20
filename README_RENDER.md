
# Unified Trading Bot - Render Deployment Guide
===============================================

This guide provides step-by-step instructions for deploying the unified trading bot as a single service on Render, combining both RSI-MACD and options trading strategies.

## 🚀 Quick Start

### Prerequisites
- Render account (free tier available)
- GitHub repository with the trading bot code
- Basic understanding of environment variables

### 1. Repository Setup

1. **Create a new GitHub repository** or use an existing one
2. **Upload all files** from the `trading_bot_fixes` directory to your repository
3. **Ensure the following files are in the root directory:**
   ```
   unified_trading_bot.py
   config.yaml
   requirements.txt
   render_start.sh
   logger.py
   config_manager.py
   market_data.py
   signal_analyzer.py
   options_trader.py
   risk_manager.py
   ```

### 2. Render Service Creation

1. **Log into Render** (https://render.com)
2. **Click "New +"** and select "Web Service"
3. **Connect your GitHub repository**
4. **Configure the service:**
   - **Name:** `unified-trading-bot`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `bash render_start.sh`

### 3. Environment Variables Configuration

In the Render dashboard, add these environment variables:

#### Required Variables
```bash
# Trading Mode
LIVE_TRADING=false          # Set to 'true' for live trading
TEST_MODE=true              # Set to 'false' for production

# Bot Configuration
LOG_LEVEL=INFO
UPDATE_INTERVAL=60          # Seconds between trading cycles

# Signal Thresholds (Relaxed for Testing)
TRADING_BOT_SIGNALS_MIN_STRENGTH=0.3
TRADING_BOT_SIGNALS_MIN_CONFIDENCE=0.5
TRADING_BOT_RISK_MAX_RISK_SCORE=0.8
```

#### Optional Variables (Advanced Configuration)
```bash
# Risk Management
TRADING_BOT_RISK_MAX_POSITION_SIZE=0.05
TRADING_BOT_RISK_MAX_PORTFOLIO_RISK=0.20

# Market Data
TRADING_BOT_MARKET_DATA_PROVIDER=yfinance
TRADING_BOT_MARKET_DATA_RATE_LIMIT=5

# Options Trading
TRADING_BOT_OPTIONS_ENABLED=true
TRADING_BOT_OPTIONS_MIN_VOLUME=50

# Watchlist
TRADING_BOT_TRADING_WATCHLIST=SPY,QQQ,AAPL,MSFT,TSLA
```

### 4. Deploy the Service

1. **Click "Create Web Service"**
2. **Wait for deployment** (usually 2-5 minutes)
3. **Check logs** for successful startup
4. **Monitor the service** through Render dashboard

## 📊 Monitoring and Logs

### Viewing Logs
- **Render Dashboard:** Go to your service → "Logs" tab
- **Real-time monitoring:** Logs update automatically
- **Log levels:** DEBUG, INFO, WARNING, ERROR

### Key Log Messages to Watch For
```
✅ "Unified Trading Bot Started" - Service started successfully
✅ "Trading cycle completed" - Normal operation
✅ "Trade Signal: BUY/SELL" - Signal generated
⚠️  "No data available" - Market data issues
❌ "Error in trading cycle" - Critical errors
```

## 🔧 Configuration Management

### Adjusting Signal Sensitivity

**For Testing (More Signals):**
```bash
TRADING_BOT_SIGNALS_MIN_STRENGTH=0.3
TRADING_BOT_RISK_MAX_RISK_SCORE=0.8
TEST_MODE=true
```

**For Production (Fewer, Higher Quality Signals):**
```bash
TRADING_BOT_SIGNALS_MIN_STRENGTH=0.6
TRADING_BOT_RISK_MAX_RISK_SCORE=0.6
TEST_MODE=false
```

### Modifying Watchlist
Update the watchlist by changing the environment variable:
```bash
TRADING_BOT_TRADING_WATCHLIST=SPY,QQQ,AAPL,MSFT,TSLA,NVDA,GOOGL,AMZN,META,NFLX
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. Service Won't Start
**Symptoms:** Service fails during startup
**Solutions:**
- Check build logs for missing dependencies
- Verify `requirements.txt` is complete
- Ensure `render_start.sh` has execute permissions

#### 2. No Trading Signals Generated
**Symptoms:** Bot runs but no signals appear
**Solutions:**
- Lower signal thresholds temporarily
- Check market hours (bot may pause outside trading hours)
- Verify watchlist symbols are valid

#### 3. Market Data Errors
**Symptoms:** "No data available" errors
**Solutions:**
- Check internet connectivity
- Verify yfinance is working (rate limits)
- Try reducing update frequency

#### 4. High Memory Usage
**Symptoms:** Service restarts due to memory limits
**Solutions:**
- Reduce watchlist size
- Increase cache cleanup frequency
- Consider upgrading Render plan

### Debug Mode
Enable debug logging by setting:
```bash
LOG_LEVEL=DEBUG
```

## 📈 Performance Optimization

### Render Plan Recommendations

**Free Tier (Testing):**
- ✅ Perfect for testing and development
- ✅ Handles 5-10 symbols easily
- ⚠️ May sleep after 15 minutes of inactivity

**Starter Plan ($7/month):**
- ✅ Always-on service
- ✅ Better for production use
- ✅ Handles 20+ symbols

### Optimization Tips

1. **Reduce Update Frequency:** Set `UPDATE_INTERVAL=120` (2 minutes) or higher
2. **Limit Watchlist:** Start with 5-10 symbols
3. **Use Caching:** Enable market data caching (already configured)
4. **Monitor Resources:** Check CPU and memory usage in Render dashboard

## 🔄 Updating the Bot

### Code Updates
1. **Push changes** to your GitHub repository
2. **Render auto-deploys** from the main branch
3. **Monitor logs** during deployment
4. **Verify functionality** after update

### Configuration Updates
1. **Update environment variables** in Render dashboard
2. **Restart service** if needed
3. **Monitor logs** for configuration changes

## 🚨 Safety and Risk Management

### Important Safety Features

1. **Dry Run Mode (Default):**
   - No real money at risk
   - All trades are simulated
   - Perfect for testing strategies

2. **Risk Limits:**
   - Maximum position size: 5% of portfolio
   - Maximum portfolio risk: 20%
   - Automatic stop losses: 2%

3. **Error Handling:**
   - Graceful error recovery
   - Automatic retry mechanisms
   - Comprehensive logging

### Switching to Live Trading

⚠️ **WARNING:** Only switch to live trading after thorough testing!

1. **Test thoroughly** in dry run mode
2. **Verify all signals** are reasonable
3. **Set up proper broker integration**
4. **Update environment variable:** `LIVE_TRADING=true`
5. **Start with small position sizes**

## 📞 Support and Maintenance

### Regular Maintenance Tasks

1. **Weekly:** Review trading performance and logs
2. **Monthly:** Update dependencies if needed
3. **Quarterly:** Review and adjust risk parameters

### Getting Help

1. **Check logs first** - most issues are logged
2. **Review configuration** - ensure environment variables are correct
3. **Test locally** - use Docker for local testing
4. **Monitor Render status** - check for platform issues

### Backup and Recovery

1. **Configuration backup:** Export environment variables
2. **Code backup:** Ensure GitHub repository is up to date
3. **Data backup:** Trading logs are automatically saved

## 🎯 Next Steps

After successful deployment:

1. **Monitor for 24-48 hours** in dry run mode
2. **Analyze generated signals** for quality
3. **Adjust thresholds** based on performance
4. **Consider adding more symbols** to watchlist
5. **Set up notifications** (email/webhook) for important events

---

## 📋 Deployment Checklist

- [ ] GitHub repository created with all files
- [ ] Render service configured
- [ ] Environment variables set
- [ ] Service deployed successfully
- [ ] Logs show "Trading Bot Started"
- [ ] Signals being generated (check logs)
- [ ] Performance monitoring set up
- [ ] Safety limits verified

**Congratulations! Your unified trading bot is now running on Render! 🎉**

For questions or issues, check the logs first, then review this guide. The bot is designed to be robust and self-recovering, but monitoring is always recommended.
