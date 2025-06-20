
#!/usr/bin/env python3
"""
Unified Trading Bot - Combines RSI-MACD and Options Trading Strategies
=====================================================================

This unified bot combines both RSI-MACD and options trading strategies into a single service
that can be deployed on Render. It addresses the following issues:
- File path consistency across all modules
- Improved error handling and fallback mechanisms
- Single service architecture for easier deployment
- Better logging and debugging capabilities
- Configurable signal strength requirements for testing

Author: Trading Bot Team
Version: 2.0.0
"""

import asyncio
import logging
import os
import sys
import argparse
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import traceback

# Third-party imports
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel

# Local imports
from logger import setup_logger, log_trade_signal, log_error
from config_manager import ConfigManager
from market_data import MarketDataProvider
from signal_analyzer import SignalAnalyzer
from options_trader import OptionsTrader
from risk_manager import RiskManager

console = Console()

@dataclass
class TradingSignal:
    """Represents a trading signal with all relevant information"""
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strategy: str     # 'RSI_MACD', 'OPTIONS'
    strength: float   # Signal strength (0.0 to 1.0)
    price: float
    timestamp: datetime
    indicators: Dict[str, Any]
    confidence: float
    risk_score: float

class UnifiedTradingBot:
    """
    Unified Trading Bot that combines RSI-MACD and Options trading strategies
    """
    
    def __init__(self, config_path: str = None, dry_run: bool = True):
        """
        Initialize the unified trading bot
        
        Args:
            config_path: Path to configuration file
            dry_run: Whether to run in simulation mode
        """
        self.dry_run = dry_run
        self.running = False
        self.last_update = None
        
        # Initialize components
        self.config = ConfigManager(config_path)
        self.logger = setup_logger(
            name="unified_trading_bot",
            level=self.config.get('logging.level', 'INFO'),
            log_file=self.config.get('logging.file', 'trading_bot.log')
        )
        
        self.market_data = MarketDataProvider(self.config)
        self.signal_analyzer = SignalAnalyzer(self.config)
        self.options_trader = OptionsTrader(self.config, dry_run=dry_run)
        self.risk_manager = RiskManager(self.config)
        
        # Trading state
        self.active_positions = {}
        self.signal_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'successful_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0
        }
        
        self.logger.info(f"Unified Trading Bot initialized (dry_run={dry_run})")
        
    async def start(self):
        """Start the unified trading bot"""
        self.running = True
        self.logger.info("Starting Unified Trading Bot...")
        
        console.print(Panel.fit(
            "[bold green]🚀 Unified Trading Bot Started[/bold green]\n"
            f"Mode: {'DRY RUN' if self.dry_run else 'LIVE TRADING'}\n"
            f"Strategies: RSI-MACD + Options Trading\n"
            f"Update Interval: {self.config.get('bot.update_interval', 60)}s",
            title="Trading Bot Status"
        ))
        
        try:
            while self.running:
                await self.trading_cycle()
                await asyncio.sleep(self.config.get('bot.update_interval', 60))
                
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        except Exception as e:
            self.logger.error(f"Critical error in main loop: {e}")
            log_error(self.logger, e, "Main trading loop")
        finally:
            await self.shutdown()
    
    async def trading_cycle(self):
        """Execute one complete trading cycle"""
        try:
            cycle_start = time.time()
            self.logger.info("Starting trading cycle...")
            
            # Get watchlist symbols
            symbols = self.config.get('trading.watchlist', ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA'])
            
            # Analyze each symbol
            signals = []
            for symbol in symbols:
                try:
                    symbol_signals = await self.analyze_symbol(symbol)
                    signals.extend(symbol_signals)
                except Exception as e:
                    self.logger.error(f"Error analyzing {symbol}: {e}")
                    continue
            
            # Filter and rank signals
            filtered_signals = self.filter_signals(signals)
            
            # Execute trades based on signals
            if filtered_signals:
                await self.execute_trades(filtered_signals)
            
            # Update performance metrics
            self.update_performance_metrics()
            
            # Log cycle completion
            cycle_time = time.time() - cycle_start
            self.logger.info(f"Trading cycle completed in {cycle_time:.2f}s")
            self.last_update = datetime.now()
            
            # Display status
            self.display_status(filtered_signals)
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
            log_error(self.logger, e, "Trading cycle")
    
    async def analyze_symbol(self, symbol: str) -> List[TradingSignal]:
        """
        Analyze a symbol using both RSI-MACD and Options strategies
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            List of trading signals
        """
        signals = []
        
        try:
            # Get market data
            data = await self.market_data.get_stock_data(symbol)
            if data is None or data.empty:
                self.logger.warning(f"No data available for {symbol}")
                return signals
            
            current_price = data['Close'].iloc[-1]
            
            # RSI-MACD Analysis
            rsi_macd_signal = await self.analyze_rsi_macd(symbol, data)
            if rsi_macd_signal:
                signals.append(rsi_macd_signal)
            
            # Options Analysis (if enabled)
            if self.config.get('strategies.options.enabled', True):
                options_signal = await self.analyze_options(symbol, data)
                if options_signal:
                    signals.append(options_signal)
            
        except Exception as e:
            self.logger.error(f"Error analyzing symbol {symbol}: {e}")
            log_error(self.logger, e, f"Symbol analysis: {symbol}")
        
        return signals
    
    async def analyze_rsi_macd(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Analyze symbol using RSI-MACD strategy
        
        Args:
            symbol: Stock symbol
            data: Historical price data
            
        Returns:
            Trading signal if conditions are met
        """
        try:
            # Calculate technical indicators
            indicators = self.signal_analyzer.calculate_rsi_macd_indicators(data)
            
            # Get current values
            current_rsi = indicators['rsi'].iloc[-1]
            current_macd = indicators['macd'].iloc[-1]
            current_signal = indicators['macd_signal'].iloc[-1]
            current_histogram = indicators['macd_histogram'].iloc[-1]
            current_price = data['Close'].iloc[-1]
            
            # Determine signal
            signal_type = 'HOLD'
            strength = 0.0
            confidence = 0.0
            
            # RSI oversold/overbought conditions
            rsi_oversold = current_rsi < self.config.get('indicators.rsi.oversold_threshold', 30)
            rsi_overbought = current_rsi > self.config.get('indicators.rsi.overbought_threshold', 70)
            
            # MACD conditions
            macd_bullish = current_macd > current_signal and current_histogram > 0
            macd_bearish = current_macd < current_signal and current_histogram < 0
            
            # Generate signals with lowered thresholds for testing
            min_strength = self.config.get('signals.min_strength', 0.3)  # Lowered from 0.6
            
            if rsi_oversold and macd_bullish:
                signal_type = 'BUY'
                strength = min(0.8, (30 - current_rsi) / 30 + abs(current_histogram) / 100)
                confidence = 0.75
            elif rsi_overbought and macd_bearish:
                signal_type = 'SELL'
                strength = min(0.8, (current_rsi - 70) / 30 + abs(current_histogram) / 100)
                confidence = 0.75
            elif macd_bullish and current_rsi > 40:
                signal_type = 'BUY'
                strength = min(0.6, abs(current_histogram) / 100)
                confidence = 0.6
            elif macd_bearish and current_rsi < 60:
                signal_type = 'SELL'
                strength = min(0.6, abs(current_histogram) / 100)
                confidence = 0.6
            
            # Only return signal if strength meets minimum threshold
            if strength >= min_strength:
                risk_score = self.risk_manager.calculate_risk_score(symbol, data, signal_type)
                
                return TradingSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    strategy='RSI_MACD',
                    strength=strength,
                    price=current_price,
                    timestamp=datetime.now(),
                    indicators={
                        'rsi': current_rsi,
                        'macd': current_macd,
                        'macd_signal': current_signal,
                        'macd_histogram': current_histogram
                    },
                    confidence=confidence,
                    risk_score=risk_score
                )
            
        except Exception as e:
            self.logger.error(f"Error in RSI-MACD analysis for {symbol}: {e}")
            log_error(self.logger, e, f"RSI-MACD analysis: {symbol}")
        
        return None
    
    async def analyze_options(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Analyze symbol for options trading opportunities
        
        Args:
            symbol: Stock symbol
            data: Historical price data
            
        Returns:
            Options trading signal if conditions are met
        """
        try:
            # Get options data
            options_data = await self.market_data.get_options_data(symbol)
            if not options_data:
                return None
            
            # Analyze options flow and volatility
            signal_info = self.signal_analyzer.analyze_options_signals(symbol, data, options_data)
            
            if signal_info and signal_info['strength'] >= self.config.get('signals.min_strength', 0.3):
                current_price = data['Close'].iloc[-1]
                risk_score = self.risk_manager.calculate_risk_score(symbol, data, signal_info['signal_type'])
                
                return TradingSignal(
                    symbol=symbol,
                    signal_type=signal_info['signal_type'],
                    strategy='OPTIONS',
                    strength=signal_info['strength'],
                    price=current_price,
                    timestamp=datetime.now(),
                    indicators=signal_info['indicators'],
                    confidence=signal_info['confidence'],
                    risk_score=risk_score
                )
                
        except Exception as e:
            self.logger.error(f"Error in options analysis for {symbol}: {e}")
            log_error(self.logger, e, f"Options analysis: {symbol}")
        
        return None
    
    def filter_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """
        Filter and rank trading signals based on strength and risk
        
        Args:
            signals: List of trading signals
            
        Returns:
            Filtered and sorted list of signals
        """
        if not signals:
            return []
        
        # Filter by minimum strength and maximum risk
        filtered = []
        for signal in signals:
            if (signal.strength >= self.config.get('signals.min_strength', 0.3) and
                signal.risk_score <= self.config.get('risk.max_risk_score', 0.8)):
                filtered.append(signal)
        
        # Sort by combined score (strength * confidence / risk_score)
        filtered.sort(key=lambda s: (s.strength * s.confidence / max(s.risk_score, 0.1)), reverse=True)
        
        # Limit number of signals
        max_signals = self.config.get('signals.max_concurrent', 5)
        return filtered[:max_signals]
    
    async def execute_trades(self, signals: List[TradingSignal]):
        """
        Execute trades based on filtered signals
        
        Args:
            signals: List of trading signals to execute
        """
        for signal in signals:
            try:
                await self.execute_single_trade(signal)
            except Exception as e:
                self.logger.error(f"Error executing trade for {signal.symbol}: {e}")
                log_error(self.logger, e, f"Trade execution: {signal.symbol}")
    
    async def execute_single_trade(self, signal: TradingSignal):
        """
        Execute a single trade based on signal
        
        Args:
            signal: Trading signal to execute
        """
        try:
            # Log the signal
            log_trade_signal(self.logger, signal)
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                signal.symbol, signal.price, signal.risk_score
            )
            
            if position_size <= 0:
                self.logger.warning(f"Position size too small for {signal.symbol}")
                return
            
            # Execute based on strategy
            if signal.strategy == 'RSI_MACD':
                await self.execute_stock_trade(signal, position_size)
            elif signal.strategy == 'OPTIONS':
                await self.execute_options_trade(signal, position_size)
            
            # Update performance tracking
            self.performance_metrics['total_trades'] += 1
            self.signal_history.append(signal)
            
        except Exception as e:
            self.logger.error(f"Error executing single trade: {e}")
            log_error(self.logger, e, "Single trade execution")
    
    async def execute_stock_trade(self, signal: TradingSignal, position_size: float):
        """Execute stock trade"""
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would execute stock trade: {signal.signal_type} "
                           f"{position_size} shares of {signal.symbol} at ${signal.price:.2f}")
        else:
            # Implement actual stock trading logic here
            self.logger.info(f"Executing stock trade: {signal.signal_type} "
                           f"{position_size} shares of {signal.symbol}")
    
    async def execute_options_trade(self, signal: TradingSignal, position_size: float):
        """Execute options trade"""
        try:
            result = await self.options_trader.execute_trade(signal, position_size)
            if result:
                self.logger.info(f"Options trade executed successfully: {result}")
            else:
                self.logger.warning(f"Options trade failed for {signal.symbol}")
        except Exception as e:
            self.logger.error(f"Error in options trade execution: {e}")
    
    def update_performance_metrics(self):
        """Update performance metrics"""
        try:
            if self.performance_metrics['total_trades'] > 0:
                self.performance_metrics['win_rate'] = (
                    self.performance_metrics['successful_trades'] / 
                    self.performance_metrics['total_trades']
                )
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def display_status(self, signals: List[TradingSignal]):
        """Display current bot status"""
        try:
            # Create status table
            table = Table(title="Trading Bot Status")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Mode", "DRY RUN" if self.dry_run else "LIVE")
            table.add_row("Last Update", str(self.last_update) if self.last_update else "Never")
            table.add_row("Active Signals", str(len(signals)))
            table.add_row("Total Trades", str(self.performance_metrics['total_trades']))
            table.add_row("Win Rate", f"{self.performance_metrics['win_rate']:.2%}")
            
            console.print(table)
            
            # Display active signals
            if signals:
                signal_table = Table(title="Active Signals")
                signal_table.add_column("Symbol")
                signal_table.add_column("Strategy")
                signal_table.add_column("Signal")
                signal_table.add_column("Strength")
                signal_table.add_column("Price")
                
                for signal in signals[:5]:  # Show top 5
                    signal_table.add_row(
                        signal.symbol,
                        signal.strategy,
                        signal.signal_type,
                        f"{signal.strength:.2f}",
                        f"${signal.price:.2f}"
                    )
                
                console.print(signal_table)
                
        except Exception as e:
            self.logger.error(f"Error displaying status: {e}")
    
    async def shutdown(self):
        """Gracefully shutdown the bot"""
        self.running = False
        self.logger.info("Shutting down Unified Trading Bot...")
        
        try:
            # Close any open positions if needed
            await self.close_positions()
            
            # Save performance data
            self.save_performance_data()
            
            console.print(Panel.fit(
                "[bold red]🛑 Trading Bot Stopped[/bold red]\n"
                f"Total Trades: {self.performance_metrics['total_trades']}\n"
                f"Win Rate: {self.performance_metrics['win_rate']:.2%}",
                title="Shutdown Complete"
            ))
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def close_positions(self):
        """Close any open positions"""
        # Implement position closing logic
        pass
    
    def save_performance_data(self):
        """Save performance data to file"""
        try:
            data = {
                'performance_metrics': self.performance_metrics,
                'signal_history': [
                    {
                        'symbol': s.symbol,
                        'strategy': s.strategy,
                        'signal_type': s.signal_type,
                        'strength': s.strength,
                        'timestamp': s.timestamp.isoformat()
                    }
                    for s in self.signal_history[-100:]  # Last 100 signals
                ]
            }
            
            with open('trading_performance.json', 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving performance data: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Unified Trading Bot')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='Run in simulation mode (default: True)')
    parser.add_argument('--live', action='store_true',
                       help='Run in live trading mode (overrides dry-run)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Determine run mode
    dry_run = not args.live if args.live else args.dry_run
    
    try:
        # Create and start bot
        bot = UnifiedTradingBot(
            config_path=args.config,
            dry_run=dry_run
        )
        
        # Run the bot
        asyncio.run(bot.start())
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Critical error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
