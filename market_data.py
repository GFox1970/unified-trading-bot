
"""
Market Data Provider for Unified Trading Bot
============================================

Provides unified interface for fetching market data from various sources
including stock prices, options data, and technical indicators.

Features:
- Multiple data provider support (yfinance, Alpha Vantage, Polygon)
- Caching and rate limiting
- Error handling and fallback mechanisms
- Real-time and historical data
- Options chain data
- Technical indicator calculation
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from dataclasses import dataclass
from pathlib import Path
import pickle
import json

@dataclass
class OptionsData:
    """Container for options data"""
    calls: pd.DataFrame
    puts: pd.DataFrame
    expiration_dates: List[str]
    underlying_price: float
    timestamp: datetime

class DataCache:
    """Simple caching mechanism for market data"""
    yf.pdr_override()
    def __init__(self, cache_dir: str = "cache", max_age_minutes: int = 5):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_age = timedelta(minutes=max_age_minutes)
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached data if still valid"""
        cache_file = self.cache_dir / f"{key}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data, timestamp = pickle.load(f)
            
            if datetime.now() - timestamp < self.max_age:
                return data
            else:
                # Cache expired, remove file
                cache_file.unlink()
                return None
                
        except Exception as e:
            self.logger.warning(f"Error reading cache for {key}: {e}")
            return None
    
    def set(self, key: str, data: Any):
        """Cache data with timestamp"""
        cache_file = self.cache_dir / f"{key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump((data, datetime.now()), f)
        except Exception as e:
            self.logger.warning(f"Error writing cache for {key}: {e}")

class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, max_calls_per_second: int = 5):
        self.max_calls = max_calls_per_second
        self.calls = []
    
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        
        # Remove calls older than 1 second
        self.calls = [call_time for call_time in self.calls if now - call_time < 1.0]
        
        # If we're at the limit, wait
        if len(self.calls) >= self.max_calls:
            sleep_time = 1.0 - (now - self.calls[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        # Record this call
        self.calls.append(now)

class MarketDataProvider:
    """
    Unified market data provider supporting multiple data sources
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.cache = DataCache(
            cache_dir=config.get('market_data.cache_dir', 'cache'),
            max_age_minutes=config.get('market_data.cache_minutes', 5)
        )
        
        self.rate_limiter = RateLimiter(
            max_calls_per_second=config.get('market_data.rate_limit', 5)
        )
        
        # Data provider settings
        self.provider = config.get('market_data.provider', 'yfinance')
        self.timeout = config.get('market_data.timeout', 30)
        self.history_days = config.get('market_data.history_days', 252)
        
        self.logger.info(f"Market data provider initialized: {self.provider}")
    
    async def get_stock_data(self, symbol: str, period: str = None) -> Optional[pd.DataFrame]:
        """
        Get stock price data for a symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        cache_key = f"stock_{symbol}_{period or 'default'}"
        
        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            await self.rate_limiter.wait_if_needed()
            
            if self.provider == 'yfinance':
                data = await self._get_yfinance_data(symbol, period)
            else:
                self.logger.error(f"Unsupported data provider: {self.provider}")
                return None
            
            if data is not None and not data.empty:
                # Cache the data
                self.cache.set(cache_key, data)
                self.logger.debug(f"Retrieved stock data for {symbol}: {len(data)} rows")
                return data
            else:
                self.logger.warning(f"No data retrieved for {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting stock data for {symbol}: {e}")
            return None
    
    async def _get_yfinance_data(self, symbol: str, period: str = None) -> Optional[pd.DataFrame]:
        """Get data using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            
            if period:
                data = ticker.history(period=period)
            else:
                # Default to recent history
                end_date = datetime.now()
                start_date = end_date - timedelta(days=self.history_days)
                data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                return None
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                self.logger.error(f"Missing required columns in data for {symbol}")
                return None
            
            # Clean the data
            data = data.dropna()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in yfinance data retrieval for {symbol}: {e}")
            return None
    
    async def get_options_data(self, symbol: str) -> Optional[OptionsData]:
        """
        Get options chain data for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            OptionsData object or None if error
        """
        cache_key = f"options_{symbol}"
        
        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            await self.rate_limiter.wait_if_needed()
            
            if self.provider == 'yfinance':
                options_data = await self._get_yfinance_options(symbol)
            else:
                self.logger.error(f"Options data not supported for provider: {self.provider}")
                return None
            
            if options_data:
                # Cache the data
                self.cache.set(cache_key, options_data)
                self.logger.debug(f"Retrieved options data for {symbol}")
                return options_data
            else:
                self.logger.warning(f"No options data retrieved for {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting options data for {symbol}: {e}")
            return None
    
    async def _get_yfinance_options(self, symbol: str) -> Optional[OptionsData]:
        """Get options data using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current stock price
            info = ticker.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if not current_price:
                # Fallback to recent price data
                recent_data = ticker.history(period='1d')
                if not recent_data.empty:
                    current_price = recent_data['Close'].iloc[-1]
                else:
                    self.logger.error(f"Could not get current price for {symbol}")
                    return None
            
            # Get expiration dates
            expiration_dates = ticker.options
            if not expiration_dates:
                self.logger.warning(f"No options expiration dates for {symbol}")
                return None
            
            # Get options chain for the nearest expiration
            nearest_expiry = expiration_dates[0]
            options_chain = ticker.option_chain(nearest_expiry)
            
            calls = options_chain.calls
            puts = options_chain.puts
            
            # Filter options based on configuration
            calls = self._filter_options(calls, current_price)
            puts = self._filter_options(puts, current_price)
            
            return OptionsData(
                calls=calls,
                puts=puts,
                expiration_dates=list(expiration_dates),
                underlying_price=current_price,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error in yfinance options retrieval for {symbol}: {e}")
            return None
    
    def _filter_options(self, options_df: pd.DataFrame, underlying_price: float) -> pd.DataFrame:
        """Filter options based on configuration criteria"""
        if options_df.empty:
            return options_df
        
        try:
            # Filter by volume
            min_volume = self.config.get('options.min_volume', 50)
            options_df = options_df[options_df['volume'] >= min_volume]
            
            # Filter by open interest
            min_open_interest = self.config.get('options.min_open_interest', 100)
            options_df = options_df[options_df['openInterest'] >= min_open_interest]
            
            # Filter by bid-ask spread
            max_spread = self.config.get('options.max_bid_ask_spread', 0.20)
            spread_ratio = (options_df['ask'] - options_df['bid']) / options_df['ask']
            options_df = options_df[spread_ratio <= max_spread]
            
            # Filter by strike price range (within reasonable range of underlying)
            price_range = underlying_price * 0.3  # 30% range
            min_strike = underlying_price - price_range
            max_strike = underlying_price + price_range
            options_df = options_df[
                (options_df['strike'] >= min_strike) & 
                (options_df['strike'] <= max_strike)
            ]
            
            return options_df
            
        except Exception as e:
            self.logger.error(f"Error filtering options: {e}")
            return options_df
    
    async def get_multiple_stocks(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Get stock data for multiple symbols concurrently
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to their data
        """
        tasks = [self.get_stock_data(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data_dict = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                self.logger.error(f"Error getting data for {symbol}: {result}")
            elif result is not None:
                data_dict[symbol] = result
        
        return data_dict
    
    async def get_real_time_price(self, symbol: str) -> Optional[float]:
        """
        Get real-time price for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Current price or None if error
        """
        try:
            # For yfinance, we'll get the most recent price from 1-day data
            data = await self.get_stock_data(symbol, period='1d')
            if data is not None and not data.empty:
                return float(data['Close'].iloc[-1])
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting real-time price for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate technical indicators for price data
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Dictionary of technical indicators
        """
        indicators = {}
        
        try:
            # RSI
            rsi_period = self.config.get('indicators.rsi.period', 14)
            indicators['rsi'] = self._calculate_rsi(data['Close'], rsi_period)
            
            # MACD
            fast_period = self.config.get('indicators.macd.fast_period', 12)
            slow_period = self.config.get('indicators.macd.slow_period', 26)
            signal_period = self.config.get('indicators.macd.signal_period', 9)
            
            macd_data = self._calculate_macd(data['Close'], fast_period, slow_period, signal_period)
            indicators.update(macd_data)
            
            # Moving Averages
            short_ma = self.config.get('indicators.moving_averages.short_period', 20)
            long_ma = self.config.get('indicators.moving_averages.long_period', 50)
            
            indicators['sma_short'] = data['Close'].rolling(window=short_ma).mean()
            indicators['sma_long'] = data['Close'].rolling(window=long_ma).mean()
            
            # Bollinger Bands
            indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = \
                self._calculate_bollinger_bands(data['Close'])
            
            # Volume indicators
            indicators['volume_sma'] = data['Volume'].rolling(window=20).mean()
            indicators['volume_ratio'] = data['Volume'] / indicators['volume_sma']
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
        
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        
        return {
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on market data provider
        
        Returns:
            Health status information
        """
        health_status = {
            'provider': self.provider,
            'status': 'unknown',
            'last_check': datetime.now().isoformat(),
            'errors': []
        }
        
        try:
            # Test with a common symbol
            test_data = await self.get_stock_data('SPY', period='1d')
            
            if test_data is not None and not test_data.empty:
                health_status['status'] = 'healthy'
                health_status['test_symbol'] = 'SPY'
                health_status['test_data_points'] = len(test_data)
            else:
                health_status['status'] = 'unhealthy'
                health_status['errors'].append('No data returned for test symbol')
                
        except Exception as e:
            health_status['status'] = 'error'
            health_status['errors'].append(str(e))
        
        return health_status

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from config_manager import ConfigManager
    
    async def test_market_data():
        # Test the market data provider
        config = ConfigManager()
        provider = MarketDataProvider(config)
        
        # Test stock data
        print("Testing stock data retrieval...")
        data = await provider.get_stock_data('AAPL')
        if data is not None:
            print(f"Retrieved {len(data)} data points for AAPL")
            print(data.tail())
        
        # Test options data
        print("\nTesting options data retrieval...")
        options = await provider.get_options_data('AAPL')
        if options:
            print(f"Retrieved options data: {len(options.calls)} calls, {len(options.puts)} puts")
        
        # Test technical indicators
        if data is not None:
            print("\nTesting technical indicators...")
            indicators = provider.calculate_technical_indicators(data)
            print(f"Calculated {len(indicators)} indicators")
            if 'rsi' in indicators:
                print(f"Current RSI: {indicators['rsi'].iloc[-1]:.2f}")
        
        # Health check
        print("\nPerforming health check...")
        health = await provider.health_check()
        print(f"Health status: {health}")
    
    # Run the test
    asyncio.run(test_market_data())
