
"""
Signal Analyzer for Unified Trading Bot
=======================================

Analyzes market data to generate trading signals using various strategies
including RSI-MACD combinations and options flow analysis.

Features:
- RSI-MACD signal generation with configurable thresholds
- Options flow analysis and unusual activity detection
- Signal strength calculation and confidence scoring
- Multiple timeframe analysis
- Volume and momentum confirmation
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
# import talib  # Optional - will use pandas-based calculations if not available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available - using pandas-based calculations")

@dataclass
class SignalResult:
    """Container for signal analysis results"""
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float   # 0.0 to 1.0
    confidence: float # 0.0 to 1.0
    indicators: Dict[str, Any]
    reasoning: List[str]
    timestamp: datetime

class SignalAnalyzer:
    """
    Analyzes market data to generate trading signals
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Signal thresholds (lowered for testing)
        self.min_strength = config.get('signals.min_strength', 0.3)
        self.min_confidence = config.get('signals.min_confidence', 0.5)
        
        # RSI settings
        self.rsi_period = config.get('indicators.rsi.period', 14)
        self.rsi_oversold = config.get('indicators.rsi.oversold_threshold', 30)
        self.rsi_overbought = config.get('indicators.rsi.overbought_threshold', 70)
        
        # MACD settings
        self.macd_fast = config.get('indicators.macd.fast_period', 12)
        self.macd_slow = config.get('indicators.macd.slow_period', 26)
        self.macd_signal = config.get('indicators.macd.signal_period', 9)
        
        self.logger.info("Signal analyzer initialized with relaxed thresholds for testing")
    
    def calculate_rsi_macd_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate RSI and MACD indicators
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        try:
            prices = data['Close']
            
            if TALIB_AVAILABLE:
                # Use TA-Lib if available
                prices_array = prices.values
                
                # RSI calculation
                indicators['rsi'] = pd.Series(
                    talib.RSI(prices_array, timeperiod=self.rsi_period),
                    index=data.index
                )
                
                # MACD calculation
                macd, macd_signal, macd_hist = talib.MACD(
                    prices_array,
                    fastperiod=self.macd_fast,
                    slowperiod=self.macd_slow,
                    signalperiod=self.macd_signal
                )
                
                indicators['macd'] = pd.Series(macd, index=data.index)
                indicators['macd_signal'] = pd.Series(macd_signal, index=data.index)
                indicators['macd_histogram'] = pd.Series(macd_hist, index=data.index)
                
                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = talib.BBANDS(prices_array, timeperiod=20)
                indicators['bb_upper'] = pd.Series(bb_upper, index=data.index)
                indicators['bb_middle'] = pd.Series(bb_middle, index=data.index)
                indicators['bb_lower'] = pd.Series(bb_lower, index=data.index)
                
                # ATR for volatility
                indicators['atr'] = pd.Series(
                    talib.ATR(data['High'].values, data['Low'].values, prices_array, timeperiod=14),
                    index=data.index
                )
            else:
                # Use pandas-based calculations
                # RSI calculation
                indicators['rsi'] = self._calculate_rsi_pandas(prices, self.rsi_period)
                
                # MACD calculation
                macd_data = self._calculate_macd_pandas(prices, self.macd_fast, self.macd_slow, self.macd_signal)
                indicators.update(macd_data)
                
                # Bollinger Bands
                bb_data = self._calculate_bollinger_bands_pandas(prices, 20, 2)
                indicators.update(bb_data)
                
                # ATR for volatility
                indicators['atr'] = self._calculate_atr_pandas(data, 14)
            
            # Additional indicators (same for both)
            indicators['sma_20'] = data['Close'].rolling(window=20).mean()
            indicators['sma_50'] = data['Close'].rolling(window=50).mean()
            indicators['volume_sma'] = data['Volume'].rolling(window=20).mean()
            indicators['volume_ratio'] = data['Volume'] / indicators['volume_sma']
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            
        return indicators
    
    def analyze_rsi_macd_signal(self, data: pd.DataFrame) -> Optional[SignalResult]:
        """
        Analyze RSI-MACD signals with enhanced logic
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            SignalResult or None
        """
        try:
            indicators = self.calculate_rsi_macd_indicators(data)
            
            if not indicators or len(data) < 50:
                return None
            
            # Get current values
            current_rsi = indicators['rsi'].iloc[-1]
            current_macd = indicators['macd'].iloc[-1]
            current_signal = indicators['macd_signal'].iloc[-1]
            current_histogram = indicators['macd_histogram'].iloc[-1]
            current_price = data['Close'].iloc[-1]
            current_volume_ratio = indicators['volume_ratio'].iloc[-1]
            
            # Previous values for trend analysis
            prev_rsi = indicators['rsi'].iloc[-2]
            prev_macd = indicators['macd'].iloc[-2]
            prev_histogram = indicators['macd_histogram'].iloc[-2]
            
            # Initialize signal analysis
            signal_type = 'HOLD'
            strength = 0.0
            confidence = 0.0
            reasoning = []
            
            # RSI conditions
            rsi_oversold = current_rsi < self.rsi_oversold
            rsi_overbought = current_rsi > self.rsi_overbought
            rsi_neutral = self.rsi_oversold <= current_rsi <= self.rsi_overbought
            
            # MACD conditions
            macd_bullish = current_macd > current_signal
            macd_bearish = current_macd < current_signal
            macd_crossover_up = (current_macd > current_signal) and (prev_macd <= indicators['macd_signal'].iloc[-2])
            macd_crossover_down = (current_macd < current_signal) and (prev_macd >= indicators['macd_signal'].iloc[-2])
            histogram_increasing = current_histogram > prev_histogram
            histogram_decreasing = current_histogram < prev_histogram
            
            # Volume confirmation
            volume_confirmation = current_volume_ratio > 1.2  # 20% above average
            
            # Moving average trend
            sma_20 = indicators['sma_20'].iloc[-1]
            sma_50 = indicators['sma_50'].iloc[-1]
            uptrend = sma_20 > sma_50
            downtrend = sma_20 < sma_50
            
            # Bollinger Bands position
            bb_upper = indicators['bb_upper'].iloc[-1]
            bb_lower = indicators['bb_lower'].iloc[-1]
            near_upper_band = current_price > (bb_upper * 0.98)
            near_lower_band = current_price < (bb_lower * 1.02)
            
            # Strong Buy Signals (Lowered thresholds for testing)
            if rsi_oversold and macd_crossover_up:
                signal_type = 'BUY'
                strength = min(0.9, 0.6 + (self.rsi_oversold - current_rsi) / self.rsi_oversold * 0.3)
                confidence = 0.8
                reasoning.append("RSI oversold with MACD bullish crossover")
                
            elif rsi_oversold and macd_bullish and histogram_increasing:
                signal_type = 'BUY'
                strength = min(0.8, 0.5 + (self.rsi_oversold - current_rsi) / self.rsi_oversold * 0.3)
                confidence = 0.75
                reasoning.append("RSI oversold with strengthening MACD")
                
            elif near_lower_band and macd_crossover_up and uptrend:
                signal_type = 'BUY'
                strength = 0.7
                confidence = 0.7
                reasoning.append("Near lower Bollinger Band with MACD crossover in uptrend")
            
            # Strong Sell Signals
            elif rsi_overbought and macd_crossover_down:
                signal_type = 'SELL'
                strength = min(0.9, 0.6 + (current_rsi - self.rsi_overbought) / (100 - self.rsi_overbought) * 0.3)
                confidence = 0.8
                reasoning.append("RSI overbought with MACD bearish crossover")
                
            elif rsi_overbought and macd_bearish and histogram_decreasing:
                signal_type = 'SELL'
                strength = min(0.8, 0.5 + (current_rsi - self.rsi_overbought) / (100 - self.rsi_overbought) * 0.3)
                confidence = 0.75
                reasoning.append("RSI overbought with weakening MACD")
                
            elif near_upper_band and macd_crossover_down and downtrend:
                signal_type = 'SELL'
                strength = 0.7
                confidence = 0.7
                reasoning.append("Near upper Bollinger Band with MACD crossover in downtrend")
            
            # Moderate Buy Signals (Relaxed for testing)
            elif macd_crossover_up and uptrend and current_rsi > 40:
                signal_type = 'BUY'
                strength = 0.5
                confidence = 0.6
                reasoning.append("MACD bullish crossover in uptrend")
                
            elif histogram_increasing and current_rsi < 50 and uptrend:
                signal_type = 'BUY'
                strength = 0.4
                confidence = 0.55
                reasoning.append("Strengthening MACD momentum in uptrend")
            
            # Moderate Sell Signals
            elif macd_crossover_down and downtrend and current_rsi < 60:
                signal_type = 'SELL'
                strength = 0.5
                confidence = 0.6
                reasoning.append("MACD bearish crossover in downtrend")
                
            elif histogram_decreasing and current_rsi > 50 and downtrend:
                signal_type = 'SELL'
                strength = 0.4
                confidence = 0.55
                reasoning.append("Weakening MACD momentum in downtrend")
            
            # Volume and momentum adjustments
            if volume_confirmation and signal_type != 'HOLD':
                strength = min(1.0, strength * 1.2)
                confidence = min(1.0, confidence * 1.1)
                reasoning.append("Volume confirmation")
            
            # ATR adjustment for volatility
            atr = indicators['atr'].iloc[-1]
            atr_ratio = atr / current_price
            if atr_ratio > 0.02:  # High volatility
                confidence *= 0.9
                reasoning.append("High volatility adjustment")
            
            # Only return signal if it meets minimum thresholds
            if strength >= self.min_strength and confidence >= self.min_confidence:
                return SignalResult(
                    signal_type=signal_type,
                    strength=strength,
                    confidence=confidence,
                    indicators={
                        'rsi': current_rsi,
                        'macd': current_macd,
                        'macd_signal': current_signal,
                        'macd_histogram': current_histogram,
                        'volume_ratio': current_volume_ratio,
                        'price': current_price,
                        'atr_ratio': atr_ratio
                    },
                    reasoning=reasoning,
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in RSI-MACD signal analysis: {e}")
            return None
    
    def analyze_options_signals(self, symbol: str, stock_data: pd.DataFrame, 
                              options_data) -> Optional[Dict[str, Any]]:
        """
        Analyze options data for trading signals
        
        Args:
            symbol: Stock symbol
            stock_data: Historical stock price data
            options_data: Options chain data
            
        Returns:
            Signal information dictionary or None
        """
        try:
            if not options_data or options_data.calls.empty or options_data.puts.empty:
                return None
            
            current_price = options_data.underlying_price
            calls = options_data.calls
            puts = options_data.puts
            
            # Calculate options metrics
            metrics = self._calculate_options_metrics(calls, puts, current_price)
            
            # Analyze unusual options activity
            unusual_activity = self._detect_unusual_options_activity(calls, puts)
            
            # Generate signal based on options analysis
            signal_info = self._generate_options_signal(metrics, unusual_activity, current_price)
            
            if signal_info and signal_info['strength'] >= self.min_strength:
                return signal_info
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in options signal analysis for {symbol}: {e}")
            return None
    
    def _calculate_options_metrics(self, calls: pd.DataFrame, puts: pd.DataFrame, 
                                 current_price: float) -> Dict[str, float]:
        """Calculate key options metrics"""
        metrics = {}
        
        try:
            # Put/Call Ratio
            total_call_volume = calls['volume'].sum()
            total_put_volume = puts['volume'].sum()
            
            if total_call_volume > 0:
                metrics['put_call_ratio'] = total_put_volume / total_call_volume
            else:
                metrics['put_call_ratio'] = 0
            
            # Open Interest Ratio
            total_call_oi = calls['openInterest'].sum()
            total_put_oi = puts['openInterest'].sum()
            
            if total_call_oi > 0:
                metrics['put_call_oi_ratio'] = total_put_oi / total_call_oi
            else:
                metrics['put_call_oi_ratio'] = 0
            
            # Volume-weighted average prices
            if total_call_volume > 0:
                metrics['call_vwap'] = (calls['lastPrice'] * calls['volume']).sum() / total_call_volume
            else:
                metrics['call_vwap'] = 0
                
            if total_put_volume > 0:
                metrics['put_vwap'] = (puts['lastPrice'] * puts['volume']).sum() / total_put_volume
            else:
                metrics['put_vwap'] = 0
            
            # Implied volatility metrics
            if 'impliedVolatility' in calls.columns:
                metrics['call_iv_avg'] = calls['impliedVolatility'].mean()
                metrics['put_iv_avg'] = puts['impliedVolatility'].mean()
                metrics['iv_skew'] = metrics['put_iv_avg'] - metrics['call_iv_avg']
            
            # Delta-weighted metrics
            if 'delta' in calls.columns:
                call_delta_exposure = (calls['delta'] * calls['openInterest']).sum()
                put_delta_exposure = (puts['delta'] * puts['openInterest']).sum()
                metrics['net_delta_exposure'] = call_delta_exposure + put_delta_exposure
            
        except Exception as e:
            self.logger.error(f"Error calculating options metrics: {e}")
        
        return metrics
    
    def _detect_unusual_options_activity(self, calls: pd.DataFrame, puts: pd.DataFrame) -> Dict[str, Any]:
        """Detect unusual options activity"""
        unusual = {
            'high_volume_calls': [],
            'high_volume_puts': [],
            'unusual_strikes': [],
            'activity_score': 0.0
        }
        
        try:
            # Define thresholds (lowered for testing)
            volume_threshold = 100  # Lowered from 500
            oi_ratio_threshold = 2.0  # Volume to OI ratio
            
            # High volume calls
            high_vol_calls = calls[calls['volume'] > volume_threshold]
            for _, option in high_vol_calls.iterrows():
                if option['openInterest'] > 0:
                    vol_oi_ratio = option['volume'] / option['openInterest']
                    if vol_oi_ratio > oi_ratio_threshold:
                        unusual['high_volume_calls'].append({
                            'strike': option['strike'],
                            'volume': option['volume'],
                            'oi': option['openInterest'],
                            'ratio': vol_oi_ratio
                        })
            
            # High volume puts
            high_vol_puts = puts[puts['volume'] > volume_threshold]
            for _, option in high_vol_puts.iterrows():
                if option['openInterest'] > 0:
                    vol_oi_ratio = option['volume'] / option['openInterest']
                    if vol_oi_ratio > oi_ratio_threshold:
                        unusual['high_volume_puts'].append({
                            'strike': option['strike'],
                            'volume': option['volume'],
                            'oi': option['openInterest'],
                            'ratio': vol_oi_ratio
                        })
            
            # Calculate activity score
            total_unusual_volume = (
                sum(item['volume'] for item in unusual['high_volume_calls']) +
                sum(item['volume'] for item in unusual['high_volume_puts'])
            )
            
            total_volume = calls['volume'].sum() + puts['volume'].sum()
            if total_volume > 0:
                unusual['activity_score'] = min(1.0, total_unusual_volume / total_volume)
            
        except Exception as e:
            self.logger.error(f"Error detecting unusual options activity: {e}")
        
        return unusual
    
    def _generate_options_signal(self, metrics: Dict[str, float], 
                               unusual_activity: Dict[str, Any], 
                               current_price: float) -> Optional[Dict[str, Any]]:
        """Generate trading signal based on options analysis"""
        try:
            signal_type = 'HOLD'
            strength = 0.0
            confidence = 0.0
            reasoning = []
            
            # Put/Call ratio analysis (lowered thresholds for testing)
            pc_ratio = metrics.get('put_call_ratio', 1.0)
            
            if pc_ratio < 0.6:  # Lowered from 0.8 - Bullish sentiment
                signal_type = 'BUY'
                strength += 0.3
                confidence += 0.2
                reasoning.append(f"Low put/call ratio ({pc_ratio:.2f}) indicates bullish sentiment")
                
            elif pc_ratio > 1.5:  # Lowered from 2.0 - Bearish sentiment
                signal_type = 'SELL'
                strength += 0.3
                confidence += 0.2
                reasoning.append(f"High put/call ratio ({pc_ratio:.2f}) indicates bearish sentiment")
            
            # Unusual activity analysis
            activity_score = unusual_activity.get('activity_score', 0.0)
            if activity_score > 0.1:  # Lowered from 0.2
                strength += activity_score * 0.5
                confidence += 0.1
                reasoning.append(f"Unusual options activity detected (score: {activity_score:.2f})")
                
                # Determine direction based on calls vs puts unusual activity
                call_activity = len(unusual_activity.get('high_volume_calls', []))
                put_activity = len(unusual_activity.get('high_volume_puts', []))
                
                if call_activity > put_activity and signal_type == 'HOLD':
                    signal_type = 'BUY'
                    reasoning.append("More unusual call activity than put activity")
                elif put_activity > call_activity and signal_type == 'HOLD':
                    signal_type = 'SELL'
                    reasoning.append("More unusual put activity than call activity")
            
            # IV skew analysis
            iv_skew = metrics.get('iv_skew', 0.0)
            if abs(iv_skew) > 0.05:  # Lowered from 0.1
                if iv_skew > 0:  # Put IV > Call IV (bearish)
                    if signal_type == 'HOLD':
                        signal_type = 'SELL'
                    strength += 0.2
                    reasoning.append("Put IV higher than call IV (bearish skew)")
                else:  # Call IV > Put IV (bullish)
                    if signal_type == 'HOLD':
                        signal_type = 'BUY'
                    strength += 0.2
                    reasoning.append("Call IV higher than put IV (bullish skew)")
            
            # Ensure strength and confidence are within bounds
            strength = min(1.0, strength)
            confidence = min(1.0, confidence)
            
            if signal_type != 'HOLD' and strength >= self.min_strength:
                return {
                    'signal_type': signal_type,
                    'strength': strength,
                    'confidence': confidence,
                    'indicators': {
                        'put_call_ratio': pc_ratio,
                        'activity_score': activity_score,
                        'iv_skew': iv_skew,
                        'unusual_calls': len(unusual_activity.get('high_volume_calls', [])),
                        'unusual_puts': len(unusual_activity.get('high_volume_puts', []))
                    },
                    'reasoning': reasoning
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating options signal: {e}")
            return None
    
    def combine_signals(self, rsi_macd_signal: Optional[SignalResult], 
                       options_signal: Optional[Dict[str, Any]]) -> Optional[SignalResult]:
        """
        Combine RSI-MACD and options signals into a unified signal
        
        Args:
            rsi_macd_signal: RSI-MACD signal result
            options_signal: Options signal result
            
        Returns:
            Combined signal result
        """
        try:
            if not rsi_macd_signal and not options_signal:
                return None
            
            # If only one signal exists, return it
            if rsi_macd_signal and not options_signal:
                return rsi_macd_signal
            
            if options_signal and not rsi_macd_signal:
                return SignalResult(
                    signal_type=options_signal['signal_type'],
                    strength=options_signal['strength'],
                    confidence=options_signal['confidence'],
                    indicators=options_signal['indicators'],
                    reasoning=options_signal['reasoning'],
                    timestamp=datetime.now()
                )
            
            # Both signals exist - combine them
            rsi_weight = self.config.get('strategies.rsi_macd.weight', 0.6)
            options_weight = self.config.get('strategies.options.weight', 0.4)
            
            # Determine combined signal type
            if rsi_macd_signal.signal_type == options_signal['signal_type']:
                # Signals agree
                combined_signal_type = rsi_macd_signal.signal_type
                combined_strength = (
                    rsi_macd_signal.strength * rsi_weight + 
                    options_signal['strength'] * options_weight
                )
                combined_confidence = (
                    rsi_macd_signal.confidence * rsi_weight + 
                    options_signal['confidence'] * options_weight
                )
            else:
                # Signals disagree - use the stronger one
                if rsi_macd_signal.strength > options_signal['strength']:
                    combined_signal_type = rsi_macd_signal.signal_type
                    combined_strength = rsi_macd_signal.strength * 0.8  # Reduce due to disagreement
                    combined_confidence = rsi_macd_signal.confidence * 0.8
                else:
                    combined_signal_type = options_signal['signal_type']
                    combined_strength = options_signal['strength'] * 0.8
                    combined_confidence = options_signal['confidence'] * 0.8
            
            # Combine indicators and reasoning
            combined_indicators = {**rsi_macd_signal.indicators}
            combined_indicators.update(options_signal['indicators'])
            
            combined_reasoning = rsi_macd_signal.reasoning + options_signal['reasoning']
            
            return SignalResult(
                signal_type=combined_signal_type,
                strength=combined_strength,
                confidence=combined_confidence,
                indicators=combined_indicators,
                reasoning=combined_reasoning,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error combining signals: {e}")
            return rsi_macd_signal or SignalResult(
                signal_type=options_signal['signal_type'],
                strength=options_signal['strength'],
                confidence=options_signal['confidence'],
                indicators=options_signal['indicators'],
                reasoning=options_signal['reasoning'],
                timestamp=datetime.now()
            ) if options_signal else None
    
    def _calculate_rsi_pandas(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI using pandas"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd_pandas(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD using pandas"""
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
    
    def _calculate_bollinger_bands_pandas(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands using pandas"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'bb_upper': upper_band,
            'bb_middle': sma,
            'bb_lower': lower_band
        }
    
    def _calculate_atr_pandas(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR using pandas"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr

# Example usage and testing
if __name__ == "__main__":
    from config_manager import ConfigManager
    import yfinance as yf
    
    # Test the signal analyzer
    config = ConfigManager()
    analyzer = SignalAnalyzer(config)
    
    # Get test data
    ticker = yf.Ticker('AAPL')
    data = ticker.history(period='3mo')
    
    if not data.empty:
        print("Testing RSI-MACD signal analysis...")
        signal = analyzer.analyze_rsi_macd_signal(data)
        
        if signal:
            print(f"Signal: {signal.signal_type}")
            print(f"Strength: {signal.strength:.2f}")
            print(f"Confidence: {signal.confidence:.2f}")
            print(f"Reasoning: {', '.join(signal.reasoning)}")
        else:
            print("No signal generated")
    
    print("Signal analyzer test completed.")
