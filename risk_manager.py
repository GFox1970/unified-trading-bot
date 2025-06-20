
"""
Risk Manager for Unified Trading Bot
===================================

Provides comprehensive risk management including position sizing,
portfolio risk assessment, and trade validation.

Features:
- Dynamic position sizing based on volatility and risk score
- Portfolio-level risk monitoring
- Maximum drawdown protection
- Correlation analysis for diversification
- Real-time risk metrics calculation
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import math

@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    portfolio_risk: float
    position_risk: float
    var_1day: float  # Value at Risk (1 day)
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    correlation_risk: float

@dataclass
class PositionRisk:
    """Risk assessment for a single position"""
    symbol: str
    risk_score: float
    volatility: float
    beta: float
    max_position_size: float
    recommended_size: float
    stop_loss_level: float
    take_profit_level: float

class RiskManager:
    """
    Comprehensive risk management for trading operations
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk parameters
        self.max_position_size = config.get('risk.max_position_size', 0.05)  # 5% max per position
        self.max_portfolio_risk = config.get('risk.max_portfolio_risk', 0.20)  # 20% max portfolio risk
        self.max_risk_score = config.get('risk.max_risk_score', 0.8)  # Lowered for testing
        self.stop_loss_pct = config.get('risk.stop_loss_percentage', 0.02)  # 2% stop loss
        self.take_profit_pct = config.get('risk.take_profit_percentage', 0.06)  # 6% take profit
        
        # Position sizing parameters
        self.base_position_size = config.get('risk.base_position_size', 1000)  # $1000 base
        self.risk_per_trade = config.get('risk.risk_per_trade', 0.01)  # 1% risk per trade
        
        # Portfolio tracking
        self.portfolio_value = 100000  # Starting portfolio value
        self.current_positions = {}
        self.position_history = []
        self.daily_returns = []
        
        # Risk-free rate for Sharpe ratio calculation
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
        self.logger.info("Risk manager initialized with relaxed parameters for testing")
    
    def calculate_risk_score(self, symbol: str, data: pd.DataFrame, signal_type: str) -> float:
        """
        Calculate comprehensive risk score for a symbol
        
        Args:
            symbol: Stock symbol
            data: Historical price data
            signal_type: 'BUY' or 'SELL'
            
        Returns:
            Risk score (0.0 = low risk, 1.0 = high risk)
        """
        try:
            if data.empty or len(data) < 20:
                return 1.0  # Maximum risk for insufficient data
            
            risk_factors = {}
            
            # 1. Volatility risk
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            risk_factors['volatility'] = min(1.0, volatility / 0.5)  # Normalize to 50% vol
            
            # 2. Trend consistency risk
            sma_20 = data['Close'].rolling(window=20).mean()
            sma_50 = data['Close'].rolling(window=min(50, len(data))).mean()
            
            if len(sma_50.dropna()) > 0:
                trend_consistency = abs(sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
                risk_factors['trend'] = min(1.0, trend_consistency * 2)
            else:
                risk_factors['trend'] = 0.5
            
            # 3. Volume risk (low volume = higher risk)
            avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
            recent_volume = data['Volume'].iloc[-5:].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 0.5
            risk_factors['volume'] = max(0.0, 1.0 - volume_ratio)
            
            # 4. Price momentum risk
            price_change_5d = (data['Close'].iloc[-1] - data['Close'].iloc[-6]) / data['Close'].iloc[-6]
            momentum_risk = abs(price_change_5d)
            risk_factors['momentum'] = min(1.0, momentum_risk * 5)
            
            # 5. Market cap risk (smaller = riskier, simulated)
            # In real implementation, this would come from fundamental data
            risk_factors['market_cap'] = 0.3  # Assume medium-cap stock
            
            # 6. Sector/correlation risk
            # Simplified - would normally check sector correlations
            risk_factors['correlation'] = 0.2
            
            # 7. Signal direction risk
            if signal_type == 'SELL':
                risk_factors['direction'] = 0.1  # Slightly higher risk for short positions
            else:
                risk_factors['direction'] = 0.0
            
            # Calculate weighted risk score
            weights = {
                'volatility': 0.25,
                'trend': 0.20,
                'volume': 0.15,
                'momentum': 0.15,
                'market_cap': 0.10,
                'correlation': 0.10,
                'direction': 0.05
            }
            
            risk_score = sum(risk_factors[factor] * weights[factor] for factor in weights)
            
            # Apply risk adjustments for testing (more lenient)
            if self.config.get('development.test_mode', False):
                risk_score *= 0.8  # Reduce risk scores by 20% in test mode
            
            return min(1.0, max(0.0, risk_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score for {symbol}: {e}")
            return 0.8  # Conservative default
    
    def calculate_position_size(self, symbol: str, price: float, risk_score: float) -> float:
        """
        Calculate optimal position size based on risk
        
        Args:
            symbol: Stock symbol
            price: Current price
            risk_score: Risk score (0.0 to 1.0)
            
        Returns:
            Position size in number of shares
        """
        try:
            # Base position size in dollars
            base_size_dollars = self.base_position_size
            
            # Adjust for risk score (lower risk = larger position)
            risk_adjustment = 1.0 - (risk_score * 0.5)  # 50% max reduction
            adjusted_size_dollars = base_size_dollars * risk_adjustment
            
            # Apply maximum position size limit
            max_size_dollars = self.portfolio_value * self.max_position_size
            final_size_dollars = min(adjusted_size_dollars, max_size_dollars)
            
            # Convert to shares
            shares = final_size_dollars / price
            
            # Ensure minimum viable position
            min_shares = max(1, 100 / price)  # At least $100 position or 1 share
            
            return max(min_shares, shares)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0
    
    def validate_trade(self, symbol: str, signal_type: str, position_size: float, 
                      risk_score: float) -> Tuple[bool, str]:
        """
        Validate if a trade meets risk management criteria
        
        Args:
            symbol: Stock symbol
            signal_type: 'BUY' or 'SELL'
            position_size: Proposed position size
            risk_score: Risk score for the trade
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Check risk score
            if risk_score > self.max_risk_score:
                return False, f"Risk score {risk_score:.2f} exceeds maximum {self.max_risk_score}"
            
            # Check position size
            position_value = position_size * 100  # Assuming $100 average price
            max_position_value = self.portfolio_value * self.max_position_size
            
            if position_value > max_position_value:
                return False, f"Position size ${position_value:.0f} exceeds maximum ${max_position_value:.0f}"
            
            # Check portfolio risk
            current_portfolio_risk = self._calculate_current_portfolio_risk()
            if current_portfolio_risk > self.max_portfolio_risk:
                return False, f"Portfolio risk {current_portfolio_risk:.2%} exceeds maximum {self.max_portfolio_risk:.2%}"
            
            # Check for existing position in same symbol
            if symbol in self.current_positions:
                existing_position = self.current_positions[symbol]
                if existing_position['signal_type'] == signal_type:
                    return False, f"Already have {signal_type} position in {symbol}"
            
            # Check maximum number of positions
            max_positions = self.config.get('bot.max_concurrent_trades', 5)
            if len(self.current_positions) >= max_positions:
                return False, f"Maximum number of positions ({max_positions}) reached"
            
            return True, "Trade validated"
            
        except Exception as e:
            self.logger.error(f"Error validating trade: {e}")
            return False, f"Validation error: {str(e)}"
    
    def _calculate_current_portfolio_risk(self) -> float:
        """Calculate current portfolio risk level"""
        try:
            if not self.current_positions:
                return 0.0
            
            # Simple risk calculation based on position sizes and risk scores
            total_risk = 0.0
            for position in self.current_positions.values():
                position_risk = position['size'] * position['risk_score']
                total_risk += position_risk
            
            return total_risk / self.portfolio_value
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk: {e}")
            return 1.0  # Conservative default
    
    def calculate_stop_loss_take_profit(self, entry_price: float, signal_type: str, 
                                      volatility: float = None) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels
        
        Args:
            entry_price: Entry price
            signal_type: 'BUY' or 'SELL'
            volatility: Optional volatility adjustment
            
        Returns:
            Tuple of (stop_loss, take_profit)
        """
        try:
            # Base percentages
            stop_loss_pct = self.stop_loss_pct
            take_profit_pct = self.take_profit_pct
            
            # Adjust for volatility if provided
            if volatility:
                # Higher volatility = wider stops
                vol_adjustment = min(2.0, max(0.5, volatility / 0.2))  # Normalize to 20% vol
                stop_loss_pct *= vol_adjustment
                take_profit_pct *= vol_adjustment
            
            if signal_type == 'BUY':
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + take_profit_pct)
            else:  # SELL
                stop_loss = entry_price * (1 + stop_loss_pct)
                take_profit = entry_price * (1 - take_profit_pct)
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss/take profit: {e}")
            return entry_price * 0.98, entry_price * 1.06  # Default 2%/6%
    
    def update_position(self, symbol: str, current_price: float, signal_type: str, 
                       position_size: float, risk_score: float):
        """Update or add position to tracking"""
        try:
            self.current_positions[symbol] = {
                'signal_type': signal_type,
                'size': position_size,
                'entry_price': current_price,
                'current_price': current_price,
                'risk_score': risk_score,
                'entry_time': datetime.now(),
                'unrealized_pnl': 0.0
            }
            
            self.logger.debug(f"Position updated: {symbol} {signal_type} {position_size} shares")
            
        except Exception as e:
            self.logger.error(f"Error updating position: {e}")
    
    def close_position(self, symbol: str, exit_price: float) -> Dict[str, Any]:
        """Close a position and calculate final P&L"""
        try:
            if symbol not in self.current_positions:
                return {}
            
            position = self.current_positions[symbol]
            
            # Calculate P&L
            if position['signal_type'] == 'BUY':
                pnl = (exit_price - position['entry_price']) * position['size']
            else:  # SELL
                pnl = (position['entry_price'] - exit_price) * position['size']
            
            # Calculate return percentage
            return_pct = pnl / (position['entry_price'] * position['size'])
            
            # Create trade record
            trade_record = {
                'symbol': symbol,
                'signal_type': position['signal_type'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'size': position['size'],
                'pnl': pnl,
                'return_pct': return_pct,
                'risk_score': position['risk_score'],
                'entry_time': position['entry_time'],
                'exit_time': datetime.now(),
                'hold_days': (datetime.now() - position['entry_time']).days
            }
            
            # Add to history
            self.position_history.append(trade_record)
            
            # Update portfolio value
            self.portfolio_value += pnl
            
            # Remove from current positions
            del self.current_positions[symbol]
            
            self.logger.info(f"Position closed: {symbol} P&L: ${pnl:.2f} ({return_pct:.2%})")
            
            return trade_record
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return {}
    
    def calculate_portfolio_metrics(self) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            # Calculate returns from trade history
            if len(self.position_history) < 2:
                return RiskMetrics(
                    portfolio_risk=0.0,
                    position_risk=0.0,
                    var_1day=0.0,
                    max_drawdown=0.0,
                    sharpe_ratio=0.0,
                    volatility=0.0,
                    correlation_risk=0.0
                )
            
            returns = [trade['return_pct'] for trade in self.position_history]
            returns_series = pd.Series(returns)
            
            # Portfolio risk (current exposure)
            portfolio_risk = self._calculate_current_portfolio_risk()
            
            # Position risk (average risk of current positions)
            if self.current_positions:
                position_risk = np.mean([pos['risk_score'] for pos in self.current_positions.values()])
            else:
                position_risk = 0.0
            
            # Volatility
            volatility = returns_series.std() * np.sqrt(252) if len(returns_series) > 1 else 0.0
            
            # Sharpe ratio
            mean_return = returns_series.mean() * 252  # Annualized
            if volatility > 0:
                sharpe_ratio = (mean_return - self.risk_free_rate) / volatility
            else:
                sharpe_ratio = 0.0
            
            # Maximum drawdown
            cumulative_returns = (1 + returns_series).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
            
            # Value at Risk (1 day, 95% confidence)
            if len(returns_series) > 10:
                var_1day = abs(np.percentile(returns_series, 5))
            else:
                var_1day = 0.0
            
            # Correlation risk (simplified)
            correlation_risk = min(0.5, len(self.current_positions) * 0.1)  # More positions = more correlation risk
            
            return RiskMetrics(
                portfolio_risk=portfolio_risk,
                position_risk=position_risk,
                var_1day=var_1day,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                volatility=volatility,
                correlation_risk=correlation_risk
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return RiskMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        try:
            metrics = self.calculate_portfolio_metrics()
            
            # Calculate win rate
            if self.position_history:
                winning_trades = sum(1 for trade in self.position_history if trade['pnl'] > 0)
                win_rate = winning_trades / len(self.position_history)
            else:
                win_rate = 0.0
            
            # Calculate average trade metrics
            if self.position_history:
                avg_return = np.mean([trade['return_pct'] for trade in self.position_history])
                avg_hold_days = np.mean([trade['hold_days'] for trade in self.position_history])
            else:
                avg_return = 0.0
                avg_hold_days = 0.0
            
            return {
                'portfolio_value': self.portfolio_value,
                'portfolio_risk': metrics.portfolio_risk,
                'position_risk': metrics.position_risk,
                'var_1day': metrics.var_1day,
                'max_drawdown': metrics.max_drawdown,
                'sharpe_ratio': metrics.sharpe_ratio,
                'volatility': metrics.volatility,
                'correlation_risk': metrics.correlation_risk,
                'active_positions': len(self.current_positions),
                'total_trades': len(self.position_history),
                'win_rate': win_rate,
                'avg_return': avg_return,
                'avg_hold_days': avg_hold_days,
                'risk_limits': {
                    'max_position_size': self.max_position_size,
                    'max_portfolio_risk': self.max_portfolio_risk,
                    'max_risk_score': self.max_risk_score
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting risk summary: {e}")
            return {}
    
    def check_risk_limits(self) -> List[str]:
        """Check if any risk limits are being violated"""
        warnings = []
        
        try:
            metrics = self.calculate_portfolio_metrics()
            
            # Check portfolio risk
            if metrics.portfolio_risk > self.max_portfolio_risk:
                warnings.append(f"Portfolio risk {metrics.portfolio_risk:.2%} exceeds limit {self.max_portfolio_risk:.2%}")
            
            # Check maximum drawdown
            if metrics.max_drawdown > 0.15:  # 15% max drawdown warning
                warnings.append(f"Maximum drawdown {metrics.max_drawdown:.2%} is high")
            
            # Check position concentration
            if len(self.current_positions) > 0:
                max_position_pct = max(pos['size'] * pos['current_price'] / self.portfolio_value 
                                     for pos in self.current_positions.values())
                if max_position_pct > self.max_position_size:
                    warnings.append(f"Largest position {max_position_pct:.2%} exceeds limit {self.max_position_size:.2%}")
            
            # Check volatility
            if metrics.volatility > 0.4:  # 40% annual volatility warning
                warnings.append(f"Portfolio volatility {metrics.volatility:.2%} is high")
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            warnings.append("Error checking risk limits")
        
        return warnings

# Example usage and testing
if __name__ == "__main__":
    from config_manager import ConfigManager
    import yfinance as yf
    
    # Test the risk manager
    config = ConfigManager()
    risk_manager = RiskManager(config)
    
    # Get test data
    ticker = yf.Ticker('AAPL')
    data = ticker.history(period='3mo')
    
    if not data.empty:
        print("Testing risk score calculation...")
        risk_score = risk_manager.calculate_risk_score('AAPL', data, 'BUY')
        print(f"Risk score for AAPL: {risk_score:.2f}")
        
        print("\nTesting position sizing...")
        position_size = risk_manager.calculate_position_size('AAPL', 150.0, risk_score)
        print(f"Recommended position size: {position_size:.0f} shares")
        
        print("\nTesting trade validation...")
        is_valid, reason = risk_manager.validate_trade('AAPL', 'BUY', position_size, risk_score)
        print(f"Trade validation: {is_valid} - {reason}")
        
        print("\nTesting stop loss/take profit...")
        stop_loss, take_profit = risk_manager.calculate_stop_loss_take_profit(150.0, 'BUY')
        print(f"Stop loss: ${stop_loss:.2f}, Take profit: ${take_profit:.2f}")
    
    print("Risk manager test completed.")
