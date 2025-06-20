
"""
Options Trader Module for Unified Trading Bot
=============================================

Handles options trading execution, strategy selection, and position management
with support for various options strategies and risk management.

Features:
- Multiple options strategies (long calls/puts, spreads, covered calls)
- Options chain analysis and selection
- Position sizing and risk management
- Paper trading simulation
- Real-time options pricing
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from enum import Enum

class OptionsStrategy(Enum):
    """Supported options strategies"""
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    COVERED_CALL = "covered_call"
    CASH_SECURED_PUT = "cash_secured_put"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"

@dataclass
class OptionsContract:
    """Represents an options contract"""
    symbol: str
    strike: float
    expiration: str
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    last_price: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float

@dataclass
class OptionsPosition:
    """Represents an options position"""
    contract: OptionsContract
    quantity: int
    entry_price: float
    entry_time: datetime
    strategy: OptionsStrategy
    target_profit: float
    stop_loss: float
    current_pnl: float = 0.0
    is_open: bool = True

@dataclass
class TradeResult:
    """Result of an options trade execution"""
    success: bool
    position: Optional[OptionsPosition]
    error_message: str = ""
    execution_price: float = 0.0
    commission: float = 0.0
    timestamp: datetime = None

class OptionsTrader:
    """
    Options trading execution and management
    """
    
    def __init__(self, config, dry_run: bool = True):
        self.config = config
        self.dry_run = dry_run
        self.logger = logging.getLogger(__name__)
        
        # Trading settings
        self.enabled_strategies = config.get('options.strategies', ['long_calls', 'long_puts'])
        self.min_volume = config.get('options.min_volume', 50)
        self.min_open_interest = config.get('options.min_open_interest', 100)
        self.max_bid_ask_spread = config.get('options.max_bid_ask_spread', 0.20)
        self.min_days_to_expiration = config.get('options.min_days_to_expiration', 7)
        self.max_days_to_expiration = config.get('options.max_days_to_expiration', 45)
        
        # Delta preferences (relaxed for testing)
        self.preferred_delta_min = config.get('options.preferred_delta_range.min', 0.30)
        self.preferred_delta_max = config.get('options.preferred_delta_range.max', 0.70)
        
        # Position management
        self.active_positions: Dict[str, OptionsPosition] = {}
        self.position_history: List[OptionsPosition] = []
        
        # Paper trading simulation
        self.paper_balance = config.get('paper_trading.initial_balance', 100000)
        self.paper_positions_value = 0.0
        
        self.logger.info(f"Options trader initialized (dry_run={dry_run})")
    
    async def execute_trade(self, signal, position_size: float) -> Optional[TradeResult]:
        """
        Execute an options trade based on signal
        
        Args:
            signal: Trading signal object
            position_size: Position size in dollars
            
        Returns:
            TradeResult object
        """
        try:
            # Determine strategy based on signal
            strategy = self._select_strategy(signal)
            if not strategy:
                return TradeResult(
                    success=False,
                    position=None,
                    error_message="No suitable strategy found for signal"
                )
            
            # Find suitable options contract
            contract = await self._find_optimal_contract(signal.symbol, signal.signal_type, strategy)
            if not contract:
                return TradeResult(
                    success=False,
                    position=None,
                    error_message="No suitable options contract found"
                )
            
            # Calculate quantity based on position size
            quantity = self._calculate_quantity(contract, position_size)
            if quantity <= 0:
                return TradeResult(
                    success=False,
                    position=None,
                    error_message="Position size too small for options contract"
                )
            
            # Execute the trade
            if self.dry_run:
                result = await self._execute_paper_trade(contract, quantity, strategy, signal)
            else:
                result = await self._execute_live_trade(contract, quantity, strategy, signal)
            
            # Update position tracking
            if result.success and result.position:
                self.active_positions[f"{signal.symbol}_{strategy.value}_{datetime.now().timestamp()}"] = result.position
                self.logger.info(f"Options trade executed: {strategy.value} {signal.symbol}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing options trade: {e}")
            return TradeResult(
                success=False,
                position=None,
                error_message=str(e)
            )
    
    def _select_strategy(self, signal) -> Optional[OptionsStrategy]:
        """
        Select appropriate options strategy based on signal
        
        Args:
            signal: Trading signal
            
        Returns:
            Selected options strategy
        """
        try:
            signal_type = signal.signal_type
            strength = signal.strength
            
            # Map signal to strategy based on configuration
            if signal_type == 'BUY':
                if 'long_calls' in self.enabled_strategies:
                    return OptionsStrategy.LONG_CALL
                elif 'bull_call_spread' in self.enabled_strategies and strength > 0.7:
                    return OptionsStrategy.BULL_CALL_SPREAD
                    
            elif signal_type == 'SELL':
                if 'long_puts' in self.enabled_strategies:
                    return OptionsStrategy.LONG_PUT
                elif 'bear_put_spread' in self.enabled_strategies and strength > 0.7:
                    return OptionsStrategy.BEAR_PUT_SPREAD
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error selecting strategy: {e}")
            return None
    
    async def _find_optimal_contract(self, symbol: str, signal_type: str, 
                                   strategy: OptionsStrategy) -> Optional[OptionsContract]:
        """
        Find optimal options contract for the strategy
        
        Args:
            symbol: Stock symbol
            signal_type: 'BUY' or 'SELL'
            strategy: Options strategy
            
        Returns:
            Optimal options contract or None
        """
        try:
            # This would typically fetch real options data
            # For now, we'll simulate contract selection
            
            # Get current stock price (simulated)
            current_price = 150.0  # This should come from market data
            
            # Determine option type and strike selection
            if strategy in [OptionsStrategy.LONG_CALL, OptionsStrategy.BULL_CALL_SPREAD]:
                option_type = 'call'
                # Select slightly OTM call
                strike = current_price * 1.02  # 2% OTM
            elif strategy in [OptionsStrategy.LONG_PUT, OptionsStrategy.BEAR_PUT_SPREAD]:
                option_type = 'put'
                # Select slightly OTM put
                strike = current_price * 0.98  # 2% OTM
            else:
                return None
            
            # Calculate expiration (2-3 weeks out)
            expiration_date = datetime.now() + timedelta(days=21)
            expiration_str = expiration_date.strftime('%Y-%m-%d')
            
            # Simulate contract data (in real implementation, this would come from market data)
            contract = OptionsContract(
                symbol=f"{symbol}_{expiration_str}_{option_type}_{strike}",
                strike=strike,
                expiration=expiration_str,
                option_type=option_type,
                bid=2.50,
                ask=2.60,
                last_price=2.55,
                volume=150,  # Above minimum threshold
                open_interest=500,  # Above minimum threshold
                implied_volatility=0.25,
                delta=0.45 if option_type == 'call' else -0.45,
                gamma=0.02,
                theta=-0.05,
                vega=0.15
            )
            
            # Validate contract meets criteria
            if self._validate_contract(contract):
                return contract
            else:
                self.logger.warning(f"Contract for {symbol} does not meet criteria")
                return None
                
        except Exception as e:
            self.logger.error(f"Error finding optimal contract: {e}")
            return None
    
    def _validate_contract(self, contract: OptionsContract) -> bool:
        """
        Validate that contract meets trading criteria
        
        Args:
            contract: Options contract to validate
            
        Returns:
            True if contract is valid
        """
        try:
            # Volume check
            if contract.volume < self.min_volume:
                return False
            
            # Open interest check
            if contract.open_interest < self.min_open_interest:
                return False
            
            # Bid-ask spread check
            if contract.ask > 0:
                spread_ratio = (contract.ask - contract.bid) / contract.ask
                if spread_ratio > self.max_bid_ask_spread:
                    return False
            
            # Delta check (relaxed for testing)
            abs_delta = abs(contract.delta)
            if abs_delta < self.preferred_delta_min or abs_delta > self.preferred_delta_max:
                return False
            
            # Days to expiration check
            expiration_date = datetime.strptime(contract.expiration, '%Y-%m-%d')
            days_to_expiry = (expiration_date - datetime.now()).days
            
            if days_to_expiry < self.min_days_to_expiration or days_to_expiry > self.max_days_to_expiration:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating contract: {e}")
            return False
    
    def _calculate_quantity(self, contract: OptionsContract, position_size: float) -> int:
        """
        Calculate number of contracts to trade
        
        Args:
            contract: Options contract
            position_size: Position size in dollars
            
        Returns:
            Number of contracts
        """
        try:
            # Use mid-price for calculation
            mid_price = (contract.bid + contract.ask) / 2
            
            # Each options contract represents 100 shares
            contract_value = mid_price * 100
            
            # Calculate quantity
            quantity = int(position_size / contract_value)
            
            # Ensure minimum quantity
            return max(1, quantity)
            
        except Exception as e:
            self.logger.error(f"Error calculating quantity: {e}")
            return 0
    
    async def _execute_paper_trade(self, contract: OptionsContract, quantity: int, 
                                 strategy: OptionsStrategy, signal) -> TradeResult:
        """
        Execute paper trade (simulation)
        
        Args:
            contract: Options contract
            quantity: Number of contracts
            strategy: Options strategy
            signal: Trading signal
            
        Returns:
            TradeResult
        """
        try:
            # Use mid-price for execution
            execution_price = (contract.bid + contract.ask) / 2
            
            # Calculate total cost
            total_cost = execution_price * quantity * 100  # 100 shares per contract
            commission = 1.0 * quantity  # $1 per contract
            total_cost += commission
            
            # Check if we have enough paper money
            if total_cost > self.paper_balance:
                return TradeResult(
                    success=False,
                    position=None,
                    error_message="Insufficient paper trading balance"
                )
            
            # Update paper balance
            self.paper_balance -= total_cost
            
            # Calculate target profit and stop loss
            target_profit = execution_price * 2.0  # 100% profit target
            stop_loss = execution_price * 0.5     # 50% stop loss
            
            # Create position
            position = OptionsPosition(
                contract=contract,
                quantity=quantity,
                entry_price=execution_price,
                entry_time=datetime.now(),
                strategy=strategy,
                target_profit=target_profit,
                stop_loss=stop_loss
            )
            
            self.logger.info(f"Paper trade executed: {quantity} contracts of {contract.symbol} at ${execution_price:.2f}")
            
            return TradeResult(
                success=True,
                position=position,
                execution_price=execution_price,
                commission=commission,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error in paper trade execution: {e}")
            return TradeResult(
                success=False,
                position=None,
                error_message=str(e)
            )
    
    async def _execute_live_trade(self, contract: OptionsContract, quantity: int, 
                                strategy: OptionsStrategy, signal) -> TradeResult:
        """
        Execute live trade (placeholder for real broker integration)
        
        Args:
            contract: Options contract
            quantity: Number of contracts
            strategy: Options strategy
            signal: Trading signal
            
        Returns:
            TradeResult
        """
        # This would integrate with a real broker API
        # For now, return a placeholder result
        
        self.logger.warning("Live trading not implemented - use paper trading mode")
        
        return TradeResult(
            success=False,
            position=None,
            error_message="Live trading not implemented"
        )
    
    async def update_positions(self):
        """Update all active positions with current market data"""
        try:
            for position_id, position in list(self.active_positions.items()):
                if position.is_open:
                    # Get current option price (simulated)
                    current_price = await self._get_current_option_price(position.contract)
                    
                    if current_price:
                        # Update P&L
                        position.current_pnl = (current_price - position.entry_price) * position.quantity * 100
                        
                        # Check exit conditions
                        if await self._should_exit_position(position, current_price):
                            await self._close_position(position_id, position, current_price)
                            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    async def _get_current_option_price(self, contract: OptionsContract) -> Optional[float]:
        """Get current option price (simulated)"""
        try:
            # In real implementation, this would fetch current market data
            # For simulation, we'll add some random movement
            import random
            
            base_price = (contract.bid + contract.ask) / 2
            # Add random movement (-10% to +10%)
            movement = random.uniform(-0.1, 0.1)
            current_price = base_price * (1 + movement)
            
            return max(0.01, current_price)  # Minimum price of $0.01
            
        except Exception as e:
            self.logger.error(f"Error getting current option price: {e}")
            return None
    
    async def _should_exit_position(self, position: OptionsPosition, current_price: float) -> bool:
        """Determine if position should be closed"""
        try:
            # Check profit target
            if current_price >= position.target_profit:
                self.logger.info(f"Position {position.contract.symbol} hit profit target")
                return True
            
            # Check stop loss
            if current_price <= position.stop_loss:
                self.logger.info(f"Position {position.contract.symbol} hit stop loss")
                return True
            
            # Check time decay (close if < 7 days to expiration)
            expiration_date = datetime.strptime(position.contract.expiration, '%Y-%m-%d')
            days_to_expiry = (expiration_date - datetime.now()).days
            
            if days_to_expiry < 7:
                self.logger.info(f"Position {position.contract.symbol} approaching expiration")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
            return False
    
    async def _close_position(self, position_id: str, position: OptionsPosition, current_price: float):
        """Close an options position"""
        try:
            # Calculate final P&L
            final_pnl = (current_price - position.entry_price) * position.quantity * 100
            commission = 1.0 * position.quantity  # $1 per contract
            net_pnl = final_pnl - commission
            
            # Update paper balance if in paper trading mode
            if self.dry_run:
                proceeds = current_price * position.quantity * 100 - commission
                self.paper_balance += proceeds
            
            # Mark position as closed
            position.is_open = False
            position.current_pnl = net_pnl
            
            # Move to history
            self.position_history.append(position)
            del self.active_positions[position_id]
            
            self.logger.info(f"Position closed: {position.contract.symbol} P&L: ${net_pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        try:
            total_positions = len(self.active_positions)
            total_pnl = sum(pos.current_pnl for pos in self.active_positions.values())
            
            # Calculate win rate from history
            closed_positions = len(self.position_history)
            winning_positions = sum(1 for pos in self.position_history if pos.current_pnl > 0)
            win_rate = winning_positions / closed_positions if closed_positions > 0 else 0
            
            return {
                'paper_balance': self.paper_balance,
                'active_positions': total_positions,
                'unrealized_pnl': total_pnl,
                'total_trades': closed_positions,
                'win_rate': win_rate,
                'positions': [
                    {
                        'symbol': pos.contract.symbol,
                        'strategy': pos.strategy.value,
                        'quantity': pos.quantity,
                        'entry_price': pos.entry_price,
                        'current_pnl': pos.current_pnl,
                        'days_held': (datetime.now() - pos.entry_time).days
                    }
                    for pos in self.active_positions.values()
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    from config_manager import ConfigManager
    from unified_trading_bot import TradingSignal
    
    # Test the options trader
    config = ConfigManager()
    trader = OptionsTrader(config, dry_run=True)
    
    # Create a test signal
    test_signal = TradingSignal(
        symbol='AAPL',
        signal_type='BUY',
        strategy='OPTIONS',
        strength=0.7,
        price=150.0,
        timestamp=datetime.now(),
        indicators={},
        confidence=0.8,
        risk_score=0.3
    )
    
    async def test_trading():
        print("Testing options trading...")
        
        # Execute test trade
        result = await trader.execute_trade(test_signal, 1000)  # $1000 position
        
        if result.success:
            print(f"Trade executed successfully: {result.position.contract.symbol}")
            print(f"Execution price: ${result.execution_price:.2f}")
        else:
            print(f"Trade failed: {result.error_message}")
        
        # Get portfolio summary
        summary = trader.get_portfolio_summary()
        print(f"Portfolio summary: {summary}")
    
    # Run the test
    import asyncio
    asyncio.run(test_trading())
    
    print("Options trader test completed.")
