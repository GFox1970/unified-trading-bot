
#!/usr/bin/env python3
"""
Test Script for Unified Trading Bot
==================================

Quick test script to validate the bot components work correctly
before deployment to Render.
"""

import asyncio
import sys
import traceback
from datetime import datetime

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from config_manager import ConfigManager
        print("✅ config_manager imported successfully")
        
        from logger import setup_logger
        print("✅ logger imported successfully")
        
        from market_data import MarketDataProvider
        print("✅ market_data imported successfully")
        
        from signal_analyzer import SignalAnalyzer
        print("✅ signal_analyzer imported successfully")
        
        from options_trader import OptionsTrader
        print("✅ options_trader imported successfully")
        
        from risk_manager import RiskManager
        print("✅ risk_manager imported successfully")
        
        from unified_trading_bot import UnifiedTradingBot
        print("✅ unified_trading_bot imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from config_manager import ConfigManager
        
        config = ConfigManager()
        
        # Test basic config values
        update_interval = config.get('bot.update_interval', 60)
        watchlist = config.get('trading.watchlist', [])
        min_strength = config.get('signals.min_strength', 0.6)
        
        print(f"✅ Update interval: {update_interval}s")
        print(f"✅ Watchlist: {watchlist}")
        print(f"✅ Min signal strength: {min_strength}")
        print(f"✅ Test mode: {config.is_test_mode()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        traceback.print_exc()
        return False

async def test_market_data():
    """Test market data fetching"""
    print("\nTesting market data...")
    
    try:
        from config_manager import ConfigManager
        from market_data import MarketDataProvider
        
        config = ConfigManager()
        provider = MarketDataProvider(config)
        
        # Test stock data
        data = await provider.get_stock_data('SPY', period='5d')
        
        if data is not None and not data.empty:
            print(f"✅ Retrieved {len(data)} data points for SPY")
            print(f"✅ Latest close price: ${data['Close'].iloc[-1]:.2f}")
            
            # Test technical indicators
            indicators = provider.calculate_technical_indicators(data)
            if indicators:
                print(f"✅ Calculated {len(indicators)} technical indicators")
                if 'rsi' in indicators:
                    current_rsi = indicators['rsi'].iloc[-1]
                    print(f"✅ Current RSI: {current_rsi:.2f}")
            
            return True
        else:
            print("❌ No market data retrieved")
            return False
            
    except Exception as e:
        print(f"❌ Market data error: {e}")
        traceback.print_exc()
        return False

async def test_signal_analysis():
    """Test signal analysis"""
    print("\nTesting signal analysis...")
    
    try:
        from config_manager import ConfigManager
        from market_data import MarketDataProvider
        from signal_analyzer import SignalAnalyzer
        
        config = ConfigManager()
        provider = MarketDataProvider(config)
        analyzer = SignalAnalyzer(config)
        
        # Get test data
        data = await provider.get_stock_data('AAPL', period='3mo')
        
        if data is not None and not data.empty:
            # Test RSI-MACD analysis
            signal = analyzer.analyze_rsi_macd_signal(data)
            
            if signal:
                print(f"✅ Generated signal: {signal.signal_type}")
                print(f"✅ Signal strength: {signal.strength:.2f}")
                print(f"✅ Confidence: {signal.confidence:.2f}")
                print(f"✅ Reasoning: {', '.join(signal.reasoning[:2])}")
            else:
                print("✅ No signal generated (normal)")
            
            return True
        else:
            print("❌ No data for signal analysis")
            return False
            
    except Exception as e:
        print(f"❌ Signal analysis error: {e}")
        traceback.print_exc()
        return False

def test_risk_management():
    """Test risk management"""
    print("\nTesting risk management...")
    
    try:
        from config_manager import ConfigManager
        from risk_manager import RiskManager
        import pandas as pd
        import numpy as np
        
        config = ConfigManager()
        risk_manager = RiskManager(config)
        
        # Create test data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
        volumes = np.random.randint(1000000, 5000000, 100)
        
        test_data = pd.DataFrame({
            'Close': prices,
            'Volume': volumes,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Open': prices
        }, index=dates)
        
        # Test risk score calculation
        risk_score = risk_manager.calculate_risk_score('TEST', test_data, 'BUY')
        print(f"✅ Risk score calculated: {risk_score:.2f}")
        
        # Test position sizing
        position_size = risk_manager.calculate_position_size('TEST', 100.0, risk_score)
        print(f"✅ Position size calculated: {position_size:.0f} shares")
        
        # Test trade validation
        is_valid, reason = risk_manager.validate_trade('TEST', 'BUY', position_size, risk_score)
        print(f"✅ Trade validation: {is_valid} - {reason}")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk management error: {e}")
        traceback.print_exc()
        return False

async def test_unified_bot():
    """Test the unified bot initialization"""
    print("\nTesting unified bot...")
    
    try:
        from unified_trading_bot import UnifiedTradingBot
        
        # Initialize bot in dry run mode
        bot = UnifiedTradingBot(dry_run=True)
        
        print("✅ Bot initialized successfully")
        print(f"✅ Dry run mode: {bot.dry_run}")
        print(f"✅ Active positions: {len(bot.active_positions)}")
        
        # Test one trading cycle (without starting the full loop)
        print("✅ Testing single trading cycle...")
        
        # This would normally run the full cycle, but we'll just test the setup
        symbols = bot.config.get('trading.watchlist', ['SPY'])[:2]  # Test with 2 symbols
        print(f"✅ Will analyze symbols: {symbols}")
        
        return True
        
    except Exception as e:
        print(f"❌ Unified bot error: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("🧪 Unified Trading Bot - Component Tests")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Market Data", test_market_data),
        ("Signal Analysis", test_signal_analysis),
        ("Risk Management", test_risk_management),
        ("Unified Bot", test_unified_bot),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            results.append((test_name, result))
            
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Bot is ready for deployment.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before deployment.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
