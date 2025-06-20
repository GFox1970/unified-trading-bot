
"""
Configuration Manager for Unified Trading Bot
=============================================

Handles loading, validation, and management of configuration settings
from YAML files with environment variable overrides and validation.

Features:
- YAML configuration file loading
- Environment variable overrides
- Configuration validation
- Default value handling
- Dynamic configuration updates
- Configuration schema validation
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
import logging
from dataclasses import dataclass, field

@dataclass
class ConfigSchema:
    """Configuration schema definition for validation"""
    
    # Required fields
    required_fields: List[str] = field(default_factory=lambda: [
        'bot.update_interval',
        'trading.watchlist',
        'signals.min_strength',
        'risk.max_position_size'
    ])
    
    # Field types
    field_types: Dict[str, type] = field(default_factory=lambda: {
        'bot.update_interval': int,
        'bot.max_concurrent_trades': int,
        'signals.min_strength': float,
        'signals.min_confidence': float,
        'risk.max_position_size': float,
        'risk.max_portfolio_risk': float,
        'trading.watchlist': list,
        'strategies.rsi_macd.enabled': bool,
        'strategies.options.enabled': bool
    })
    
    # Value ranges
    value_ranges: Dict[str, Dict[str, Union[int, float]]] = field(default_factory=lambda: {
        'bot.update_interval': {'min': 10, 'max': 3600},
        'signals.min_strength': {'min': 0.0, 'max': 1.0},
        'signals.min_confidence': {'min': 0.0, 'max': 1.0},
        'risk.max_position_size': {'min': 0.001, 'max': 1.0},
        'risk.max_portfolio_risk': {'min': 0.01, 'max': 1.0}
    })

class ConfigManager:
    """
    Configuration manager for the trading bot
    
    Handles loading configuration from YAML files, environment variables,
    and provides validation and default value management.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path
        self.config_data = {}
        self.schema = ConfigSchema()
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self._load_config()
        self._apply_env_overrides()
        self._validate_config()
    
    def _load_config(self):
        """Load configuration from file"""
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    self.config_data = yaml.safe_load(f) or {}
                self.logger.info(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                self.logger.error(f"Error loading config file {self.config_path}: {e}")
                self.config_data = {}
        else:
            # Load default configuration
            self.config_data = self._get_default_config()
            self.logger.info("Using default configuration")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'bot': {
                'update_interval': 60,
                'max_concurrent_trades': 5,
                'timezone': 'US/Eastern'
            },
            'logging': {
                'level': 'INFO',
                'file': 'trading_bot.log',
                'console_output': True
            },
            'trading': {
                'watchlist': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA'],
                'market_hours_only': True,
                'extended_hours': False
            },
            'strategies': {
                'rsi_macd': {
                    'enabled': True,
                    'weight': 0.6
                },
                'options': {
                    'enabled': True,
                    'weight': 0.4
                }
            },
            'indicators': {
                'rsi': {
                    'period': 14,
                    'oversold_threshold': 30,
                    'overbought_threshold': 70
                },
                'macd': {
                    'fast_period': 12,
                    'slow_period': 26,
                    'signal_period': 9
                }
            },
            'signals': {
                'min_strength': 0.3,  # Lowered for testing
                'min_confidence': 0.5,  # Lowered for testing
                'max_concurrent': 5
            },
            'risk': {
                'max_position_size': 0.05,
                'max_portfolio_risk': 0.20,
                'max_risk_score': 0.8,  # Lowered for testing
                'stop_loss_percentage': 0.02,
                'take_profit_percentage': 0.06
            },
            'options': {
                'enabled': True,
                'min_volume': 50,  # Lowered for testing
                'min_open_interest': 100,  # Lowered for testing
                'max_bid_ask_spread': 0.20,  # Increased for testing
                'min_days_to_expiration': 7,  # Lowered for testing
                'max_days_to_expiration': 45
            },
            'market_data': {
                'provider': 'yfinance',
                'update_frequency': 60,
                'history_days': 252,
                'rate_limit': 5,
                'timeout': 30
            },
            'notifications': {
                'enabled': True,
                'channels': ['console', 'file']
            },
            'development': {
                'debug_mode': False,
                'test_mode': True,  # Enables relaxed thresholds
                'mock_market_data': False
            }
        }
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        env_prefix = 'TRADING_BOT_'
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                # Convert environment variable name to config path
                config_key = key[len(env_prefix):].lower().replace('_', '.')
                
                # Convert value to appropriate type
                converted_value = self._convert_env_value(value)
                
                # Set the configuration value
                self._set_nested_value(self.config_data, config_key, converted_value)
                self.logger.info(f"Environment override: {config_key} = {converted_value}")
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type"""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Number conversion
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # List conversion (comma-separated)
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        # String value
        return value
    
    def _set_nested_value(self, data: Dict[str, Any], key_path: str, value: Any):
        """Set a nested dictionary value using dot notation"""
        keys = key_path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _validate_config(self):
        """Validate configuration against schema"""
        errors = []
        
        # Check required fields
        for field in self.schema.required_fields:
            if not self._has_nested_key(self.config_data, field):
                errors.append(f"Missing required field: {field}")
        
        # Check field types
        for field, expected_type in self.schema.field_types.items():
            if self._has_nested_key(self.config_data, field):
                value = self.get(field)
                if not isinstance(value, expected_type):
                    errors.append(f"Field {field} should be {expected_type.__name__}, got {type(value).__name__}")
        
        # Check value ranges
        for field, range_config in self.schema.value_ranges.items():
            if self._has_nested_key(self.config_data, field):
                value = self.get(field)
                if isinstance(value, (int, float)):
                    if 'min' in range_config and value < range_config['min']:
                        errors.append(f"Field {field} value {value} is below minimum {range_config['min']}")
                    if 'max' in range_config and value > range_config['max']:
                        errors.append(f"Field {field} value {value} is above maximum {range_config['max']}")
        
        if errors:
            error_msg = "Configuration validation errors:\n" + "\n".join(f"  - {error}" for error in errors)
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info("Configuration validation passed")
    
    def _has_nested_key(self, data: Dict[str, Any], key_path: str) -> bool:
        """Check if nested key exists using dot notation"""
        keys = key_path.split('.')
        current = data
        
        try:
            for key in keys:
                current = current[key]
            return True
        except (KeyError, TypeError):
            return False
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to configuration value
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        current = self.config_data
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to configuration value
            value: Value to set
        """
        self._set_nested_value(self.config_data, key_path, value)
        self.logger.debug(f"Configuration updated: {key_path} = {value}")
    
    def update(self, updates: Dict[str, Any]):
        """
        Update multiple configuration values
        
        Args:
            updates: Dictionary of key-value pairs to update
        """
        for key_path, value in updates.items():
            self.set(key_path, value)
    
    def save(self, file_path: Optional[str] = None):
        """
        Save configuration to file
        
        Args:
            file_path: Path to save configuration (uses original path if not provided)
        """
        save_path = file_path or self.config_path
        if not save_path:
            raise ValueError("No file path provided for saving configuration")
        
        try:
            with open(save_path, 'w') as f:
                yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
            self.logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Error saving configuration to {save_path}: {e}")
            raise
    
    def reload(self):
        """Reload configuration from file"""
        self._load_config()
        self._apply_env_overrides()
        self._validate_config()
        self.logger.info("Configuration reloaded")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration data"""
        return self.config_data.copy()
    
    def is_test_mode(self) -> bool:
        """Check if running in test mode"""
        return self.get('development.test_mode', False)
    
    def is_debug_mode(self) -> bool:
        """Check if running in debug mode"""
        return self.get('development.debug_mode', False)
    
    def get_watchlist(self) -> List[str]:
        """Get trading watchlist"""
        return self.get('trading.watchlist', [])
    
    def get_signal_thresholds(self) -> Dict[str, float]:
        """Get signal threshold configuration"""
        return {
            'min_strength': self.get('signals.min_strength', 0.6),
            'min_confidence': self.get('signals.min_confidence', 0.7),
            'max_risk_score': self.get('risk.max_risk_score', 0.6)
        }
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"ConfigManager(config_path={self.config_path}, keys={len(self.config_data)})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"ConfigManager(config_path='{self.config_path}', config_data={self.config_data})"

# Example usage and testing
if __name__ == "__main__":
    # Test the configuration manager
    config = ConfigManager()
    
    print("Default configuration loaded:")
    print(f"Update interval: {config.get('bot.update_interval')}")
    print(f"Watchlist: {config.get('trading.watchlist')}")
    print(f"Min signal strength: {config.get('signals.min_strength')}")
    print(f"Test mode: {config.is_test_mode()}")
    
    # Test environment variable override
    os.environ['TRADING_BOT_SIGNALS_MIN_STRENGTH'] = '0.8'
    config.reload()
    print(f"After env override - Min signal strength: {config.get('signals.min_strength')}")
    
    print("Configuration manager test completed.")
