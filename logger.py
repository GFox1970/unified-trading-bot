
"""
Enhanced Logging Module for Unified Trading Bot
===============================================

Provides comprehensive logging functionality with structured logging,
performance tracking, and trade signal logging.

Features:
- Rich console output with colors and formatting
- Structured logging for better analysis
- Trade signal logging with detailed information
- Error logging with stack traces
- Performance metrics logging
- Log rotation and file management
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

# Install rich traceback handler
install(show_locals=True)

console = Console()

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""
    
    def format(self, record):
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)
        
        return json.dumps(log_entry)

class TradingBotLogger:
    """Enhanced logger for trading bot operations"""
    
    def __init__(self, name: str, level: str = 'INFO', log_file: str = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler with rich formatting
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True
        )
        console_handler.setLevel(getattr(logging, level.upper()))
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with structured logging
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(StructuredFormatter())
            self.logger.addHandler(file_handler)
        
        # Trade signals log
        if log_file:
            signals_log = log_path.parent / 'trade_signals.log'
            self.signals_handler = logging.handlers.RotatingFileHandler(
                signals_log,
                maxBytes=5*1024*1024,  # 5MB
                backupCount=3
            )
            self.signals_handler.setLevel(logging.INFO)
            self.signals_handler.setFormatter(StructuredFormatter())
        else:
            self.signals_handler = None
    
    def info(self, message: str, extra_data: Dict[str, Any] = None):
        """Log info message with optional extra data"""
        self._log_with_extra(logging.INFO, message, extra_data)
    
    def debug(self, message: str, extra_data: Dict[str, Any] = None):
        """Log debug message with optional extra data"""
        self._log_with_extra(logging.DEBUG, message, extra_data)
    
    def warning(self, message: str, extra_data: Dict[str, Any] = None):
        """Log warning message with optional extra data"""
        self._log_with_extra(logging.WARNING, message, extra_data)
    
    def error(self, message: str, extra_data: Dict[str, Any] = None):
        """Log error message with optional extra data"""
        self._log_with_extra(logging.ERROR, message, extra_data)
    
    def critical(self, message: str, extra_data: Dict[str, Any] = None):
        """Log critical message with optional extra data"""
        self._log_with_extra(logging.CRITICAL, message, extra_data)
    
    def _log_with_extra(self, level: int, message: str, extra_data: Dict[str, Any] = None):
        """Internal method to log with extra data"""
        if extra_data:
            # Create a custom LogRecord with extra data
            record = self.logger.makeRecord(
                self.logger.name, level, __file__, 0, message, (), None
            )
            record.extra_data = extra_data
            self.logger.handle(record)
        else:
            self.logger.log(level, message)
    
    def log_trade_signal(self, signal):
        """Log a trade signal with detailed information"""
        if self.signals_handler:
            signal_data = {
                'event_type': 'trade_signal',
                'symbol': signal.symbol,
                'signal_type': signal.signal_type,
                'strategy': signal.strategy,
                'strength': signal.strength,
                'price': signal.price,
                'confidence': signal.confidence,
                'risk_score': signal.risk_score,
                'indicators': signal.indicators,
                'timestamp': signal.timestamp.isoformat()
            }
            
            # Create log record for signals
            record = logging.LogRecord(
                name='trade_signals',
                level=logging.INFO,
                pathname=__file__,
                lineno=0,
                msg='Trade signal generated',
                args=(),
                exc_info=None
            )
            record.extra_data = signal_data
            self.signals_handler.emit(record)
        
        # Also log to main logger
        self.info(
            f"Trade Signal: {signal.signal_type} {signal.symbol} "
            f"(Strength: {signal.strength:.2f}, Strategy: {signal.strategy})",
            extra_data={'signal_data': signal_data}
        )
    
    def log_trade_execution(self, symbol: str, action: str, quantity: float, 
                          price: float, success: bool, details: Dict[str, Any] = None):
        """Log trade execution details"""
        execution_data = {
            'event_type': 'trade_execution',
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        if details:
            execution_data.update(details)
        
        level = logging.INFO if success else logging.ERROR
        message = f"Trade {'Executed' if success else 'Failed'}: {action} {quantity} {symbol} @ ${price:.2f}"
        
        self._log_with_extra(level, message, execution_data)
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics"""
        performance_data = {
            'event_type': 'performance_metrics',
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        self.info("Performance metrics updated", extra_data=performance_data)
    
    def log_error_with_context(self, error: Exception, context: str, 
                              extra_data: Dict[str, Any] = None):
        """Log error with full context and stack trace"""
        import traceback
        
        error_data = {
            'event_type': 'error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'stack_trace': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }
        
        if extra_data:
            error_data.update(extra_data)
        
        self.error(f"Error in {context}: {error}", extra_data=error_data)

def setup_logger(name: str, level: str = 'INFO', log_file: str = None) -> TradingBotLogger:
    """
    Setup and return a configured logger instance
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    
    Returns:
        Configured TradingBotLogger instance
    """
    return TradingBotLogger(name, level, log_file)

def log_trade_signal(logger: TradingBotLogger, signal):
    """Convenience function to log trade signals"""
    logger.log_trade_signal(signal)

def log_error(logger: TradingBotLogger, error: Exception, context: str, 
              extra_data: Dict[str, Any] = None):
    """Convenience function to log errors with context"""
    logger.log_error_with_context(error, context, extra_data)

def log_performance(logger: TradingBotLogger, metrics: Dict[str, Any]):
    """Convenience function to log performance metrics"""
    logger.log_performance_metrics(metrics)

# Example usage and testing
if __name__ == "__main__":
    # Test the logger
    logger = setup_logger("test_logger", "DEBUG", "test.log")
    
    logger.info("Test info message", {"test_data": "example"})
    logger.debug("Test debug message")
    logger.warning("Test warning message")
    logger.error("Test error message")
    
    try:
        raise ValueError("Test exception")
    except Exception as e:
        log_error(logger, e, "Testing error logging")
    
    print("Logger test completed. Check test.log and trade_signals.log files.")
