"""
Strategy Engine Base Classes
Defines abstract interfaces for backtest and live trading engines
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from decimal import Decimal
from django.utils import timezone
import pandas as pd


class BaseBacktestEngine(ABC):
    """
    Abstract base class for backtest engines
    
    Each strategy should implement this interface to define how it executes
    in a backtest environment (using historical data).
    """
    
    def __init__(self, strategy, parameters: Dict, symbols: List, position_mode: str = 'all'):
        """
        Initialize the backtest engine
        
        Args:
            strategy: StrategyDefinition instance
            parameters: Strategy parameters dict
            symbols: List of Symbol instances
            position_mode: 'all', 'long', or 'short'
        """
        self.strategy = strategy
        self.parameters = parameters
        self.symbols = symbols
        self.position_mode = position_mode
    
    @abstractmethod
    def load_data(self, start_date=None, end_date=None) -> Dict:
        """
        Load historical OHLCV data for all symbols
        
        Args:
            start_date: Optional start date for data filtering
            end_date: Optional end date for data filtering
        
        Returns:
            Dict mapping Symbol -> DataFrame with OHLCV data
        """
        pass
    
    @abstractmethod
    def compute_indicators(self, ohlcv_data: Dict) -> Dict:
        """
        Compute required indicators for all symbols
        
        Args:
            ohlcv_data: Dict mapping Symbol -> DataFrame
        
        Returns:
            Dict mapping Symbol -> Dict of indicator_name -> Series
        """
        pass
    
    @abstractmethod
    def execute_strategy(
        self,
        ohlcv_data: Dict,
        indicators: Dict,
        initial_capital: Decimal,
        bet_size_percentage: float
    ) -> Dict:
        """
        Execute the strategy and generate trades
        
        Args:
            ohlcv_data: Dict mapping Symbol -> DataFrame
            indicators: Dict mapping Symbol -> Dict of indicators
            initial_capital: Starting capital
            bet_size_percentage: Percentage of capital to bet per trade
        
        Returns:
            Dict with:
            - 'trades': List of trade dicts
            - 'equity_curve': List of (timestamp, equity) tuples
            - 'statistics': Dict of calculated statistics
        """
        pass
    
    @abstractmethod
    def generate_signal(
        self,
        symbol,
        current_row: pd.Series,
        indicators: Dict,
        position: Optional[Dict]
    ) -> Optional[str]:
        """
        Generate trading signal for current bar
        
        Args:
            symbol: Symbol instance
            current_row: Current OHLCV row as Series
            indicators: Dict of indicator_name -> Series or value
            position: Current position dict or None
        
        Returns:
            'buy', 'sell', or None
        """
        pass


class BaseLiveTradingEngine(ABC):
    """
    Abstract base class for live trading engines
    
    Each strategy should implement this interface to define how it executes
    in a live trading environment (using real-time data and broker APIs).
    """
    
    def __init__(
        self,
        strategy,
        parameters: Dict,
        deployment,
        broker_adapter,
        position_mode: str = 'all'
    ):
        """
        Initialize the live trading engine
        
        Args:
            strategy: StrategyDefinition instance
            parameters: Strategy parameters dict
            deployment: LiveTradingDeployment instance
            broker_adapter: BrokerAdapter instance for executing trades
            position_mode: 'all', 'long', or 'short'
        """
        self.strategy = strategy
        self.parameters = parameters
        self.deployment = deployment
        self.broker_adapter = broker_adapter
        self.position_mode = position_mode
    
    @abstractmethod
    def initialize(self):
        """Initialize the engine (load historical data, compute indicators, etc.)"""
        pass
    
    @abstractmethod
    def process_market_data(self, symbol, market_data: Dict) -> Optional[Dict]:
        """
        Process new market data and potentially generate trades
        
        Args:
            symbol: Symbol instance
            market_data: Dict with OHLCV data and timestamp
        
        Returns:
            Trade execution dict if trade was executed, None otherwise
        """
        pass
    
    @abstractmethod
    def generate_signal(
        self,
        symbol,
        market_data: Dict,
        indicators: Dict,
        position: Optional[Dict]
    ) -> Optional[str]:
        """
        Generate trading signal from current market data
        
        Args:
            symbol: Symbol instance
            market_data: Dict with OHLCV data
            indicators: Dict of indicator_name -> value
            position: Current position dict or None
        
        Returns:
            'buy', 'sell', or None
        """
        pass
    
    @abstractmethod
    def execute_trade(
        self,
        symbol,
        signal: str,
        price: Decimal,
        quantity: Decimal
    ) -> Dict:
        """
        Execute a trade via the broker adapter
        
        Args:
            symbol: Symbol instance
            signal: 'buy' or 'sell'
            price: Execution price
            quantity: Quantity to trade
        
        Returns:
            Dict with trade execution details including broker_order_id
        """
        pass
    
    @abstractmethod
    def update_indicators(self, symbol, market_data: Dict):
        """
        Update indicators with new market data
        
        Args:
            symbol: Symbol instance
            market_data: Dict with OHLCV data
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict:
        """
        Get current positions from broker
        
        Returns:
            Dict mapping symbol -> position dict
        """
        pass
    
    @abstractmethod
    def shutdown(self):
        """Cleanup and shutdown the engine"""
        pass


class EngineRegistry:
    """
    Registry for mapping strategies to their engine implementations
    
    Strategies can register their backtest and/or live trading engines here.
    """
    
    _backtest_engines = {}  # {strategy_name: BacktestEngineClass}
    _live_trading_engines = {}  # {strategy_name: LiveTradingEngineClass}
    
    @classmethod
    def register_backtest_engine(cls, strategy_name: str, engine_class: type):
        """Register a backtest engine for a strategy"""
        if not issubclass(engine_class, BaseBacktestEngine):
            raise TypeError(f"Engine must inherit from BaseBacktestEngine")
        cls._backtest_engines[strategy_name] = engine_class
        return engine_class
    
    @classmethod
    def register_live_trading_engine(cls, strategy_name: str, engine_class: type):
        """Register a live trading engine for a strategy"""
        if not issubclass(engine_class, BaseLiveTradingEngine):
            raise TypeError(f"Engine must inherit from BaseLiveTradingEngine")
        cls._live_trading_engines[strategy_name] = engine_class
        return engine_class
    
    @classmethod
    def get_backtest_engine(cls, strategy_name: str) -> Optional[type]:
        """Get backtest engine class for a strategy"""
        return cls._backtest_engines.get(strategy_name)
    
    @classmethod
    def get_live_trading_engine(cls, strategy_name: str) -> Optional[type]:
        """Get live trading engine class for a strategy"""
        return cls._live_trading_engines.get(strategy_name)
    
    @classmethod
    def has_backtest_engine(cls, strategy_name: str) -> bool:
        """Check if strategy has a registered backtest engine"""
        return strategy_name in cls._backtest_engines
    
    @classmethod
    def has_live_trading_engine(cls, strategy_name: str) -> bool:
        """Check if strategy has a registered live trading engine"""
        return strategy_name in cls._live_trading_engines
    
    @classmethod
    def get_all_strategies_with_backtest_engines(cls) -> List[str]:
        """Get list of all strategy names that have backtest engines"""
        return list(cls._backtest_engines.keys())
    
    @classmethod
    def get_all_strategies_with_live_trading_engines(cls) -> List[str]:
        """Get list of all strategy names that have live trading engines"""
        return list(cls._live_trading_engines.keys())


# Decorator shortcuts for registering engines
def backtest_engine(strategy_name: str):
    """Decorator to register a backtest engine"""
    def decorator(engine_class):
        EngineRegistry.register_backtest_engine(strategy_name, engine_class)
        return engine_class
    return decorator


def live_trading_engine(strategy_name: str):
    """Decorator to register a live trading engine"""
    def decorator(engine_class):
        EngineRegistry.register_live_trading_engine(strategy_name, engine_class)
        return engine_class
    return decorator


