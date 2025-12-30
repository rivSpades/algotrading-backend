"""
Base Broker Adapter Interface
Defines the interface that all broker adapters must implement
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime


@dataclass
class OrderResult:
    """Result of an order execution"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: Decimal
    filled_quantity: Decimal
    price: Decimal
    status: str  # 'filled', 'partial', 'rejected', 'pending'
    timestamp: datetime
    broker_order_id: Optional[str] = None
    fees: Optional[Decimal] = None
    error_message: Optional[str] = None


@dataclass
class PositionInfo:
    """Information about a position"""
    symbol: str
    quantity: Decimal
    average_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    position_type: str  # 'long' or 'short'
    timestamp: datetime


class BaseBrokerAdapter(ABC):
    """
    Abstract base class for broker API adapters
    
    Each broker (Alpaca, Interactive Brokers, etc.) should implement this interface
    to provide a unified API for trade execution.
    """
    
    def __init__(self, broker, paper_trading: bool = True):
        """
        Initialize the broker adapter
        
        Args:
            broker: Broker model instance
            paper_trading: Whether to use paper trading credentials (True) or real money (False)
        """
        self.broker = broker
        self.paper_trading = paper_trading
        
        if paper_trading:
            if not broker.paper_trading_endpoint_url or not broker.paper_trading_api_key or not broker.paper_trading_secret_key:
                raise ValueError("Broker must have paper_trading_endpoint_url, paper_trading_api_key, and paper_trading_secret_key configured")
            self.endpoint_url = broker.paper_trading_endpoint_url
            self.api_key = broker.paper_trading_api_key
            self.api_secret = broker.paper_trading_secret_key
        else:
            if not broker.real_money_endpoint_url or not broker.real_money_api_key or not broker.real_money_secret_key:
                raise ValueError("Broker must have real_money_endpoint_url, real_money_api_key, and real_money_secret_key configured")
            self.endpoint_url = broker.real_money_endpoint_url
            self.api_key = broker.real_money_api_key
            self.api_secret = broker.real_money_secret_key
        
        self.config = broker.api_config or {}
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the broker API
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from the broker API"""
        pass
    
    @abstractmethod
    def verify_credentials(self) -> bool:
        """
        Verify API credentials are valid
        
        Returns:
            True if credentials are valid, False otherwise
        """
        pass
    
    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        order_type: str = 'market',
        limit_price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None
    ) -> OrderResult:
        """
        Place an order
        
        Args:
            symbol: Symbol ticker
            side: 'buy' or 'sell'
            quantity: Quantity to trade
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            limit_price: Limit price (required for limit orders)
            stop_price: Stop price (required for stop orders)
        
        Returns:
            OrderResult instance
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
        
        Returns:
            True if cancellation successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderResult:
        """
        Get status of an order
        
        Args:
            order_id: Order ID
        
        Returns:
            OrderResult instance with current status
        """
        pass
    
    @abstractmethod
    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """
        Get current position for a symbol
        
        Args:
            symbol: Symbol ticker
        
        Returns:
            PositionInfo instance or None if no position
        """
        pass
    
    @abstractmethod
    def get_all_positions(self) -> List[PositionInfo]:
        """
        Get all current positions
        
        Returns:
            List of PositionInfo instances
        """
        pass
    
    @abstractmethod
    def get_account_balance(self) -> Decimal:
        """
        Get account balance (cash available for trading)
        
        Returns:
            Account balance as Decimal
        """
        pass
    
    @abstractmethod
    def get_account_equity(self) -> Decimal:
        """
        Get total account equity (cash + positions value)
        
        Returns:
            Total equity as Decimal
        """
        pass
    
    @abstractmethod
    def is_symbol_tradable(self, symbol: str) -> bool:
        """
        Check if a symbol is tradable on this broker
        
        Args:
            symbol: Symbol ticker
        
        Returns:
            True if symbol is tradable, False otherwise
        """
        pass
    
    @abstractmethod
    def get_symbol_capabilities(self, symbol: str) -> Dict:
        """
        Get trading capabilities for a symbol (long/short support, etc.)
        
        Args:
            symbol: Symbol ticker
        
        Returns:
            Dict with:
            - 'long_supported': bool
            - 'short_supported': bool
            - 'min_order_size': Decimal
            - 'max_order_size': Decimal
            - Other broker-specific capabilities
        """
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get current market price for a symbol
        
        Args:
            symbol: Symbol ticker
        
        Returns:
            Current price or None if unavailable
        """
        pass
    
    @abstractmethod
    def get_market_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: str = '1min'
    ) -> List[Dict]:
        """
        Get historical market data
        
        Args:
            symbol: Symbol ticker
            start_date: Start date (optional)
            end_date: End date (optional)
            timeframe: '1min', '5min', '15min', '1hour', '1day', etc.
        
        Returns:
            List of dicts with OHLCV data
        """
        pass


