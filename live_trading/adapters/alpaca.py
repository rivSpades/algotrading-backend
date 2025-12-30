"""
Alpaca Broker Adapter
Implements BaseBrokerAdapter for Alpaca API
"""

import requests
from typing import Optional, Dict, List
from decimal import Decimal
from datetime import datetime

from .base import BaseBrokerAdapter, OrderResult, PositionInfo
from ..models import Broker


class AlpacaBrokerAdapter(BaseBrokerAdapter):
    """Alpaca broker adapter implementation"""
    
    def __init__(self, broker: Broker, paper_trading: bool = True):
        """
        Initialize Alpaca adapter
        
        Args:
            broker: Broker instance
            paper_trading: Whether to use paper trading credentials
        """
        super().__init__(broker, paper_trading)
        # Remove trailing slash and /v2 if present (we'll add /v2 in requests)
        self.base_url = self.endpoint_url.rstrip('/').rstrip('/v2')
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret,
        }
    
    def connect(self) -> bool:
        """Connect to Alpaca API (no connection needed for REST API)"""
        return True
    
    def disconnect(self):
        """Disconnect from Alpaca API (no connection needed for REST API)"""
        pass
    
    def verify_credentials(self) -> bool:
        """
        Verify API credentials by making a test request to Alpaca API
        
        Returns:
            bool: True if credentials are valid, False otherwise
        """
        try:
            # Test credentials by getting account information
            response = requests.get(
                f'{self.base_url}/v2/account',
                headers=self.headers,
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error verifying Alpaca credentials: {e}")
            return False
    
    def get_account_balance(self) -> Decimal:
        """
        Get account balance
        
        Returns:
            Decimal: Account cash balance (returns 0 if error)
        """
        try:
            response = requests.get(
                f'{self.base_url}/v2/account',
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                account = response.json()
                return Decimal(str(account.get('cash', '0')))
            return Decimal('0')
        except Exception as e:
            print(f"Error getting Alpaca account balance: {e}")
            return Decimal('0')
    
    def get_account_equity(self) -> Decimal:
        """
        Get total account equity (cash + positions value)
        
        Returns:
            Decimal: Total account equity (returns 0 if error)
        """
        try:
            response = requests.get(
                f'{self.base_url}/v2/account',
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                account = response.json()
                return Decimal(str(account.get('equity', '0')))
            return Decimal('0')
        except Exception as e:
            print(f"Error getting Alpaca account equity: {e}")
            return Decimal('0')
    
    def get_current_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get current price for a symbol
        
        Args:
            symbol: Symbol ticker (e.g., 'AAPL')
        
        Returns:
            Decimal: Current price, or None if error
        """
        try:
            response = requests.get(
                f'{self.base_url}/v2/stocks/{symbol}/bars/latest',
                headers=self.headers,
                params={'feed': 'iex'},  # Use IEX feed for free tier
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                bar = data.get('bar', {})
                return Decimal(str(bar.get('c', '0')))  # 'c' is close price
            return None
        except Exception as e:
            print(f"Error getting Alpaca current price for {symbol}: {e}")
            return None
    
    def place_order(
        self,
        symbol: str,
        side: str,  # 'buy' or 'sell'
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
            quantity: Number of shares
            order_type: Order type (default: 'market')
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
        
        Returns:
            OrderResult: Order execution result
        """
        try:
            order_data = {
                'symbol': symbol,
                'qty': str(int(quantity)),
                'side': side,
                'type': order_type,
                'time_in_force': 'day',
            }
            
            if limit_price:
                order_data['limit_price'] = str(limit_price)
            if stop_price:
                order_data['stop_price'] = str(stop_price)
            
            response = requests.post(
                f'{self.base_url}/v2/orders',
                headers=self.headers,
                json=order_data,
                timeout=10
            )
            
            if response.status_code == 200 or response.status_code == 201:
                order = response.json()
                filled_qty = Decimal(str(order.get('filled_qty', '0')))
                filled_price = Decimal(str(order.get('filled_avg_price', '0')))
                order_status = order.get('status', 'pending')
                return OrderResult(
                    order_id=str(order.get('id', '')),
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    filled_quantity=filled_qty,
                    price=filled_price if filled_qty > 0 else Decimal('0'),
                    status=order_status,
                    timestamp=datetime.now(),
                    broker_order_id=str(order.get('id', '')),
                )
            else:
                error_msg = response.json().get('message', 'Unknown error')
                return OrderResult(
                    order_id='',
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    filled_quantity=Decimal('0'),
                    price=Decimal('0'),
                    status='rejected',
                    timestamp=datetime.now(),
                    error_message=f"Order failed: {error_msg}",
                )
        except Exception as e:
            return OrderResult(
                order_id='',
                symbol=symbol,
                side=side,
                quantity=quantity,
                filled_quantity=Decimal('0'),
                price=Decimal('0'),
                status='rejected',
                timestamp=datetime.now(),
                error_message=f"Error placing order: {str(e)}",
            )
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            response = requests.delete(
                f'{self.base_url}/v2/orders/{order_id}',
                headers=self.headers,
                timeout=10
            )
            return response.status_code == 204
        except Exception:
            return False
    
    def get_order_status(self, order_id: str) -> OrderResult:
        """Get order status"""
        try:
            response = requests.get(
                f'{self.base_url}/v2/orders/{order_id}',
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                order = response.json()
                filled_qty = Decimal(str(order.get('filled_qty', '0')))
                filled_price = Decimal(str(order.get('filled_avg_price', '0')))
                return OrderResult(
                    order_id=str(order.get('id', order_id)),
                    symbol=order.get('symbol', ''),
                    side=order.get('side', 'buy'),
                    quantity=Decimal(str(order.get('qty', '0'))),
                    filled_quantity=filled_qty,
                    price=filled_price if filled_qty > 0 else Decimal('0'),
                    status=order.get('status', 'pending'),
                    timestamp=datetime.now(),
                    broker_order_id=str(order.get('id', order_id)),
                )
            else:
                # Return a rejected order result if not found
                return OrderResult(
                    order_id=order_id,
                    symbol='',
                    side='buy',
                    quantity=Decimal('0'),
                    filled_quantity=Decimal('0'),
                    price=Decimal('0'),
                    status='rejected',
                    timestamp=datetime.now(),
                    error_message='Order not found',
                )
        except Exception as e:
            return OrderResult(
                order_id=order_id,
                symbol='',
                side='buy',
                quantity=Decimal('0'),
                filled_quantity=Decimal('0'),
                price=Decimal('0'),
                status='rejected',
                timestamp=datetime.now(),
                error_message=f"Error getting order status: {str(e)}",
            )
    
    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Get position information for a symbol"""
        try:
            response = requests.get(
                f'{self.base_url}/v2/positions/{symbol}',
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                pos = response.json()
                return PositionInfo(
                    symbol=symbol,
                    quantity=Decimal(str(pos.get('qty', '0'))),
                    average_price=Decimal(str(pos.get('avg_entry_price', '0'))),
                    current_price=Decimal(str(pos.get('current_price', '0'))),
                    unrealized_pnl=Decimal(str(pos.get('unrealized_pl', '0'))),
                    position_type='long' if Decimal(str(pos.get('qty', '0'))) > 0 else 'short',
                    timestamp=datetime.now(),
                )
            return None
        except Exception:
            return None
    
    def get_symbol_capabilities(self, symbol: str) -> Dict:
        """
        Get trading capabilities for a symbol (long/short support, etc.)
        
        Args:
            symbol: Symbol ticker
        
        Returns:
            Dict with:
            - 'long_supported': bool
            - 'short_supported': bool
            - 'min_order_size': Decimal (optional)
            - 'max_order_size': Decimal (optional)
        """
        try:
            # Get asset information to check if shortable
            response = requests.get(
                f'{self.base_url}/v2/assets/{symbol}',
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                asset = response.json()
                tradable = asset.get('tradable', False)
                shortable = asset.get('shortable', False)
                return {
                    'long_supported': tradable,
                    'short_supported': shortable and tradable,
                }
            return {'long_supported': False, 'short_supported': False}
        except Exception:
            # On error, assume long only
            return {'long_supported': True, 'short_supported': False}
    
    def get_tradable_symbols(self) -> List[str]:
        """
        Get list of tradable symbols
        
        Returns:
            List of symbol tickers
        """
        try:
            # Use the method that gets all symbols with capabilities, then extract just the symbols
            all_symbols_data = self.get_all_symbols_with_capabilities()
            return list(all_symbols_data.keys())
        except Exception:
            return []
    
    def get_all_symbols_with_capabilities(self) -> Dict[str, Dict]:
        """
        Get all tradable symbols with their capabilities in a single API call
        
        Returns:
            Dictionary mapping symbol -> capabilities dict with:
            - 'tradable': bool
            - 'shortable': bool
            - 'long_supported': bool
            - 'short_supported': bool
        """
        try:
            response = requests.get(
                f'{self.base_url}/v2/assets',
                headers=self.headers,
                params={'status': 'active', 'asset_class': 'us_equity'},
                timeout=30  # Longer timeout for large response
            )
            if response.status_code == 200:
                assets = response.json()
                result = {}
                for asset in assets:
                    symbol = asset.get('symbol')
                    if symbol:
                        tradable = asset.get('tradable', False)
                        shortable = asset.get('shortable', False)
                        result[symbol] = {
                            'tradable': tradable,
                            'shortable': shortable,
                            'long_supported': tradable,
                            'short_supported': shortable and tradable,
                        }
                return result
            return {}
        except Exception as e:
            # Fallback: if logger is not available, use print
            try:
                logger.error(f"Error getting all symbols with capabilities: {e}")
            except:
                print(f"Error getting all symbols with capabilities: {e}")
            return {}
    
    def is_symbol_tradable(self, symbol: str) -> bool:
        """
        Check if a symbol is tradable on this broker
        
        Args:
            symbol: Symbol ticker
        
        Returns:
            True if symbol is tradable, False otherwise
        """
        try:
            response = requests.get(
                f'{self.base_url}/v2/assets/{symbol}',
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                asset = response.json()
                return asset.get('tradable', False) and asset.get('status') == 'active'
            return False
        except Exception:
            return False
    
    def get_all_positions(self) -> List[PositionInfo]:
        """
        Get all current positions
        
        Returns:
            List of PositionInfo instances
        """
        try:
            response = requests.get(
                f'{self.base_url}/v2/positions',
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                positions = response.json()
                result = []
                for pos in positions:
                    qty = Decimal(str(pos.get('qty', '0')))
                    if qty != 0:  # Only include non-zero positions
                        result.append(PositionInfo(
                            symbol=pos.get('symbol', ''),
                            quantity=qty,
                            average_price=Decimal(str(pos.get('avg_entry_price', '0'))),
                            current_price=Decimal(str(pos.get('current_price', '0'))),
                            unrealized_pnl=Decimal(str(pos.get('unrealized_pl', '0'))),
                            position_type='long' if qty > 0 else 'short',
                            timestamp=datetime.now(),
                        ))
                return result
            return []
        except Exception as e:
            print(f"Error getting Alpaca all positions: {e}")
            return []
    
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
        try:
            params = {
                'timeframe': timeframe,
                'feed': 'iex',
            }
            
            if start_date:
                params['start'] = start_date.isoformat()
            if end_date:
                params['end'] = end_date.isoformat()
            if not start_date and not end_date:
                params['limit'] = 100  # Default limit if no dates specified
            
            response = requests.get(
                f'{self.base_url}/v2/stocks/{symbol}/bars',
                headers=self.headers,
                params=params,
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                bars = data.get('bars', [])
                # Convert list of dicts to proper format
                return [bar for bar in bars]
            return []
        except Exception as e:
            print(f"Error getting Alpaca market data for {symbol}: {e}")
            return []

