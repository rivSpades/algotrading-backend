"""
Live Trading Executor Service
Orchestrates live trading execution for deployments
"""

import logging
from typing import Dict, Optional, List
from decimal import Decimal
from django.utils import timezone
from django.db import transaction

from ..models import LiveTradingDeployment, LiveTrade
from ..services.evaluation_service import EvaluationService
from strategies.engines import EngineRegistry, BaseLiveTradingEngine
from ..adapters.factory import get_broker_adapter
from market_data.models import Symbol

logger = logging.getLogger(__name__)


class LiveTradingExecutor:
    """Executes live trading for a deployment"""
    
    def __init__(self, deployment: LiveTradingDeployment):
        """
        Initialize live trading executor
        
        Args:
            deployment: LiveTradingDeployment instance to execute
        """
        self.deployment = deployment
        self.broker = deployment.broker
        self.strategy = deployment.backtest.strategy
        self.parameters = deployment.strategy_parameters
        self.position_mode = deployment.position_mode
        
        # Get broker adapter
        from ..adapters.factory import get_broker_adapter
        paper_trading = deployment.deployment_type == 'paper'
        self.broker_adapter = get_broker_adapter(self.broker, paper_trading=paper_trading)
        
        if not self.broker_adapter:
            raise ValueError(f"No broker adapter available for broker {self.broker.code}")
        
        # Get live trading engine
        engine_class = EngineRegistry.get_live_trading_engine(self.strategy.name)
        if not engine_class:
            raise ValueError(f"No live trading engine registered for strategy {self.strategy.name}")
        
        # Initialize engine
        self.engine = engine_class(
            strategy=self.strategy,
            parameters=self.parameters,
            deployment=self.deployment,
            broker_adapter=self.broker_adapter,
            position_mode=self.position_mode
        )
        
        # Cache for positions
        self._positions_cache: Dict[str, Dict] = {}
        
        logger.info(
            f"Initialized LiveTradingExecutor for deployment {deployment.id} "
            f"(strategy: {self.strategy.name}, broker: {self.broker.name}, mode: {self.position_mode})"
        )
    
    def start(self):
        """Start the live trading executor"""
        try:
            # Connect to broker
            if not self.broker_adapter.connect():
                raise ConnectionError(f"Failed to connect to broker {self.broker.name}")
            
            logger.info(f"Connected to broker {self.broker.name}")
            
            # Initialize engine
            self.engine.initialize()
            
            logger.info(f"Live trading executor started for deployment {self.deployment.id}")
            
            # Update deployment status
            self.deployment.status = 'evaluating' if self.deployment.deployment_type == 'paper' else 'active'
            self.deployment.started_at = timezone.now()
            self.deployment.save()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting live trading executor: {e}", exc_info=True)
            self.deployment.status = 'failed'
            self.deployment.error_message = str(e)
            self.deployment.save()
            return False
    
    def process_market_update(self, symbol: Symbol, market_data: Dict) -> Optional[Dict]:
        """
        Process a market data update and potentially execute trades
        
        Args:
            symbol: Symbol instance
            market_data: Dict with OHLCV data:
                - timestamp: datetime
                - open: Decimal
                - high: Decimal
                - low: Decimal
                - close: Decimal
                - volume: int
        
        Returns:
            Trade execution dict if trade was executed, None otherwise
        """
        try:
            # Check if symbol is in deployment
            if symbol not in self.deployment.symbols.all():
                logger.warning(f"Symbol {symbol.ticker} not in deployment {self.deployment.id}")
                return None
            
            # Process market data through engine
            trade_result = self.engine.process_market_data(symbol, market_data)
            
            if trade_result:
                # Save trade to database
                trade = self._save_trade(trade_result, symbol, market_data)
                
                # Check evaluation if paper trading
                if self.deployment.deployment_type == 'paper':
                    self._check_evaluation()
                
                return trade_result
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing market update for {symbol.ticker}: {e}", exc_info=True)
            return None
    
    def _save_trade(self, trade_result: Dict, symbol: Symbol, market_data: Dict) -> LiveTrade:
        """
        Save trade execution to database
        
        Args:
            trade_result: Dict from engine.process_market_data()
            symbol: Symbol instance
            market_data: Market data dict
        
        Returns:
            LiveTrade instance
        """
        with transaction.atomic():
            trade = LiveTrade.objects.create(
                deployment=self.deployment,
                symbol=symbol,
                position_mode=trade_result.get('position_mode', self.position_mode),
                trade_type=trade_result.get('trade_type'),  # 'buy' or 'sell'
                entry_price=trade_result.get('entry_price'),
                quantity=trade_result.get('quantity'),
                entry_timestamp=trade_result.get('timestamp', timezone.now()),
                status='open' if trade_result.get('status') != 'closed' else 'closed',
                broker_order_id=trade_result.get('broker_order_id'),
                metadata=trade_result.get('metadata', {})
            )
            
            # Update exit price and PnL if trade is closed
            if trade_result.get('status') == 'closed':
                trade.exit_price = trade_result.get('exit_price')
                trade.exit_timestamp = trade_result.get('exit_timestamp', timezone.now())
                trade.pnl = trade_result.get('pnl')
                trade.pnl_percentage = trade_result.get('pnl_percentage')
                trade.is_winner = trade_result.get('pnl', 0) > 0 if trade_result.get('pnl') is not None else None
                trade.status = 'closed'
                trade.save()
            
            logger.info(
                f"Saved trade for {symbol.ticker}: {trade.trade_type} @ {trade.entry_price} "
                f"(order_id: {trade.broker_order_id})"
            )
            
            return trade
    
    def _check_evaluation(self):
        """Check if deployment meets evaluation criteria"""
        try:
            EvaluationService.check_and_update_evaluation(self.deployment)
        except Exception as e:
            logger.error(f"Error checking evaluation: {e}", exc_info=True)
    
    def update_positions(self):
        """Update position cache from broker"""
        try:
            positions = self.broker_adapter.get_all_positions()
            self._positions_cache = {
                pos.symbol: {
                    'quantity': pos.quantity,
                    'average_price': pos.average_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'position_type': pos.position_type,
                }
                for pos in positions
            }
            return self._positions_cache
        except Exception as e:
            logger.error(f"Error updating positions: {e}", exc_info=True)
            return {}
    
    def get_open_trades(self) -> List[LiveTrade]:
        """Get all open trades for this deployment"""
        return list(self.deployment.live_trades.filter(status='open'))
    
    def close_trade(self, trade: LiveTrade, exit_price: Decimal, exit_timestamp=None):
        """
        Close an open trade
        
        Args:
            trade: LiveTrade instance
            exit_price: Exit price
            exit_timestamp: Exit timestamp (defaults to now)
        """
        try:
            with transaction.atomic():
                trade.exit_price = exit_price
                trade.exit_timestamp = exit_timestamp or timezone.now()
                
                # Calculate PnL
                if trade.position_mode == 'long':
                    pnl = (exit_price - trade.entry_price) * trade.quantity
                else:  # short
                    pnl = (trade.entry_price - exit_price) * trade.quantity
                
                trade.pnl = pnl
                trade.pnl_percentage = (float(pnl) / float(trade.entry_price * trade.quantity)) * 100
                trade.is_winner = pnl > 0
                trade.status = 'closed'
                trade.save()
                
                logger.info(
                    f"Closed trade {trade.id} for {trade.symbol.ticker}: "
                    f"PnL={pnl}, PnL%={trade.pnl_percentage:.2f}%"
                )
                
                # Check evaluation if paper trading
                if self.deployment.deployment_type == 'paper':
                    self._check_evaluation()
                
                return trade
                
        except Exception as e:
            logger.error(f"Error closing trade {trade.id}: {e}", exc_info=True)
            raise
    
    def pause(self):
        """Pause live trading"""
        try:
            self.deployment.status = 'paused'
            self.deployment.save()
            logger.info(f"Paused deployment {self.deployment.id}")
            return True
        except Exception as e:
            logger.error(f"Error pausing deployment: {e}", exc_info=True)
            return False
    
    def stop(self):
        """Stop live trading and cleanup"""
        try:
            # Close all open positions (optional - depends on strategy)
            # For now, just mark deployment as stopped
            
            # Disconnect from broker
            if self.broker_adapter:
                self.broker_adapter.disconnect()
            
            # Shutdown engine
            if self.engine:
                self.engine.shutdown()
            
            # Update deployment status
            self.deployment.status = 'stopped'
            self.deployment.save()
            
            logger.info(f"Stopped deployment {self.deployment.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping deployment: {e}", exc_info=True)
            return False
    
    def get_statistics(self) -> Dict:
        """Get current deployment statistics"""
        closed_trades = self.deployment.live_trades.filter(status='closed')
        open_trades = self.deployment.live_trades.filter(status='open')
        
        total_pnl = sum(float(trade.pnl or 0) for trade in closed_trades)
        
        # Get unrealized PnL from open positions
        self.update_positions()
        unrealized_pnl = sum(
            float(pos.get('unrealized_pnl', 0))
            for pos in self._positions_cache.values()
        )
        
        return {
            'total_trades': self.deployment.live_trades.count(),
            'open_trades': open_trades.count(),
            'closed_trades': closed_trades.count(),
            'total_pnl': total_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_equity': total_pnl + unrealized_pnl + float(self.deployment.initial_capital),
            'evaluation_results': self.deployment.evaluation_results,
            'evaluation_criteria': self.deployment.evaluation_criteria,
            'status': self.deployment.status,
        }


