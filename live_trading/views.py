"""
API Views for Live Trading
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.filters import SearchFilter, OrderingFilter
from rest_framework.pagination import PageNumberPagination
from django.shortcuts import get_object_or_404
from django.db.models import Q
from django.utils import timezone
from .models import Broker, SymbolBrokerAssociation, LiveTradingDeployment, LiveTrade
from .serializers import (
    BrokerSerializer,
    SymbolBrokerAssociationSerializer,
    LiveTradingDeploymentListSerializer,
    LiveTradingDeploymentDetailSerializer,
    LiveTradingDeploymentCreateSerializer,
    LiveTradeSerializer,
    BrokerSymbolLinkSerializer
)
from market_data.models import Symbol, Exchange
from backtest_engine.models import Backtest
from .services.evaluation_service import EvaluationService
import logging

logger = logging.getLogger(__name__)


class BrokerPagination(PageNumberPagination):
    """Pagination for brokers"""
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100


class BrokerSymbolPagination(PageNumberPagination):
    """Pagination for broker symbols"""
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100


class BrokerViewSet(viewsets.ModelViewSet):
    """ViewSet for Broker"""
    queryset = Broker.objects.all()
    serializer_class = BrokerSerializer
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['name', 'code']
    ordering_fields = ['name', 'code', 'created_at']
    ordering = ['name']
    pagination_class = BrokerPagination
    
    def filter_queryset(self, queryset):
        """Override to skip filters for actions that use custom search parameters"""
        # For the 'symbols' action, bypass all filters to avoid SearchFilter conflict
        if self.action == 'symbols':
            return queryset
        return super().filter_queryset(queryset)
    
    @action(detail=True, methods=['get'], url_path='symbols')
    def symbols(self, request, pk=None):
        """Get all symbols associated with this broker with pagination and search"""
        # Use get_object_or_404 directly to bypass any filter backend interference
        broker = get_object_or_404(Broker, pk=pk)
        
        associations = SymbolBrokerAssociation.objects.filter(broker=broker).select_related('symbol', 'symbol__exchange')
        
        # Apply search filtering if search parameter is provided
        # Use 'symbol_search' to avoid conflict with ViewSet's SearchFilter on 'search'
        search_query = request.query_params.get('search', None) or request.query_params.get('symbol_search', None)
        if search_query:
            associations = associations.filter(
                Q(symbol__ticker__icontains=search_query) | 
                Q(symbol__name__icontains=search_query)
            )
        
        # Apply pagination
        paginator = BrokerSymbolPagination()
        page = paginator.paginate_queryset(associations, request)
        
        if page is not None:
            serializer = SymbolBrokerAssociationSerializer(page, many=True)
            return paginator.get_paginated_response(serializer.data)
        
        # Fallback if pagination is not applied (shouldn't happen with paginator)
        serializer = SymbolBrokerAssociationSerializer(associations, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'], url_path='link-symbols')
    def link_symbols(self, request, pk=None):
        """Link symbols to this broker asynchronously (individual, by exchange, or all available)"""
        broker = self.get_object()
        serializer = BrokerSymbolLinkSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        data = serializer.validated_data
        symbol_tickers = data.get('symbol_tickers', [])
        exchange_code = data.get('exchange_code', '')
        link_all_available = data.get('link_all_available', False)
        verify_capabilities = data.get('verify_capabilities', True)
        
        # Start Celery task for symbol linking
        from .tasks import link_broker_symbols_task
        
        try:
            task = link_broker_symbols_task.delay(
                broker_id=broker.id,
                symbol_tickers=symbol_tickers if symbol_tickers else None,
                exchange_code=exchange_code if exchange_code else None,
                link_all_available=link_all_available,
                verify_capabilities=verify_capabilities
            )
            logger.info(f"Started symbol linking task for broker {broker.id}, task_id: {task.id}")
            
            return Response({
                'message': 'Symbol linking task started',
                'task_id': task.id,
                'broker_id': broker.id,
                'broker_name': broker.name
            }, status=status.HTTP_202_ACCEPTED)
        except Exception as e:
            logger.error(f"Error starting symbol linking task: {str(e)}")
            return Response(
                {'error': f'Error starting symbol linking task: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['post'], url_path='test-connection')
    def test_connection(self, request, pk=None):
        """Test broker API connection and verify credentials for paper or real money"""
        broker = self.get_object()
        deployment_type = request.data.get('deployment_type', 'paper')  # 'paper' or 'real_money'
        
        from .adapters.factory import get_broker_adapter
        
        if deployment_type == 'paper':
            if not broker.has_paper_trading_credentials():
                return Response(
                    {'error': 'Paper trading credentials not configured. Please provide endpoint_url, api_key, and secret_key.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            adapter = get_broker_adapter(broker, paper_trading=True)
        elif deployment_type == 'real_money':
            if not broker.has_real_money_credentials():
                return Response(
                    {'error': 'Real money credentials not configured. Please provide endpoint_url, api_key, and secret_key.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            adapter = get_broker_adapter(broker, paper_trading=False)
        else:
            return Response(
                {'error': 'Invalid deployment_type. Must be "paper" or "real_money".'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not adapter:
            return Response(
                {'error': f'Broker adapter not found for code: {broker.code}. Only ALPACA is currently supported.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            is_valid = adapter.verify_credentials()
            if is_valid:
                # Update broker active status for the specific deployment type
                if deployment_type == 'paper':
                    broker.paper_trading_active = True
                else:
                    broker.real_money_active = True
                broker.save()
                return Response({
                    'success': True,
                    'message': f'{deployment_type.replace("_", " ").title()} connection test successful. Broker is now active for this type.',
                    'deployment_type': deployment_type,
                    'is_active': True
                })
            else:
                if deployment_type == 'paper':
                    broker.paper_trading_active = False
                else:
                    broker.real_money_active = False
                broker.save()
                return Response({
                    'success': False,
                    'message': f'{deployment_type.replace("_", " ").title()} connection test failed. Invalid credentials.',
                    'deployment_type': deployment_type,
                    'is_active': False
                }, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            if deployment_type == 'paper':
                broker.paper_trading_active = False
            else:
                broker.real_money_active = False
            broker.save()
            return Response(
                {'error': f'Connection test failed: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )
    
    @action(detail=True, methods=['get'], url_path='account-balance')
    def account_balance(self, request, pk=None):
        """Get account balance for paper or real money trading"""
        broker = self.get_object()
        deployment_type = request.query_params.get('deployment_type', 'paper')
        
        from .adapters.factory import get_broker_adapter
        
        if deployment_type == 'paper':
            if not broker.has_paper_trading_credentials():
                return Response(
                    {'error': 'Paper trading credentials not configured.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            adapter = get_broker_adapter(broker, paper_trading=True)
        elif deployment_type == 'real_money':
            if not broker.has_real_money_credentials():
                return Response(
                    {'error': 'Real money credentials not configured.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            adapter = get_broker_adapter(broker, paper_trading=False)
        else:
            return Response(
                {'error': 'Invalid deployment_type. Must be "paper" or "real_money".'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not adapter:
            return Response(
                {'error': f'Broker adapter not found for code: {broker.code}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            balance = adapter.get_account_balance()
            equity = adapter.get_account_equity()
            return Response({
                'balance': str(balance),
                'equity': str(equity),
                'deployment_type': deployment_type
            })
        except Exception as e:
            return Response(
                {'error': f'Failed to get account balance: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )
    
    @action(detail=True, methods=['get'], url_path='check-symbol')
    def check_symbol(self, request, pk=None):
        """Check if a symbol exists and is tradable on the broker"""
        broker = self.get_object()
        symbol = request.query_params.get('symbol', '').upper()
        deployment_type = request.query_params.get('deployment_type', 'paper')
        
        if not symbol:
            return Response(
                {'error': 'Symbol parameter is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        from .adapters.factory import get_broker_adapter
        
        if deployment_type == 'paper':
            if not broker.has_paper_trading_credentials():
                return Response(
                    {'error': 'Paper trading credentials not configured.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            adapter = get_broker_adapter(broker, paper_trading=True)
        elif deployment_type == 'real_money':
            if not broker.has_real_money_credentials():
                return Response(
                    {'error': 'Real money credentials not configured.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            adapter = get_broker_adapter(broker, paper_trading=False)
        else:
            return Response(
                {'error': 'Invalid deployment_type. Must be "paper" or "real_money".'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not adapter:
            return Response(
                {'error': f'Broker adapter not found for code: {broker.code}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            is_tradable = adapter.is_symbol_tradable(symbol)
            capabilities = adapter.get_symbol_capabilities(symbol)
            current_price = adapter.get_current_price(symbol)
            
            return Response({
                'symbol': symbol,
                'is_tradable': is_tradable,
                'capabilities': {
                    'long_supported': capabilities.get('long_supported', False),
                    'short_supported': capabilities.get('short_supported', False),
                },
                'current_price': str(current_price) if current_price else None,
                'deployment_type': deployment_type
            })
        except Exception as e:
            return Response(
                {'error': f'Failed to check symbol: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )
    
    @action(detail=True, methods=['get'], url_path='positions')
    def positions(self, request, pk=None):
        """Get all current positions"""
        broker = self.get_object()
        deployment_type = request.query_params.get('deployment_type', 'paper')
        
        from .adapters.factory import get_broker_adapter
        
        if deployment_type == 'paper':
            if not broker.has_paper_trading_credentials():
                return Response(
                    {'error': 'Paper trading credentials not configured.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            adapter = get_broker_adapter(broker, paper_trading=True)
        elif deployment_type == 'real_money':
            if not broker.has_real_money_credentials():
                return Response(
                    {'error': 'Real money credentials not configured.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            adapter = get_broker_adapter(broker, paper_trading=False)
        else:
            return Response(
                {'error': 'Invalid deployment_type. Must be "paper" or "real_money".'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not adapter:
            return Response(
                {'error': f'Broker adapter not found for code: {broker.code}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            positions = adapter.get_all_positions()
            from .adapters.base import PositionInfo
            positions_data = [{
                'symbol': pos.symbol,
                'quantity': str(pos.quantity),
                'average_price': str(pos.average_price),
                'current_price': str(pos.current_price),
                'unrealized_pnl': str(pos.unrealized_pnl),
                'position_type': pos.position_type,
            } for pos in positions]
            
            return Response({
                'positions': positions_data,
                'count': len(positions_data),
                'deployment_type': deployment_type
            })
        except Exception as e:
            return Response(
                {'error': f'Failed to get positions: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )


class SymbolBrokerAssociationViewSet(viewsets.ModelViewSet):
    """ViewSet for SymbolBrokerAssociation"""
    queryset = SymbolBrokerAssociation.objects.select_related('symbol', 'broker').all()
    serializer_class = SymbolBrokerAssociationSerializer
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['symbol__ticker', 'broker__name', 'broker__code']
    ordering_fields = ['symbol__ticker', 'broker__name', 'updated_at']
    ordering = ['broker__name', 'symbol__ticker']
    
    def get_queryset(self):
        """Filter by broker or symbol if provided"""
        queryset = super().get_queryset()
        
        broker_id = self.request.query_params.get('broker', None)
        if broker_id:
            queryset = queryset.filter(broker_id=broker_id)
        
        symbol_ticker = self.request.query_params.get('symbol', None)
        if symbol_ticker:
            queryset = queryset.filter(symbol__ticker=symbol_ticker)
        
        # Filter by long_active or short_active
        long_active = self.request.query_params.get('long_active', None)
        if long_active is not None:
            queryset = queryset.filter(long_active=long_active.lower() == 'true')
        
        short_active = self.request.query_params.get('short_active', None)
        if short_active is not None:
            queryset = queryset.filter(short_active=short_active.lower() == 'true')
        
        return queryset


class LiveTradingDeploymentPagination(PageNumberPagination):
    """Pagination for live trading deployments"""
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100


class LiveTradingDeploymentViewSet(viewsets.ModelViewSet):
    """ViewSet for LiveTradingDeployment"""
    queryset = LiveTradingDeployment.objects.select_related('broker', 'backtest', 'backtest__strategy').all()
    serializer_class = LiveTradingDeploymentDetailSerializer
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['name', 'backtest__strategy__name', 'broker__name']
    ordering_fields = ['created_at', 'updated_at', 'started_at', 'status']
    ordering = ['-created_at']
    pagination_class = LiveTradingDeploymentPagination
    
    def get_queryset(self):
        """Filter by broker, status, or deployment_type if provided"""
        queryset = super().get_queryset()
        
        broker_id = self.request.query_params.get('broker', None)
        if broker_id:
            queryset = queryset.filter(broker_id=broker_id)
        
        deployment_type = self.request.query_params.get('deployment_type', None)
        if deployment_type:
            queryset = queryset.filter(deployment_type=deployment_type)
        
        status_filter = self.request.query_params.get('status', None)
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        return queryset
    
    def get_serializer_class(self):
        if self.action == 'create':
            return LiveTradingDeploymentCreateSerializer
        elif self.action == 'list':
            return LiveTradingDeploymentListSerializer
        return LiveTradingDeploymentDetailSerializer
    
    def create(self, request, *args, **kwargs):
        """Create a new live trading deployment"""
        serializer = LiveTradingDeploymentCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        data = serializer.validated_data
        
        # Get backtest
        try:
            backtest = Backtest.objects.get(id=data['backtest_id'])
        except Backtest.DoesNotExist:
            return Response(
                {'error': f'Backtest with id {data["backtest_id"]} not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Get broker
        try:
            broker = Broker.objects.get(id=data['broker_id'])
        except Broker.DoesNotExist:
            return Response(
                {'error': f'Broker with id {data["broker_id"]} not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Validate position mode
        position_mode = data['position_mode']
        if position_mode not in ['all', 'long', 'short']:
            return Response(
                {'error': 'Invalid position_mode. Must be one of: all, long, short'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get symbols
        symbols_to_deploy = []
        symbol_tickers = data.get('symbol_tickers', [])
        exchange_code = data.get('exchange_code', '')
        
        if symbol_tickers:
            symbols = Symbol.objects.filter(ticker__in=symbol_tickers)
            symbols_to_deploy.extend(symbols)
        
        if exchange_code:
            exchange = get_object_or_404(Exchange, code=exchange_code)
            exchange_symbols = Symbol.objects.filter(exchange=exchange)
            symbols_to_deploy.extend(exchange_symbols)
        
        # Remove duplicates
        symbols_to_deploy = list(set(symbols_to_deploy))
        
        if not symbols_to_deploy:
            return Response(
                {'error': 'No symbols selected for deployment'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Filter symbols based on broker associations and position mode
        filtered_symbols = []
        for symbol in symbols_to_deploy:
            try:
                association = SymbolBrokerAssociation.objects.get(symbol=symbol, broker=broker)
                if association.supports_mode(position_mode):
                    filtered_symbols.append(symbol)
            except SymbolBrokerAssociation.DoesNotExist:
                # Symbol not associated with broker, skip it
                continue
        
        if not filtered_symbols:
            return Response(
                {'error': f'No symbols support {position_mode} mode on this broker'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Create deployment
        deployment = LiveTradingDeployment.objects.create(
            name=data.get('name', ''),
            backtest=backtest,
            position_mode=position_mode,
            broker=broker,
            deployment_type='paper',  # Always start with paper trading
            status='pending',
            evaluation_criteria=data['evaluation_criteria'],
            initial_capital=backtest.initial_capital,
            bet_size_percentage=backtest.bet_size_percentage,
            strategy_parameters=backtest.strategy_parameters
        )
        
        deployment.symbols.set(filtered_symbols)
        
        serializer = LiveTradingDeploymentDetailSerializer(deployment)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    
    @action(detail=True, methods=['post'])
    def activate(self, request, pk=None):
        """Activate the deployment (start paper trading)"""
        deployment = self.get_object()
        
        if deployment.status != 'pending':
            return Response(
                {'error': f'Deployment must be in "pending" status to activate. Current status: {deployment.status}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if deployment.deployment_type == 'paper' and not deployment.broker.has_credentials():
            return Response(
                {'error': 'Broker does not have paper trading credentials configured'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if deployment.deployment_type == 'real_money' and not deployment.broker.is_active_for_deployment_type('real_money'):
            return Response(
                {'error': 'Broker does not have real money credentials configured'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        deployment.status = 'pending'  # Will be set to 'evaluating'/'active' by executor
        deployment.started_at = None
        deployment.save()
        
        # Start the live trading task asynchronously
        try:
            from .tasks import start_deployment_task
            task = start_deployment_task.delay(deployment.id)
            logger.info(f"Started live trading task for deployment {deployment.id}, task_id: {task.id}")
            
            serializer = self.get_serializer(deployment)
            response_data = serializer.data
            response_data['task_id'] = task.id
            return Response(response_data)
        except Exception as e:
            logger.error(f"Error starting live trading task: {str(e)}")
            deployment.status = 'failed'
            deployment.error_message = f"Error starting deployment: {str(e)}"
            deployment.save()
            return Response(
                {'error': f'Error starting deployment: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['post'], url_path='promote-to-real-money')
    def promote_to_real_money(self, request, pk=None):
        """Promote paper trading deployment to real money (if evaluation passed)"""
        deployment = self.get_object()
        
        if deployment.deployment_type != 'paper':
            return Response(
                {'error': 'Can only promote paper trading deployments to real money'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if deployment.status != 'passed':
            return Response(
                {'error': f'Deployment evaluation must pass before promoting to real money. Current status: {deployment.status}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not deployment.broker.is_active_for_deployment_type('real_money'):
            return Response(
                {'error': 'Broker is not active for real money trading. Please test real money connection first.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Create a new deployment for real money
        real_money_deployment = LiveTradingDeployment.objects.create(
            name=f"{deployment.name} (Real Money)",
            backtest=deployment.backtest,
            position_mode=deployment.position_mode,
            broker=deployment.broker,
            deployment_type='real_money',
            status='pending',
            evaluation_criteria=deployment.evaluation_criteria,
            evaluation_results=deployment.evaluation_results,
            initial_capital=deployment.initial_capital,
            bet_size_percentage=deployment.bet_size_percentage,
            strategy_parameters=deployment.strategy_parameters,
            activated_at=timezone.now()
        )
        
        real_money_deployment.symbols.set(deployment.symbols.all())
        
        serializer = self.get_serializer(real_money_deployment)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    
    @action(detail=True, methods=['post'])
    def pause(self, request, pk=None):
        """Pause an active deployment"""
        deployment = self.get_object()
        
        if deployment.status != 'active':
            return Response(
                {'error': f'Can only pause active deployments. Current status: {deployment.status}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        deployment.status = 'paused'
        deployment.save()
        
        # Pause is handled by status check in process_market_data_task
        # Active deployments check status before processing
        
        serializer = self.get_serializer(deployment)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def stop(self, request, pk=None):
        """Stop a deployment"""
        deployment = self.get_object()
        
        if deployment.status in ['stopped']:
            return Response(
                {'error': 'Deployment is already stopped'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Stop deployment asynchronously
        try:
            from .tasks import stop_deployment_task
            task = stop_deployment_task.delay(deployment.id)
            logger.info(f"Stopped live trading task for deployment {deployment.id}, task_id: {task.id}")
            
            serializer = self.get_serializer(deployment)
            response_data = serializer.data
            response_data['task_id'] = task.id
            return Response(response_data)
        except Exception as e:
            logger.error(f"Error stopping live trading task: {str(e)}")
            # Still update status locally
            deployment.status = 'stopped'
            deployment.save()
            serializer = self.get_serializer(deployment)
            return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def statistics(self, request, pk=None):
        """Get deployment statistics"""
        deployment = self.get_object()
        
        # Check and update evaluation if needed (for paper trading)
        if deployment.deployment_type == 'paper' and deployment.status in ['evaluating', 'active']:
            EvaluationService.check_and_update_evaluation(deployment)
            deployment.refresh_from_db()  # Refresh to get updated evaluation results
        
        closed_trades = deployment.live_trades.filter(status='closed')
        total_pnl = sum(float(trade.pnl or 0) for trade in closed_trades)
        
        stats = {
            'total_trades': deployment.live_trades.count(),
            'open_trades': deployment.live_trades.filter(status='open').count(),
            'closed_trades': closed_trades.count(),
            'total_pnl': total_pnl,
            'evaluation_results': deployment.evaluation_results if deployment.evaluation_results else None,
            'evaluation_criteria': deployment.evaluation_criteria,
            'status': deployment.status,
        }
        
        return Response(stats)
    
    @action(detail=True, methods=['post'], url_path='check-evaluation')
    def check_evaluation(self, request, pk=None):
        """Manually trigger evaluation check for paper trading deployment"""
        deployment = self.get_object()
        
        if deployment.deployment_type != 'paper':
            return Response(
                {'error': 'Can only evaluate paper trading deployments'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        results = EvaluationService.check_and_update_evaluation(deployment)
        
        if results is None:
            # Not ready for evaluation yet
            closed_trades_count = deployment.live_trades.filter(status='closed').count()
            min_trades = deployment.evaluation_criteria.get('min_trades', 0)
            return Response({
                'message': f'Not ready for evaluation yet. {closed_trades_count}/{min_trades} trades completed.',
                'trades_completed': closed_trades_count,
                'min_trades_required': min_trades
            })
        
        deployment.refresh_from_db()
        serializer = self.get_serializer(deployment)
        return Response({
            'message': 'Evaluation completed',
            'results': results,
            'deployment': serializer.data
        })


class LiveTradeViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for LiveTrade (read-only for now)"""
    queryset = LiveTrade.objects.select_related('deployment', 'symbol').all()
    serializer_class = LiveTradeSerializer
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['symbol__ticker', 'deployment__name', 'broker_order_id']
    ordering_fields = ['entry_timestamp', 'exit_timestamp', 'pnl']
    ordering = ['-entry_timestamp']
    
    def get_queryset(self):
        """Filter by deployment, symbol, status, or deployment_type if provided"""
        queryset = super().get_queryset()
        
        deployment_id = self.request.query_params.get('deployment', None)
        if deployment_id:
            queryset = queryset.filter(deployment_id=deployment_id)
        
        symbol_ticker = self.request.query_params.get('symbol', None)
        if symbol_ticker:
            queryset = queryset.filter(symbol__ticker=symbol_ticker)
        
        status_filter = self.request.query_params.get('status', None)
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        # Filter by deployment type (paper or real_money)
        deployment_type = self.request.query_params.get('deployment_type', None)
        if deployment_type:
            queryset = queryset.filter(deployment__deployment_type=deployment_type)
        
        return queryset
