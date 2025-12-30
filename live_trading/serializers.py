"""
Serializers for Live Trading models
"""

from rest_framework import serializers
from .models import Broker, SymbolBrokerAssociation, LiveTradingDeployment, LiveTrade
from market_data.serializers import SymbolListSerializer
from backtest_engine.serializers import BacktestListSerializer


class BrokerSerializer(serializers.ModelSerializer):
    """Serializer for Broker"""
    has_paper_trading = serializers.SerializerMethodField()
    has_real_money = serializers.SerializerMethodField()
    
    class Meta:
        model = Broker
        fields = [
            'id', 'name', 'code',
            'paper_trading_endpoint_url', 'paper_trading_api_key', 'paper_trading_secret_key', 'paper_trading_active',
            'real_money_endpoint_url', 'real_money_api_key', 'real_money_secret_key', 'real_money_active',
            'api_config', 'has_paper_trading', 'has_real_money', 'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at', 'has_paper_trading', 'has_real_money']
        # Note: API keys are included in serialization for editing purposes
        # In production, ensure proper authentication/authorization is in place
    
    def to_representation(self, instance):
        """Override to always include API keys for editing purposes"""
        data = super().to_representation(instance)
        # Include API keys and secrets in the representation so they can be edited
        # This allows the frontend to pre-fill the form when editing
        return data
    
    def get_has_paper_trading(self, obj):
        """Check if paper trading credentials are configured"""
        return obj.has_paper_trading_credentials()
    
    def get_has_real_money(self, obj):
        """Check if real money credentials are configured"""
        return obj.has_real_money_credentials()


class SymbolBrokerAssociationSerializer(serializers.ModelSerializer):
    """Serializer for SymbolBrokerAssociation"""
    symbol_info = SymbolListSerializer(source='symbol', read_only=True)
    broker_name = serializers.CharField(source='broker.name', read_only=True)
    broker_code = serializers.CharField(source='broker.code', read_only=True)
    
    class Meta:
        model = SymbolBrokerAssociation
        fields = [
            'id', 'symbol', 'symbol_info', 'broker', 'broker_name', 'broker_code',
            'long_active', 'short_active', 'verified_at', 'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at', 'verified_at']


class LiveTradeSerializer(serializers.ModelSerializer):
    """Serializer for LiveTrade"""
    symbol_info = SymbolListSerializer(source='symbol', read_only=True)
    deployment_name = serializers.CharField(source='deployment.name', read_only=True)
    
    class Meta:
        model = LiveTrade
        fields = [
            'id', 'deployment', 'deployment_name', 'symbol', 'symbol_info',
            'position_mode', 'trade_type', 'entry_price', 'exit_price',
            'quantity', 'entry_timestamp', 'exit_timestamp', 'pnl', 'pnl_percentage',
            'is_winner', 'status', 'broker_order_id', 'metadata',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']


class LiveTradingDeploymentListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for LiveTradingDeployment list views"""
    broker_name = serializers.CharField(source='broker.name', read_only=True)
    broker_code = serializers.CharField(source='broker.code', read_only=True)
    backtest_name = serializers.CharField(source='backtest.name', read_only=True)
    strategy_name = serializers.CharField(source='backtest.strategy.name', read_only=True)
    symbols_count = serializers.SerializerMethodField()
    
    class Meta:
        model = LiveTradingDeployment
        fields = [
            'id', 'name', 'backtest', 'backtest_name', 'strategy_name',
            'position_mode', 'broker', 'broker_name', 'broker_code',
            'deployment_type', 'status', 'symbols_count',
            'started_at', 'evaluated_at', 'activated_at', 'created_at'
        ]
        read_only_fields = ['created_at', 'updated_at', 'started_at', 'evaluated_at', 'activated_at']
    
    def get_symbols_count(self, obj):
        """Get count of symbols in deployment"""
        return obj.symbols.count()


class LiveTradingDeploymentDetailSerializer(serializers.ModelSerializer):
    """Detail serializer for LiveTradingDeployment"""
    broker_info = BrokerSerializer(source='broker', read_only=True)
    backtest_info = BacktestListSerializer(source='backtest', read_only=True)
    symbols = SymbolListSerializer(many=True, read_only=True)
    can_promote = serializers.SerializerMethodField()
    evaluation_passed = serializers.SerializerMethodField()
    
    class Meta:
        model = LiveTradingDeployment
        fields = [
            'id', 'name', 'backtest', 'backtest_info', 'position_mode',
            'broker', 'broker_info', 'symbols', 'deployment_type', 'status',
            'evaluation_criteria', 'evaluation_results', 'initial_capital',
            'bet_size_percentage', 'strategy_parameters', 'can_promote', 'evaluation_passed',
            'started_at', 'evaluated_at', 'activated_at', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'created_at', 'updated_at', 'started_at', 'evaluated_at',
            'activated_at', 'can_promote', 'evaluation_passed'
        ]
    
    def get_can_promote(self, obj):
        """Check if deployment can be promoted to real money"""
        return obj.can_promote_to_real_money()
    
    def get_evaluation_passed(self, obj):
        """Check if evaluation has passed"""
        return obj.has_evaluation_passed()


class LiveTradingDeploymentCreateSerializer(serializers.Serializer):
    """Serializer for creating a new live trading deployment"""
    name = serializers.CharField(required=False, allow_blank=True)
    backtest_id = serializers.IntegerField(required=True)
    position_mode = serializers.ChoiceField(
        choices=['all', 'long', 'short'],
        required=True
    )
    broker_id = serializers.IntegerField(required=True)
    symbol_tickers = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        allow_empty=True,
        help_text="List of symbol tickers. If empty, must provide exchange_code"
    )
    exchange_code = serializers.CharField(
        required=False,
        allow_blank=True,
        help_text="Exchange code to filter symbols by exchange"
    )
    evaluation_criteria = serializers.DictField(
        required=True,
        help_text="Evaluation criteria: {min_trades: int, min_sharpe_ratio: float, min_pnl: float}"
    )


class BrokerSymbolLinkSerializer(serializers.Serializer):
    """Serializer for linking symbols to a broker"""
    symbol_tickers = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        allow_empty=True,
        help_text="List of symbol tickers to link"
    )
    exchange_code = serializers.CharField(
        required=False,
        allow_blank=True,
        help_text="Exchange code to link all symbols from that exchange"
    )
    link_all_available = serializers.BooleanField(
        default=False,
        help_text="Link all available broker symbols that exist in database and have no broker association"
    )
    verify_capabilities = serializers.BooleanField(
        default=True,
        help_text="Whether to verify broker capabilities (long/short support) via API"
    )


