"""Serializers for Live Trading models."""

from rest_framework import serializers

from market_data.serializers import SymbolListSerializer

from .models import (
    Broker,
    DeploymentEvent,
    DeploymentSymbol,
    LiveTrade,
    StrategyDeployment,
    SymbolBrokerAssociation,
)


class BrokerSerializer(serializers.ModelSerializer):
    """Serializer for Broker."""

    has_paper_trading = serializers.SerializerMethodField()
    has_real_money = serializers.SerializerMethodField()

    class Meta:
        model = Broker
        fields = [
            'id', 'name', 'code',
            'paper_trading_endpoint_url', 'paper_trading_api_key', 'paper_trading_secret_key',
            'paper_trading_active',
            'real_money_endpoint_url', 'real_money_api_key', 'real_money_secret_key',
            'real_money_active',
            'api_config', 'has_paper_trading', 'has_real_money',
            'created_at', 'updated_at',
        ]
        read_only_fields = ['created_at', 'updated_at', 'has_paper_trading', 'has_real_money']

    def get_has_paper_trading(self, obj):
        return obj.has_paper_trading_credentials()

    def get_has_real_money(self, obj):
        return obj.has_real_money_credentials()


class SymbolBrokerAssociationSerializer(serializers.ModelSerializer):
    """Serializer for SymbolBrokerAssociation."""

    symbol_info = SymbolListSerializer(source='symbol', read_only=True)
    broker_name = serializers.CharField(source='broker.name', read_only=True)
    broker_code = serializers.CharField(source='broker.code', read_only=True)

    class Meta:
        model = SymbolBrokerAssociation
        fields = [
            'id', 'symbol', 'symbol_info', 'broker', 'broker_name', 'broker_code',
            'long_active', 'short_active', 'verified_at', 'created_at', 'updated_at',
        ]
        read_only_fields = ['created_at', 'updated_at', 'verified_at']


class BrokerSymbolLinkSerializer(serializers.Serializer):
    """Serializer for linking symbols to a broker."""

    symbol_tickers = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        allow_empty=True,
        help_text="List of symbol tickers to link",
    )
    exchange_code = serializers.CharField(
        required=False,
        allow_blank=True,
        help_text="Exchange code to link all symbols from that exchange",
    )
    link_all_available = serializers.BooleanField(
        default=False,
        help_text=(
            "Link all available broker symbols that exist in database "
            "and have no broker association"
        ),
    )
    verify_capabilities = serializers.BooleanField(
        default=True,
        help_text="Whether to verify broker capabilities (long/short support) via API",
    )


class LiveTradeSerializer(serializers.ModelSerializer):
    """Serializer for LiveTrade."""

    symbol_info = SymbolListSerializer(source='symbol', read_only=True)
    deployment_name = serializers.CharField(source='deployment.name', read_only=True)
    deployment_type = serializers.CharField(source='deployment.deployment_type', read_only=True)
    hedge_legs = serializers.SerializerMethodField()

    def get_hedge_legs(self, obj):
        if self.context.get('skip_hedge_embed'):
            return []
        meta = obj.metadata or {}
        if meta.get('is_hedge_leg'):
            return []
        by_parent = self.context.get('hedge_by_parent')
        if not by_parent:
            return []
        legs = by_parent.get(obj.id, [])
        return LiveTradeSerializer(
            legs,
            many=True,
            context={**self.context, 'skip_hedge_embed': True},
        ).data

    class Meta:
        model = LiveTrade
        fields = [
            'id', 'deployment', 'deployment_name', 'deployment_type',
            'deployment_symbol', 'symbol', 'symbol_info',
            'position_mode', 'trade_type', 'entry_price', 'exit_price',
            'quantity', 'entry_timestamp', 'exit_timestamp', 'pnl', 'pnl_percentage',
            'is_winner', 'status', 'broker_order_id', 'metadata',
            'hedge_legs',
            'created_at', 'updated_at',
        ]
        read_only_fields = ['created_at', 'updated_at', 'hedge_legs']


# ---------------------------------------------------------------------------
# Strategy Deployment serializers (v2)
# ---------------------------------------------------------------------------


class DeploymentSymbolSerializer(serializers.ModelSerializer):
    """Serializer for `DeploymentSymbol` rows."""

    symbol_info = SymbolListSerializer(source='symbol', read_only=True)
    ticker = serializers.CharField(source='symbol.ticker', read_only=True)

    class Meta:
        model = DeploymentSymbol
        fields = [
            'id', 'deployment', 'symbol', 'ticker', 'symbol_info',
            'position_mode', 'status',
            'sharpe_long', 'sharpe_short', 'max_dd_long', 'max_dd_short',
            'total_trades_long', 'total_trades_short',
            'color_long', 'color_short', 'color_overall', 'tier',
            'priority', 'last_signal_at', 'last_evaluated_at',
            'created_at', 'updated_at',
        ]
        read_only_fields = [
            'created_at', 'updated_at',
            'sharpe_long', 'sharpe_short', 'max_dd_long', 'max_dd_short',
            'total_trades_long', 'total_trades_short',
            'color_long', 'color_short', 'color_overall', 'tier',
            'last_signal_at', 'last_evaluated_at',
        ]


class DeploymentEventSerializer(serializers.ModelSerializer):
    """Serializer for `DeploymentEvent` audit rows."""

    deployment_symbol_ticker = serializers.CharField(
        source='deployment_symbol.symbol.ticker', read_only=True, default=None,
    )
    deployment_name = serializers.CharField(
        source='deployment.name', read_only=True, default=None, allow_null=True,
    )

    class Meta:
        model = DeploymentEvent
        fields = [
            'id', 'deployment', 'deployment_name', 'deployment_symbol', 'deployment_symbol_ticker',
            'event_type', 'level', 'actor_type', 'actor_id',
            'message', 'context', 'error', 'created_at',
        ]
        read_only_fields = fields


class StrategyDeploymentListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for list views."""

    strategy_name = serializers.CharField(source='strategy.name', read_only=True)
    broker_name = serializers.CharField(source='broker.name', read_only=True)
    parameter_set_label = serializers.CharField(source='parameter_set.label', read_only=True)
    symbol_count = serializers.SerializerMethodField()
    active_symbol_count = serializers.SerializerMethodField()

    class Meta:
        model = StrategyDeployment
        fields = [
            'id', 'name', 'strategy', 'strategy_name',
            'parameter_set', 'parameter_set_label',
            'broker', 'broker_name',
            'position_mode', 'deployment_type', 'status',
            'initial_capital', 'bet_size_percentage',
            'hedge_enabled',
            'parent_deployment',
            'symbol_count', 'active_symbol_count',
            'last_signal_at', 'started_at', 'activated_at',
            'created_at', 'updated_at',
        ]
        read_only_fields = fields

    def get_symbol_count(self, obj):
        return obj.deployment_symbols.count()

    def get_active_symbol_count(self, obj):
        return obj.deployment_symbols.filter(status='active').count()


class StrategyDeploymentCoreSerializer(serializers.ModelSerializer):
    """Deployment detail for header + tabs: no heavy `deployment_symbols` list."""

    strategy_name = serializers.CharField(source='strategy.name', read_only=True)
    broker_name = serializers.CharField(source='broker.name', read_only=True)
    parameter_set_label = serializers.CharField(source='parameter_set.label', read_only=True)
    symbol_count = serializers.SerializerMethodField()
    active_symbol_count = serializers.SerializerMethodField()

    class Meta:
        model = StrategyDeployment
        fields = [
            'id', 'name', 'strategy', 'strategy_name',
            'parameter_set', 'parameter_set_label',
            'broker', 'broker_name',
            'position_mode', 'deployment_type', 'status',
            'initial_capital', 'bet_size_percentage', 'strategy_parameters',
            'hedge_enabled', 'hedge_config',
            'evaluation_criteria', 'evaluation_results',
            'parent_deployment',
            'symbol_count', 'active_symbol_count',
            'last_signal_at', 'last_error',
            'started_at', 'activated_at', 'evaluated_at',
            'created_at', 'updated_at',
        ]
        read_only_fields = [
            'strategy_name', 'broker_name', 'parameter_set_label',
            'symbol_count', 'active_symbol_count',
            'last_signal_at', 'last_error',
            'started_at', 'activated_at', 'evaluated_at',
            'created_at', 'updated_at',
            'evaluation_results',
        ]

    def get_symbol_count(self, obj):
        return obj.deployment_symbols.count()

    def get_active_symbol_count(self, obj):
        return obj.deployment_symbols.filter(status='active').count()


class StrategyDeploymentDetailSerializer(serializers.ModelSerializer):
    """Full deployment with nested symbols (e.g. internal or explicit include)."""

    strategy_name = serializers.CharField(source='strategy.name', read_only=True)
    broker_name = serializers.CharField(source='broker.name', read_only=True)
    parameter_set_label = serializers.CharField(source='parameter_set.label', read_only=True)
    deployment_symbols = DeploymentSymbolSerializer(many=True, read_only=True)

    class Meta:
        model = StrategyDeployment
        fields = [
            'id', 'name', 'strategy', 'strategy_name',
            'parameter_set', 'parameter_set_label',
            'broker', 'broker_name',
            'position_mode', 'deployment_type', 'status',
            'initial_capital', 'bet_size_percentage', 'strategy_parameters',
            'hedge_enabled', 'hedge_config',
            'evaluation_criteria', 'evaluation_results',
            'parent_deployment',
            'deployment_symbols',
            'last_signal_at', 'last_error',
            'started_at', 'activated_at', 'evaluated_at',
            'created_at', 'updated_at',
        ]
        read_only_fields = [
            'strategy_name', 'broker_name', 'parameter_set_label',
            'deployment_symbols',
            'last_signal_at', 'last_error',
            'started_at', 'activated_at', 'evaluated_at',
            'created_at', 'updated_at',
            'evaluation_results',
        ]


class SymbolOverrideSerializer(serializers.Serializer):
    """One symbol override entry sent in deployment create."""

    # Symbol's primary key is its ticker, but accept both keys for API ergonomics.
    symbol_id = serializers.CharField(required=False)
    ticker = serializers.CharField(required=False)
    position_mode = serializers.ChoiceField(
        choices=DeploymentSymbol.POSITION_MODE_CHOICES,
        required=False,
    )
    priority = serializers.IntegerField(required=False)

    def validate(self, attrs):
        if not attrs.get('symbol_id') and not attrs.get('ticker'):
            raise serializers.ValidationError("Provide either 'symbol_id' or 'ticker'.")
        return attrs


class StrategyDeploymentCreateSerializer(serializers.ModelSerializer):
    """Serializer used to create a new deployment.

    Always creates a `paper` deployment. The backend auto-fills
    `DeploymentSymbol` rows from the parameter-set snapshot symbols using the
    default green-bucket selector unless `symbol_overrides` is provided.
    """

    symbol_overrides = SymbolOverrideSerializer(many=True, required=False)
    use_default_symbols = serializers.BooleanField(
        default=True,
        help_text="If True, ignore overrides and seed deployment with default green selection.",
    )
    hedge_enabled = serializers.BooleanField(
        required=False,
        allow_null=True,
        help_text="If omitted, inherit from latest SymbolBacktestRun for this strategy+parameter set.",
    )
    hedge_config = serializers.JSONField(
        required=False,
        allow_null=True,
        help_text="Optional hedge overrides; merged with lab defaults when hedge_enabled.",
    )

    class Meta:
        model = StrategyDeployment
        fields = [
            'name', 'strategy', 'parameter_set', 'broker',
            'position_mode',
            'initial_capital', 'bet_size_percentage',
            'strategy_parameters', 'evaluation_criteria',
            'symbol_overrides', 'use_default_symbols',
            'hedge_enabled', 'hedge_config',
        ]

    def validate(self, attrs):
        strategy = attrs.get('strategy')
        parameter_set = attrs.get('parameter_set')
        if strategy and parameter_set and parameter_set.strategy_id != strategy.id:
            raise serializers.ValidationError(
                {'parameter_set': 'Parameter set does not belong to the selected strategy.'},
            )
        broker = attrs.get('broker')
        if broker and not broker.has_paper_trading_credentials():
            raise serializers.ValidationError(
                {'broker': 'Broker has no paper-trading credentials configured.'},
            )
        return attrs


class StrategyDeploymentPreviewSerializer(serializers.Serializer):
    """Input for the `preview-symbols` action."""

    parameter_set = serializers.CharField()
    position_mode = serializers.ChoiceField(
        choices=StrategyDeployment.POSITION_MODE_CHOICES,
        default='long',
    )
    default_only = serializers.BooleanField(default=True)
