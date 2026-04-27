"""
Live Trading Models
Defines Broker, SymbolBrokerAssociation, StrategyDeployment, DeploymentSymbol,
DeploymentEvent, and LiveTrade models.

The legacy `LiveTradingDeployment` model has been replaced by `StrategyDeployment`
which is anchored to a `SymbolBacktestParameterSet` (the global single-symbol
backtest parent) instead of a portfolio `Backtest`.
"""

from django.db import models
from django.core.validators import MinValueValidator
from market_data.models import Symbol
from strategies.models import StrategyDefinition


class Broker(models.Model):
    """Broker model for managing trading broker connections"""
    name = models.CharField(max_length=100, unique=True, help_text="Broker name (e.g., Alpaca, Interactive Brokers)")
    code = models.CharField(
        max_length=50,
        unique=True,
        help_text="Unique broker code identifier (e.g., 'ALPACA', 'IB')"
    )
    # Paper Trading Credentials
    paper_trading_endpoint_url = models.URLField(
        max_length=500,
        blank=True,
        null=True,
        help_text="Paper trading API endpoint URL (e.g., https://paper-api.alpaca.markets)"
    )
    paper_trading_api_key = models.CharField(
        max_length=500,
        blank=True,
        null=True,
        help_text="Paper trading API key (should be encrypted in production)"
    )
    paper_trading_secret_key = models.CharField(
        max_length=500,
        blank=True,
        null=True,
        help_text="Paper trading API secret key (should be encrypted in production)"
    )
    paper_trading_active = models.BooleanField(
        default=False,
        help_text="Whether paper trading credentials are tested and active"
    )
    # Real Money Trading Credentials
    real_money_endpoint_url = models.URLField(
        max_length=500,
        blank=True,
        null=True,
        help_text="Real money trading API endpoint URL (e.g., https://api.alpaca.markets)"
    )
    real_money_api_key = models.CharField(
        max_length=500,
        blank=True,
        null=True,
        help_text="Real money trading API key (should be encrypted in production)"
    )
    real_money_secret_key = models.CharField(
        max_length=500,
        blank=True,
        null=True,
        help_text="Real money trading API secret key (should be encrypted in production)"
    )
    real_money_active = models.BooleanField(
        default=False,
        help_text="Whether real money trading credentials are tested and active"
    )
    api_config = models.JSONField(
        default=dict,
        blank=True,
        help_text="Broker-specific configuration (e.g., timeout, rate_limits)"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['name']
        indexes = [
            models.Index(fields=['code']),
            models.Index(fields=['paper_trading_active']),
            models.Index(fields=['real_money_active']),
        ]
        verbose_name = 'Broker'
        verbose_name_plural = 'Brokers'

    def __str__(self):
        return f"{self.name} ({self.code})"

    def has_paper_trading_credentials(self):
        return bool(
            self.paper_trading_api_key
            and self.paper_trading_secret_key
            and self.paper_trading_endpoint_url
        )

    def has_real_money_credentials(self):
        return bool(
            self.real_money_api_key
            and self.real_money_secret_key
            and self.real_money_endpoint_url
        )

    def has_credentials(self, deployment_type='paper'):
        """Return True if credentials for the given deployment type are configured."""
        if deployment_type == 'real_money':
            return self.has_real_money_credentials()
        return self.has_paper_trading_credentials()

    def is_active_for_deployment_type(self, deployment_type: str) -> bool:
        if deployment_type == 'paper':
            return self.paper_trading_active and self.has_paper_trading_credentials()
        if deployment_type == 'real_money':
            return self.real_money_active and self.has_real_money_credentials()
        return False


class SymbolBrokerAssociation(models.Model):
    """Association between symbols and brokers with trading capabilities"""
    symbol = models.ForeignKey(
        Symbol,
        on_delete=models.CASCADE,
        related_name='broker_associations',
    )
    broker = models.ForeignKey(
        Broker,
        on_delete=models.CASCADE,
        related_name='symbol_associations',
    )
    long_active = models.BooleanField(default=False)
    short_active = models.BooleanField(default=False)
    verified_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['symbol__ticker', 'broker__name']
        unique_together = [['symbol', 'broker']]
        indexes = [
            models.Index(fields=['symbol', 'broker']),
            models.Index(fields=['broker', 'long_active']),
            models.Index(fields=['broker', 'short_active']),
        ]
        verbose_name = 'Symbol-Broker Association'
        verbose_name_plural = 'Symbol-Broker Associations'

    def __str__(self):
        capabilities = []
        if self.long_active:
            capabilities.append('LONG')
        if self.short_active:
            capabilities.append('SHORT')
        caps_str = '/'.join(capabilities) if capabilities else 'NONE'
        return f"{self.symbol.ticker} @ {self.broker.name} ({caps_str})"

    def supports_all_modes(self):
        return self.long_active and self.short_active

    def supports_mode(self, mode):
        if mode == 'all':
            return self.supports_all_modes()
        if mode == 'long':
            return self.long_active
        if mode == 'short':
            return self.short_active
        return False


# ---------------------------------------------------------------------------
# Strategy Deployment (replaces legacy LiveTradingDeployment)
# ---------------------------------------------------------------------------


class StrategyDeployment(models.Model):
    """Deployment of a strategy parameter set to live (paper or real) trading.

    Each deployment is anchored to a `SymbolBacktestParameterSet` (the global
    single-symbol backtest parent) on a strategy and runs through a chosen
    `Broker`. Symbols are limited to the snapshot symbols of that parameter set.
    """

    DEPLOYMENT_TYPE_CHOICES = [
        ('paper', 'Paper Trading'),
        ('real_money', 'Real Money Trading'),
    ]

    POSITION_MODE_CHOICES = [
        ('all', 'All'),
        ('long', 'Long Only'),
        ('short', 'Short Only'),
    ]

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('active', 'Active'),
        ('evaluating', 'Evaluating (Paper Trading)'),
        ('passed', 'Evaluation Passed'),
        ('failed', 'Evaluation Failed'),
        ('paused', 'Paused'),
        ('stopped', 'Stopped'),
    ]

    name = models.CharField(max_length=200, blank=True, help_text="Optional name for this deployment")
    strategy = models.ForeignKey(
        StrategyDefinition,
        on_delete=models.CASCADE,
        related_name='deployments',
    )
    parameter_set = models.ForeignKey(
        'backtest_engine.SymbolBacktestParameterSet',
        on_delete=models.PROTECT,
        related_name='deployments',
        help_text="Global single-symbol backtest parent that defines configuration",
    )
    broker = models.ForeignKey(
        Broker,
        on_delete=models.PROTECT,
        related_name='strategy_deployments',
    )
    position_mode = models.CharField(
        max_length=10,
        choices=POSITION_MODE_CHOICES,
        default='long',
    )
    deployment_type = models.CharField(
        max_length=20,
        choices=DEPLOYMENT_TYPE_CHOICES,
        default='paper',
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending',
    )

    initial_capital = models.DecimalField(
        max_digits=20,
        decimal_places=2,
        default=10000.0,
        validators=[MinValueValidator(0.01)],
    )
    bet_size_percentage = models.FloatField(default=100.0)
    strategy_parameters = models.JSONField(default=dict, blank=True)

    hedge_enabled = models.BooleanField(
        default=False,
        help_text=(
            'If true, split each entry bet between the strategy symbol and a VIX sleeve '
            '(same hybrid logic as backtests; inherited from symbol runs on deploy).'
        ),
    )
    hedge_config = models.JSONField(
        default=dict,
        blank=True,
        help_text='Hybrid VIX hedge parameters (merged with defaults at order time).',
    )

    # Evaluation
    evaluation_criteria = models.JSONField(default=dict, blank=True)
    evaluation_results = models.JSONField(default=dict, blank=True)

    # Promotion link
    parent_deployment = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='child_deployments',
        help_text="Set when this real-money deployment was promoted from a paper deployment",
    )

    started_at = models.DateTimeField(null=True, blank=True)
    activated_at = models.DateTimeField(null=True, blank=True)
    evaluated_at = models.DateTimeField(null=True, blank=True)
    last_signal_at = models.DateTimeField(null=True, blank=True)
    last_error = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['strategy', 'status']),
            models.Index(fields=['broker', 'status']),
            models.Index(fields=['parameter_set']),
            models.Index(fields=['deployment_type', 'status']),
            models.Index(fields=['-created_at']),
        ]
        verbose_name = 'Strategy Deployment'
        verbose_name_plural = 'Strategy Deployments'

    def __str__(self):
        return f"{self.strategy.name} [{self.deployment_type}] ({self.status})"

    def can_promote_to_real_money(self):
        # Promotion is allowed when a real-money account is configured. We do not
        # require evaluation gating here; evaluation remains optional analytics.
        return (
            self.deployment_type == 'paper'
            and self.broker.has_real_money_credentials()
        )

    def has_evaluation_passed(self):
        return bool(self.evaluation_results and self.evaluation_results.get('passed'))


class DeploymentSymbol(models.Model):
    """Symbol enrolled in a StrategyDeployment.

    Tracks the ordering tier (trade-count buckets), color status from the
    latest snapshot stats, and the lifecycle of the symbol within the
    deployment (active / flagged-for-disable / disabled / pending-enable).
    """

    STATUS_CHOICES = [
        ('active', 'Active'),
        ('flagged_for_disable', 'Flagged for Disable'),
        ('disabled', 'Disabled'),
        ('pending_enable', 'Pending Enable'),
    ]

    POSITION_MODE_CHOICES = StrategyDeployment.POSITION_MODE_CHOICES

    COLOR_CHOICES = [
        ('green', 'Green'),
        ('yellow', 'Yellow'),
        ('orange', 'Orange'),
        ('red', 'Red'),
        ('black', 'Black'),
        ('gray', 'Unknown'),
    ]

    TIER_CHOICES = [
        ('gt50', 'Trades > 50'),
        ('gt20', 'Trades > 20'),
        ('gt10', 'Trades > 10'),
        ('gt0', 'Trades > 0'),
        ('none', 'No trades'),
    ]

    deployment = models.ForeignKey(
        StrategyDeployment,
        on_delete=models.CASCADE,
        related_name='deployment_symbols',
    )
    symbol = models.ForeignKey(
        Symbol,
        on_delete=models.CASCADE,
        related_name='deployment_symbols',
    )
    position_mode = models.CharField(max_length=10, choices=POSITION_MODE_CHOICES, default='long')
    status = models.CharField(max_length=24, choices=STATUS_CHOICES, default='active')

    # Snapshot-derived stats (latest evaluation)
    sharpe_long = models.FloatField(null=True, blank=True)
    sharpe_short = models.FloatField(null=True, blank=True)
    max_dd_long = models.FloatField(null=True, blank=True)
    max_dd_short = models.FloatField(null=True, blank=True)
    total_trades_long = models.IntegerField(null=True, blank=True)
    total_trades_short = models.IntegerField(null=True, blank=True)

    color_long = models.CharField(max_length=10, choices=COLOR_CHOICES, default='gray')
    color_short = models.CharField(max_length=10, choices=COLOR_CHOICES, default='gray')
    color_overall = models.CharField(max_length=10, choices=COLOR_CHOICES, default='gray')
    tier = models.CharField(max_length=8, choices=TIER_CHOICES, default='none')

    priority = models.IntegerField(
        default=0,
        help_text="Order within the deployment (lower runs first); seeded by selection helper",
    )

    last_signal_at = models.DateTimeField(null=True, blank=True)
    last_evaluated_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['priority', 'symbol__ticker']
        unique_together = [['deployment', 'symbol']]
        indexes = [
            models.Index(fields=['deployment', 'status']),
            models.Index(fields=['deployment', 'priority']),
            models.Index(fields=['status']),
        ]
        verbose_name = 'Deployment Symbol'
        verbose_name_plural = 'Deployment Symbols'

    def __str__(self):
        return f"{self.deployment_id}/{self.symbol.ticker} [{self.status}]"


class DeploymentEvent(models.Model):
    """Append-only audit log for everything that happens to a deployment.

    Every action (manual or by a Celery task / system component) writes a row
    here so we can render a feed in the dashboard and trace failures.
    """

    EVENT_TYPES = [
        # Lifecycle
        ('deploy_created', 'Deployment Created'),
        ('deploy_activated', 'Deployment Activated'),
        ('deploy_paused', 'Deployment Paused'),
        ('deploy_stopped', 'Deployment Stopped'),
        ('deploy_failed', 'Deployment Failed'),
        ('promote_to_real', 'Promoted to Real Money'),

        # Symbols
        ('symbol_added', 'Symbol Added'),
        ('symbol_disabled', 'Symbol Disabled'),
        ('symbol_flagged', 'Symbol Flagged for Disable'),
        ('symbol_enabled', 'Symbol Enabled'),
        ('symbol_pending_enable', 'Symbol Pending Enable'),
        ('color_changed', 'Snapshot Color Changed'),

        # Live execution
        ('task_tick', 'Scheduled Task Tick'),
        ('signal_evaluated', 'Signal Evaluated'),
        ('order_placed', 'Order Placed'),
        ('order_filled', 'Order Filled'),
        ('order_failed', 'Order Failed'),
        ('trade_opened', 'Trade Opened'),
        ('trade_closed', 'Trade Closed'),
        ('positions_synced', 'Positions Synced'),

        # Recalc / evaluation
        ('recalc_started', 'Snapshot Recalc Started'),
        ('recalc_finished', 'Snapshot Recalc Finished'),
        ('evaluation_passed', 'Evaluation Passed'),
        ('evaluation_failed', 'Evaluation Failed'),

        # Generic
        ('error', 'Error'),
        ('info', 'Info'),
    ]

    LEVEL_CHOICES = [
        ('info', 'Info'),
        ('warning', 'Warning'),
        ('error', 'Error'),
    ]

    ACTOR_TYPES = [
        ('user', 'User'),
        ('task', 'Celery Task'),
        ('system', 'System'),
        ('broker', 'Broker'),
    ]

    deployment = models.ForeignKey(
        StrategyDeployment,
        on_delete=models.CASCADE,
        related_name='events',
        null=True,
        blank=True,
    )
    deployment_symbol = models.ForeignKey(
        DeploymentSymbol,
        on_delete=models.SET_NULL,
        related_name='events',
        null=True,
        blank=True,
    )
    event_type = models.CharField(max_length=40, choices=EVENT_TYPES)
    level = models.CharField(max_length=10, choices=LEVEL_CHOICES, default='info')
    actor_type = models.CharField(max_length=10, choices=ACTOR_TYPES, default='system')
    actor_id = models.CharField(
        max_length=200,
        blank=True,
        help_text="Identifier for who/what performed the action (user id, celery task id, etc.)",
    )
    message = models.CharField(max_length=500, blank=True)
    context = models.JSONField(default=dict, blank=True)
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['deployment', '-created_at']),
            models.Index(fields=['event_type', '-created_at']),
            models.Index(fields=['deployment', 'event_type', '-created_at']),
            models.Index(fields=['level', '-created_at']),
        ]
        verbose_name = 'Deployment Event'
        verbose_name_plural = 'Deployment Events'

    def __str__(self):
        target = f"deployment={self.deployment_id}" if self.deployment_id else 'system'
        return f"[{self.level}] {self.event_type} ({target})"


class LiveTrade(models.Model):
    """Live trade execution record (paper or real money)."""

    TRADE_TYPES = [
        ('buy', 'Buy'),
        ('sell', 'Sell'),
    ]

    STATUS_CHOICES = [
        ('open', 'Open'),
        ('closed', 'Closed'),
        ('cancelled', 'Cancelled'),
    ]

    deployment = models.ForeignKey(
        StrategyDeployment,
        on_delete=models.CASCADE,
        related_name='live_trades',
    )
    deployment_symbol = models.ForeignKey(
        DeploymentSymbol,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='live_trades',
    )
    symbol = models.ForeignKey(
        Symbol,
        on_delete=models.CASCADE,
        related_name='live_trades',
    )
    position_mode = models.CharField(
        max_length=10,
        choices=[('long', 'Long'), ('short', 'Short')],
    )
    trade_type = models.CharField(max_length=10, choices=TRADE_TYPES)
    entry_price = models.DecimalField(max_digits=20, decimal_places=8)
    exit_price = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    quantity = models.DecimalField(max_digits=20, decimal_places=8)
    entry_timestamp = models.DateTimeField()
    exit_timestamp = models.DateTimeField(null=True, blank=True)
    pnl = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    pnl_percentage = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)
    is_winner = models.BooleanField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='open')
    broker_order_id = models.CharField(max_length=200, null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-entry_timestamp']
        indexes = [
            models.Index(fields=['deployment', 'symbol']),
            models.Index(fields=['deployment', 'status']),
            models.Index(fields=['deployment', 'entry_timestamp']),
            models.Index(fields=['symbol', 'entry_timestamp']),
            models.Index(fields=['status']),
        ]
        verbose_name = 'Live Trade'
        verbose_name_plural = 'Live Trades'

    def __str__(self):
        status_str = f" [{self.status}]" if self.status != 'closed' else ""
        pnl_str = f" (PnL: {self.pnl})" if self.pnl is not None else ""
        return f"{self.symbol.ticker} {self.trade_type} @ {self.entry_price}{status_str}{pnl_str}"
