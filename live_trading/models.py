"""
Live Trading Models
Defines Broker, SymbolBrokerAssociation, LiveTradingDeployment, and LiveTrade models
"""

from django.db import models
from django.core.validators import MinValueValidator
from django.utils import timezone
from market_data.models import Symbol, Exchange
from backtest_engine.models import Backtest
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
        """Check if paper trading credentials are configured"""
        return bool(
            self.paper_trading_api_key and 
            self.paper_trading_secret_key and 
            self.paper_trading_endpoint_url
        )
    
    def has_real_money_credentials(self):
        """Check if real money trading credentials are configured"""
        return bool(
            self.real_money_api_key and 
            self.real_money_secret_key and 
            self.real_money_endpoint_url
        )
    
    def is_active_for_deployment_type(self, deployment_type: str) -> bool:
        """
        Check if broker is active for a specific deployment type
        
        Args:
            deployment_type: 'paper' or 'real_money'
        
        Returns:
            True if broker has active credentials for the deployment type
        """
        if deployment_type == 'paper':
            return self.paper_trading_active and self.has_paper_trading_credentials()
        elif deployment_type == 'real_money':
            return self.real_money_active and self.has_real_money_credentials()
        return False


class SymbolBrokerAssociation(models.Model):
    """Association between symbols and brokers with trading capabilities"""
    symbol = models.ForeignKey(
        Symbol,
        on_delete=models.CASCADE,
        related_name='broker_associations',
        help_text="Symbol that can be traded on this broker"
    )
    broker = models.ForeignKey(
        Broker,
        on_delete=models.CASCADE,
        related_name='symbol_associations',
        help_text="Broker where this symbol can be traded"
    )
    long_active = models.BooleanField(
        default=False,
        help_text="Whether the symbol can be traded long on this broker"
    )
    short_active = models.BooleanField(
        default=False,
        help_text="Whether the symbol can be traded short on this broker"
    )
    verified_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When broker capabilities were last verified via API"
    )
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
        """Check if symbol supports both long and short on this broker"""
        return self.long_active and self.short_active

    def supports_mode(self, mode):
        """Check if symbol supports a specific position mode on this broker"""
        if mode == 'all':
            return self.supports_all_modes()
        elif mode == 'long':
            return self.long_active
        elif mode == 'short':
            return self.short_active
        return False


class LiveTradingDeployment(models.Model):
    """Live trading deployment configuration and status"""
    DEPLOYMENT_TYPE_CHOICES = [
        ('paper', 'Paper Trading'),
        ('real_money', 'Real Money Trading'),
    ]

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('evaluating', 'Evaluating (Paper Trading)'),
        ('passed', 'Evaluation Passed'),
        ('failed', 'Evaluation Failed'),
        ('active', 'Active'),
        ('paused', 'Paused'),
        ('stopped', 'Stopped'),
    ]

    name = models.CharField(
        max_length=200,
        blank=True,
        help_text="Optional name for this deployment"
    )
    backtest = models.ForeignKey(
        Backtest,
        on_delete=models.CASCADE,
        related_name='live_deployments',
        help_text="Backtest used as the basis for this deployment"
    )
    position_mode = models.CharField(
        max_length=10,
        choices=[
            ('all', 'All'),
            ('long', 'Long Only'),
            ('short', 'Short Only'),
        ],
        help_text="Position mode from the selected backtest"
    )
    broker = models.ForeignKey(
        Broker,
        on_delete=models.PROTECT,
        related_name='deployments',
        help_text="Broker used for this deployment"
    )
    symbols = models.ManyToManyField(
        Symbol,
        related_name='live_deployments',
        help_text="Symbols included in this deployment"
    )
    deployment_type = models.CharField(
        max_length=20,
        choices=DEPLOYMENT_TYPE_CHOICES,
        default='paper',
        help_text="Type of deployment (paper or real money)"
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending',
        help_text="Current deployment status"
    )
    # Evaluation criteria
    evaluation_criteria = models.JSONField(
        default=dict,
        help_text="Evaluation criteria: {min_trades: int, min_sharpe_ratio: float, min_pnl: float}"
    )
    # Evaluation results
    evaluation_results = models.JSONField(
        default=dict,
        blank=True,
        help_text="Evaluation results: {trades_count: int, sharpe_ratio: float, total_pnl: float, passed: bool, evaluated_at: datetime}"
    )
    # Deployment configuration (inherited from backtest)
    initial_capital = models.DecimalField(
        max_digits=20,
        decimal_places=2,
        help_text="Initial capital for the deployment (from backtest)"
    )
    bet_size_percentage = models.FloatField(
        help_text="Bet size percentage per trade (from backtest)"
    )
    strategy_parameters = models.JSONField(
        default=dict,
        blank=True,
        help_text="Strategy parameters (from backtest)"
    )
    started_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the deployment was started"
    )
    evaluated_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the evaluation was completed"
    )
    activated_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the deployment was activated (for real money)"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['broker', 'status']),
            models.Index(fields=['status']),
            models.Index(fields=['deployment_type', 'status']),
            models.Index(fields=['-created_at']),
        ]
        verbose_name = 'Live Trading Deployment'
        verbose_name_plural = 'Live Trading Deployments'

    def __str__(self):
        symbols_str = ', '.join([s.ticker for s in self.symbols.all()[:3]])
        if self.symbols.count() > 3:
            symbols_str += f" (+{self.symbols.count() - 3} more)"
        return f"{self.backtest.strategy.name} - {symbols_str} ({self.status})"

    def can_promote_to_real_money(self):
        """Check if deployment can be promoted to real money trading"""
        return (
            self.deployment_type == 'paper' and
            self.status == 'passed' and
            self.broker.is_active_for_deployment_type('real_money')
        )

    def has_evaluation_passed(self):
        """Check if evaluation criteria have been met"""
        if not self.evaluation_results:
            return False
        return self.evaluation_results.get('passed', False)


class LiveTrade(models.Model):
    """Live trade execution record"""
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
        LiveTradingDeployment,
        on_delete=models.CASCADE,
        related_name='live_trades',
        help_text="Deployment that executed this trade"
    )
    symbol = models.ForeignKey(
        Symbol,
        on_delete=models.CASCADE,
        related_name='live_trades',
        help_text="Symbol traded"
    )
    position_mode = models.CharField(
        max_length=10,
        choices=[
            ('long', 'Long'),
            ('short', 'Short'),
        ],
        help_text="Position mode for this trade"
    )
    trade_type = models.CharField(
        max_length=10,
        choices=TRADE_TYPES,
        help_text="Trade type (buy or sell)"
    )
    entry_price = models.DecimalField(
        max_digits=20,
        decimal_places=8,
        help_text="Entry price"
    )
    exit_price = models.DecimalField(
        max_digits=20,
        decimal_places=8,
        null=True,
        blank=True,
        help_text="Exit price (null if trade is still open)"
    )
    quantity = models.DecimalField(
        max_digits=20,
        decimal_places=8,
        help_text="Quantity traded"
    )
    entry_timestamp = models.DateTimeField(
        help_text="When the trade was entered"
    )
    exit_timestamp = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the trade was exited (null if trade is still open)"
    )
    pnl = models.DecimalField(
        max_digits=20,
        decimal_places=8,
        null=True,
        blank=True,
        help_text="Profit/Loss (null if trade is still open)"
    )
    pnl_percentage = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        null=True,
        blank=True,
        help_text="PnL as percentage"
    )
    is_winner = models.BooleanField(
        null=True,
        blank=True,
        help_text="True if profitable trade (null if trade is still open)"
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='open',
        help_text="Trade status"
    )
    broker_order_id = models.CharField(
        max_length=200,
        null=True,
        blank=True,
        help_text="Broker's order ID for reference"
    )
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Additional trade metadata (e.g., signal strength, fees, slippage)"
    )
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
