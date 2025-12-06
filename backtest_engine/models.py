"""
Backtest Engine Models
Defines Backtest, Trade, and BacktestStatistics models
"""

from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from market_data.models import Symbol
from strategies.models import StrategyDefinition, StrategyAssignment


class Backtest(models.Model):
    """Backtest execution record"""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    name = models.CharField(max_length=200, blank=True, help_text="Optional name for this backtest")
    strategy = models.ForeignKey(StrategyDefinition, on_delete=models.CASCADE, related_name='backtests')
    strategy_assignment = models.ForeignKey(
        StrategyAssignment,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='backtests',
        help_text="Strategy assignment used (if available)"
    )
    symbols = models.ManyToManyField(Symbol, related_name='backtests', help_text="Symbols included in this backtest")
    start_date = models.DateTimeField(help_text="Backtest start date")
    end_date = models.DateTimeField(help_text="Backtest end date")
    split_ratio = models.FloatField(
        default=0.7,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Training/test split ratio (e.g., 0.7 = 70% training, 30% testing)"
    )
    initial_capital = models.DecimalField(
        max_digits=20,
        decimal_places=2,
        default=10000.0,
        validators=[MinValueValidator(0.01)],
        help_text="Initial capital for the backtest"
    )
    bet_size_percentage = models.FloatField(
        default=100.0,
        validators=[MinValueValidator(0.1), MaxValueValidator(100.0)],
        help_text="Percentage of available capital to bet per trade (0.1-100.0)"
    )
    strategy_parameters = models.JSONField(
        default=dict,
        blank=True,
        help_text="Strategy parameters used for this backtest"
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    error_message = models.TextField(blank=True, help_text="Error message if backtest failed")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['strategy', 'status']),
            models.Index(fields=['status']),
            models.Index(fields=['-created_at']),
        ]
        verbose_name = 'Backtest'
        verbose_name_plural = 'Backtests'

    def __str__(self):
        symbols_str = ', '.join([s.ticker for s in self.symbols.all()[:3]])
        if self.symbols.count() > 3:
            symbols_str += f" (+{self.symbols.count() - 3} more)"
        return f"{self.strategy.name} - {symbols_str} ({self.status})"


class Trade(models.Model):
    """Individual trade record from a backtest"""
    TRADE_TYPES = [
        ('buy', 'Buy'),
        ('sell', 'Sell'),
    ]

    backtest = models.ForeignKey(Backtest, on_delete=models.CASCADE, related_name='trades')
    symbol = models.ForeignKey(Symbol, on_delete=models.CASCADE, related_name='trades')
    trade_type = models.CharField(max_length=10, choices=TRADE_TYPES)
    entry_price = models.DecimalField(max_digits=20, decimal_places=8)
    exit_price = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    entry_timestamp = models.DateTimeField()
    exit_timestamp = models.DateTimeField(null=True, blank=True)
    quantity = models.DecimalField(max_digits=20, decimal_places=8, default=1.0)
    pnl = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True, help_text="Profit/Loss")
    pnl_percentage = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, help_text="PnL as percentage")
    is_winner = models.BooleanField(null=True, blank=True, help_text="True if profitable trade")
    max_drawdown = models.DecimalField(
        max_digits=10, 
        decimal_places=4, 
        null=True, 
        blank=True, 
        help_text="Maximum drawdown percentage during the trade period"
    )
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Additional trade metadata (e.g., signal strength, indicator values)"
    )

    class Meta:
        ordering = ['entry_timestamp']
        indexes = [
            models.Index(fields=['backtest', 'symbol']),
            models.Index(fields=['backtest', 'entry_timestamp']),
            models.Index(fields=['symbol', 'entry_timestamp']),
        ]
        verbose_name = 'Trade'
        verbose_name_plural = 'Trades'

    def __str__(self):
        return f"{self.symbol.ticker} {self.trade_type} @ {self.entry_price} (PnL: {self.pnl})"


class BacktestStatistics(models.Model):
    """Statistics calculated for a backtest"""
    backtest = models.ForeignKey(Backtest, on_delete=models.CASCADE, related_name='statistics')
    symbol = models.ForeignKey(
        Symbol,
        on_delete=models.CASCADE,
        related_name='backtest_statistics',
        null=True,
        blank=True,
        help_text="If None, these are portfolio-level statistics"
    )
    
    # Trade metrics
    total_trades = models.IntegerField(default=0)
    winning_trades = models.IntegerField(default=0)
    losing_trades = models.IntegerField(default=0)
    win_rate = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, help_text="Win rate as percentage")
    
    # PnL metrics
    total_pnl = models.DecimalField(max_digits=20, decimal_places=8, default=0)
    total_pnl_percentage = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)
    average_pnl = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    average_winner = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    average_loser = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    profit_factor = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)
    
    # Risk metrics
    max_drawdown = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, help_text="Maximum drawdown as percentage")
    max_drawdown_duration = models.IntegerField(null=True, blank=True, help_text="Duration of max drawdown in days")
    sharpe_ratio = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)
    
    # Performance metrics
    cagr = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, help_text="Compound Annual Growth Rate as percentage")
    total_return = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True, help_text="Total return as percentage")
    
    # Equity curve data (stored as JSON array of {timestamp, equity} objects)
    equity_curve = models.JSONField(
        default=list,
        blank=True,
        help_text="Equity curve data points: [{'timestamp': '2024-01-01', 'equity': 10000}, ...]"
    )
    
    # Additional statistics
    additional_stats = models.JSONField(
        default=dict,
        blank=True,
        help_text="Additional calculated statistics"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['backtest', 'symbol']),
            models.Index(fields=['backtest']),
        ]
        verbose_name = 'Backtest Statistics'
        verbose_name_plural = 'Backtest Statistics'

    def __str__(self):
        symbol_str = self.symbol.ticker if self.symbol else "Portfolio"
        return f"{self.backtest.strategy.name} - {symbol_str} Stats"
