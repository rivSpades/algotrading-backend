"""
Strategies Models
Defines StrategyDefinition and StrategyAssignment models for managing trading strategies
"""

from django.db import models
from market_data.models import Symbol


class StrategyDefinition(models.Model):
    """Definition of a trading strategy"""
    name = models.CharField(max_length=100, unique=True)
    description_short = models.CharField(max_length=200, blank=True)
    description_long = models.TextField(blank=True)
    default_parameters = models.JSONField(
        default=dict,
        blank=True,
        help_text="Default strategy parameters (e.g., {'stop_loss': 0.02, 'take_profit': 0.05})"
    )
    analytic_tools_used = models.JSONField(
        default=list,
        blank=True,
        help_text="List of analytical tool names used by this strategy (e.g., ['SMA', 'RSI'])"
    )
    required_tool_configs = models.JSONField(
        default=list,
        blank=True,
        help_text="Required tool configurations with parameters and mappings. Format: [{'tool_name': 'SMA', 'parameters': {'period': 20}, 'parameter_mapping': {'period': 'fast_period'}, 'display_name': 'Fast SMA', 'locked': True}]"
    )
    example_code = models.TextField(
        blank=True,
        help_text="Optional example code or pseudocode for the strategy"
    )
    globally_enabled = models.BooleanField(
        default=False,
        help_text="If True, strategy is globally enabled (can be activated per symbol)"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['name']
        verbose_name = 'Strategy Definition'
        verbose_name_plural = 'Strategy Definitions'

    def __str__(self):
        return self.name


class StrategyAssignment(models.Model):
    """Assignment of a strategy to a symbol with specific parameters"""
    symbol = models.ForeignKey(
        Symbol,
        on_delete=models.CASCADE,
        related_name='strategy_assignments',
        null=True,
        blank=True,
        help_text="If None, this assignment applies to all symbols globally"
    )
    strategy = models.ForeignKey(
        StrategyDefinition,
        on_delete=models.CASCADE,
        related_name='assignments'
    )
    parameters = models.JSONField(
        default=dict,
        blank=True,
        help_text="Per-symbol parameter overrides (merges with strategy default_parameters)"
    )
    enabled = models.BooleanField(
        default=False,
        help_text="If True, strategy is active for this symbol"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        constraints = [
            models.UniqueConstraint(fields=['symbol', 'strategy'], name='unique_symbol_strategy'),
            models.UniqueConstraint(fields=['strategy'], condition=models.Q(symbol__isnull=True), name='unique_global_strategy'),
        ]
        indexes = [
            models.Index(fields=['symbol', 'enabled']),
            models.Index(fields=['strategy']),
            models.Index(fields=['enabled'], condition=models.Q(symbol__isnull=True), name='strategies_global_idx'),
        ]
        verbose_name = 'Strategy Assignment'
        verbose_name_plural = 'Strategy Assignments'

    def __str__(self):
        if self.symbol:
            return f"{self.symbol.ticker} - {self.strategy.name}"
        return f"Global - {self.strategy.name}"

    def get_effective_parameters(self):
        """Get merged parameters (default + per-symbol overrides)"""
        effective = self.strategy.default_parameters.copy()
        effective.update(self.parameters)
        return effective
