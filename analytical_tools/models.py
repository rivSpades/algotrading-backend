"""
Analytical Tools Models
Defines ToolDefinition and ToolAssignment models for managing indicators and analytical tools
"""

from django.db import models
from django.core.validators import MinValueValidator
from market_data.models import Symbol


class ToolDefinition(models.Model):
    """Definition of an analytical tool/indicator"""
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    default_parameters = models.JSONField(default=dict, blank=True)
    category = models.CharField(
        max_length=50,
        choices=[
            ('indicator', 'Technical Indicator'),
            ('statistical', 'Statistical Tool'),
            ('transformation', 'Data Transformation'),
        ],
        default='indicator'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['name']
        verbose_name = 'Tool Definition'
        verbose_name_plural = 'Tool Definitions'

    def __str__(self):
        return self.name


class ToolAssignment(models.Model):
    """Assignment of a tool to a symbol (or global) with specific parameters"""
    symbol = models.ForeignKey(
        Symbol, 
        on_delete=models.CASCADE, 
        related_name='tool_assignments',
        null=True,
        blank=True,
        help_text="If None, this assignment applies to all symbols globally"
    )
    tool = models.ForeignKey(ToolDefinition, on_delete=models.CASCADE, related_name='assignments')
    parameters = models.JSONField(default=dict, blank=True)
    enabled = models.BooleanField(default=True)
    subchart = models.BooleanField(
        default=False,
        help_text="If True, indicator will be displayed in a subchart below the main chart"
    )
    style = models.JSONField(
        default=dict,
        blank=True,
        help_text="Styling options: color, line_width, line_style, etc."
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        # unique_together doesn't work well with nullable fields, use UniqueConstraint instead
        constraints = [
            models.UniqueConstraint(fields=['symbol', 'tool'], name='unique_symbol_tool'),
            models.UniqueConstraint(fields=['tool'], condition=models.Q(symbol__isnull=True), name='unique_global_tool'),
        ]
        indexes = [
            models.Index(fields=['symbol', 'enabled']),
            models.Index(fields=['tool']),
            models.Index(fields=['enabled'], condition=models.Q(symbol__isnull=True), name='analyt_global_idx'),
        ]
        verbose_name = 'Tool Assignment'
        verbose_name_plural = 'Tool Assignments'

    def __str__(self):
        if self.symbol:
            return f"{self.symbol.ticker} - {self.tool.name}"
        return f"Global - {self.tool.name}"


class IndicatorValue(models.Model):
    """Stored computed indicator values for a symbol"""
    assignment = models.ForeignKey(ToolAssignment, on_delete=models.CASCADE, related_name='values')
    timestamp = models.DateTimeField()
    value = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Additional indicator-specific data (e.g., MACD signal, histogram)"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']
        unique_together = [['assignment', 'timestamp']]
        indexes = [
            models.Index(fields=['assignment', 'timestamp']),
            models.Index(fields=['assignment', '-timestamp']),
        ]
        verbose_name = 'Indicator Value'
        verbose_name_plural = 'Indicator Values'

    def __str__(self):
        return f"{self.assignment} - {self.timestamp}"
