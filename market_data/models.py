"""
Market Data Models
Defines Symbol, Exchange, Provider, and OHLCV data models
"""

from django.db import models
from django.core.validators import MinLengthValidator
from django.utils import timezone


class Exchange(models.Model):
    """Exchange model (e.g., NYSE, NASDAQ, CRYPTO-BTC)"""
    name = models.CharField(max_length=100, unique=True)
    code = models.CharField(max_length=20, unique=True)
    country = models.CharField(max_length=100, blank=True)
    timezone = models.CharField(max_length=50, default='UTC')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return f"{self.name} ({self.code})"


class Provider(models.Model):
    """Data provider model (e.g., Alpha Vantage, Yahoo Finance, Polygon)"""
    name = models.CharField(max_length=100, unique=True)
    code = models.CharField(max_length=50, unique=True)
    api_key = models.CharField(max_length=255, blank=True, null=True, help_text="API key or access key")
    base_url = models.URLField(blank=True, null=True, help_text="Base URL for API or S3 endpoint")
    # Additional fields for S3-based providers (e.g., Polygon)
    access_key_id = models.CharField(max_length=255, blank=True, null=True, help_text="S3 Access Key ID")
    secret_access_key = models.CharField(max_length=255, blank=True, null=True, help_text="S3 Secret Access Key")
    endpoint_url = models.URLField(blank=True, null=True, help_text="S3 endpoint URL (e.g., https://files.massive.com)")
    bucket_name = models.CharField(max_length=255, blank=True, null=True, help_text="S3 bucket name for flat files")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return self.name


class Symbol(models.Model):
    """Symbol model - unique by ticker"""
    SYMBOL_TYPES = [
        ('stock', 'Stock'),
        ('crypto', 'Cryptocurrency'),
        ('etf', 'ETF'),
        ('forex', 'Forex'),
    ]

    STATUS_CHOICES = [
        ('active', 'Active'),
        ('disabled', 'Disabled'),
    ]

    ticker = models.CharField(
        max_length=20,
        unique=True,
        validators=[MinLengthValidator(1)],
        primary_key=True
    )
    exchange = models.ForeignKey(Exchange, on_delete=models.PROTECT, related_name='symbols')
    provider = models.ForeignKey(Provider, on_delete=models.PROTECT, related_name='symbols', null=True, blank=True)
    type = models.CharField(max_length=10, choices=SYMBOL_TYPES, default='stock')
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='disabled')
    name = models.CharField(max_length=200, blank=True)
    description = models.TextField(blank=True)
    validation_status = models.CharField(
        max_length=20,
        choices=[
            ('valid', 'Valid'),
            ('invalid', 'Invalid'),
            ('pending', 'Pending Validation'),
        ],
        default='pending',
        help_text="Data validation status"
    )
    validation_reason = models.TextField(
        blank=True,
        help_text="Reason why data validation failed (if invalid)"
    )
    last_updated = models.DateTimeField(default=timezone.now)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['ticker']
        indexes = [
            models.Index(fields=['ticker']),
            models.Index(fields=['exchange']),
            models.Index(fields=['provider']),
            models.Index(fields=['status']),
        ]

    def __str__(self):
        return f"{self.ticker} ({self.exchange.code})"


class OHLCV(models.Model):
    """OHLCV (Open, High, Low, Close, Volume) data model"""
    TIMEFRAME_CHOICES = [
        ('daily', 'Daily'),
        ('hourly', 'Hourly'),
        ('minute', 'Minute'),
    ]

    symbol = models.ForeignKey(Symbol, on_delete=models.CASCADE, related_name='ohlcv_data')
    timestamp = models.DateTimeField()
    timeframe = models.CharField(max_length=10, choices=TIMEFRAME_CHOICES, default='daily')
    open = models.DecimalField(max_digits=20, decimal_places=8)
    high = models.DecimalField(max_digits=20, decimal_places=8)
    low = models.DecimalField(max_digits=20, decimal_places=8)
    close = models.DecimalField(max_digits=20, decimal_places=8)
    volume = models.BigIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']
        unique_together = [['symbol', 'timestamp', 'timeframe']]
        indexes = [
            models.Index(fields=['symbol', 'timestamp', 'timeframe']),
            models.Index(fields=['symbol', 'timeframe', '-timestamp']),
        ]

    def __str__(self):
        return f"{self.symbol.ticker} - {self.timestamp} ({self.timeframe})"
