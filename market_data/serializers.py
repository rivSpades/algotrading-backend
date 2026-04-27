"""
Serializers for Market Data API
"""

from rest_framework import serializers
from django_celery_beat.models import PeriodicTask, CrontabSchedule, IntervalSchedule
from .models import Symbol, Exchange, Provider, OHLCV


class ExchangeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Exchange
        fields = ['name', 'code', 'country', 'timezone']


class ProviderSerializer(serializers.ModelSerializer):
    class Meta:
        model = Provider
        fields = ['name', 'code', 'is_active']
        read_only_fields = ['is_active']


class OHLCVSerializer(serializers.ModelSerializer):
    class Meta:
        model = OHLCV
        fields = ['timestamp', 'timeframe', 'open', 'high', 'low', 'close', 'volume']


class SymbolSerializer(serializers.ModelSerializer):
    exchange = ExchangeSerializer(read_only=True)
    provider = ProviderSerializer(read_only=True)
    exchange_code = serializers.CharField(write_only=True, required=False)
    provider_code = serializers.CharField(write_only=True, required=False)

    class Meta:
        model = Symbol
        fields = [
            'ticker', 'exchange', 'provider', 'type', 'status', 'name',
            'description', 'validation_status', 'validation_reason',
            'last_updated', 'created_at', 'updated_at',
            'exchange_code', 'provider_code'
        ]
        read_only_fields = ['created_at', 'updated_at', 'last_updated']

    def create(self, validated_data):
        exchange_code = validated_data.pop('exchange_code', None)
        provider_code = validated_data.pop('provider_code', None)

        if exchange_code:
            exchange = Exchange.objects.get(code=exchange_code)
            validated_data['exchange'] = exchange

        if provider_code:
            provider = Provider.objects.get(code=provider_code)
            validated_data['provider'] = provider

        return super().create(validated_data)

    def update(self, instance, validated_data):
        exchange_code = validated_data.pop('exchange_code', None)
        provider_code = validated_data.pop('provider_code', None)

        if exchange_code:
            exchange = Exchange.objects.get(code=exchange_code)
            validated_data['exchange'] = exchange

        if provider_code:
            provider = Provider.objects.get(code=provider_code)
            validated_data['provider'] = provider

        return super().update(instance, validated_data)


class SymbolListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for list views"""
    exchange = serializers.CharField(source='exchange.code', read_only=True)
    exchange_name = serializers.CharField(source='exchange.name', read_only=True)

    class Meta:
        model = Symbol
        fields = [
            'ticker', 'exchange', 'exchange_name', 'type', 'status',
            'name', 'validation_status', 'validation_reason', 'last_updated'
        ]


class CrontabScheduleSerializer(serializers.ModelSerializer):
    """CrontabSchedule; `timezone` is TimeZoneField (ZoneInfo) — stringify for JSON."""

    class Meta:
        model = CrontabSchedule
        fields = [
            'id', 'minute', 'hour', 'day_of_week', 'day_of_month',
            'month_of_year', 'timezone',
        ]

    def to_representation(self, instance):
        data = super().to_representation(instance)
        if data.get('timezone') is not None:
            data['timezone'] = str(data['timezone'])
        return data


class IntervalScheduleSerializer(serializers.ModelSerializer):
    """Serializer for IntervalSchedule"""
    class Meta:
        model = IntervalSchedule
        fields = ['id', 'every', 'period']


class CeleryTaskJSONTextField(serializers.Field):
    """PeriodicTask `args` / `kwargs` are JSON stored in TextFields; API uses objects."""

    def __init__(self, *, is_list: bool = False, **kwargs):
        self._is_list = is_list
        super().__init__(**kwargs)

    def to_representation(self, value):
        import json
        if value in (None, ''):
            return [] if self._is_list else {}
        if isinstance(value, (list, dict)):
            return value
        try:
            return json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return value

    def to_internal_value(self, data):
        import json
        if data is None:
            return None
        if isinstance(data, str):
            return data
        return json.dumps(data)


class PeriodicTaskSerializer(serializers.ModelSerializer):
    """Serializer for PeriodicTask"""
    crontab = CrontabScheduleSerializer(read_only=True)
    interval = IntervalScheduleSerializer(read_only=True)
    crontab_id = serializers.IntegerField(
        required=False, allow_null=True, write_only=True,
    )
    interval_id = serializers.IntegerField(
        required=False, allow_null=True, write_only=True,
    )
    last_run_at = serializers.DateTimeField(read_only=True)
    total_run_count = serializers.IntegerField(read_only=True)
    args = CeleryTaskJSONTextField(is_list=True, required=False, allow_null=True)
    kwargs = CeleryTaskJSONTextField(is_list=False, required=False, allow_null=True)
    schedule_type = serializers.SerializerMethodField()

    class Meta:
        model = PeriodicTask
        fields = [
            'id', 'name', 'task', 'enabled', 'description',
            'crontab', 'interval', 'crontab_id', 'interval_id',
            'schedule_type',
            'args', 'kwargs', 'queue', 'exchange', 'routing_key',
            'expires', 'one_off', 'start_time', 'last_run_at',
            'total_run_count', 'date_changed',
        ]
        read_only_fields = ['date_changed', 'last_run_at', 'total_run_count', 'schedule_type']

    def get_schedule_type(self, obj) -> str:
        if obj.solar_id:
            return 'solar'
        if obj.clocked_id:
            return 'clocked'
        if obj.crontab_id:
            return 'crontab'
        if obj.interval_id:
            return 'interval'
        return 'none'

