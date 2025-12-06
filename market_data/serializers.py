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
    """Serializer for CrontabSchedule"""
    class Meta:
        model = CrontabSchedule
        fields = ['minute', 'hour', 'day_of_week', 'day_of_month', 'month_of_year', 'timezone']


class IntervalScheduleSerializer(serializers.ModelSerializer):
    """Serializer for IntervalSchedule"""
    class Meta:
        model = IntervalSchedule
        fields = ['every', 'period']


class PeriodicTaskSerializer(serializers.ModelSerializer):
    """Serializer for PeriodicTask"""
    crontab = CrontabScheduleSerializer(read_only=True)
    interval = IntervalScheduleSerializer(read_only=True)
    crontab_id = serializers.IntegerField(write_only=True, required=False, allow_null=True)
    interval_id = serializers.IntegerField(write_only=True, required=False, allow_null=True)
    last_run_at = serializers.DateTimeField(read_only=True)
    total_run_count = serializers.IntegerField(read_only=True)
    kwargs = serializers.SerializerMethodField()

    class Meta:
        model = PeriodicTask
        fields = [
            'id', 'name', 'task', 'enabled', 'description',
            'crontab', 'interval', 'crontab_id', 'interval_id',
            'args', 'kwargs', 'queue', 'exchange', 'routing_key',
            'expires', 'one_off', 'start_time', 'last_run_at',
            'total_run_count', 'date_changed'
        ]
        read_only_fields = ['date_changed', 'last_run_at', 'total_run_count']

    def get_kwargs(self, obj):
        """Parse kwargs JSON string to dict"""
        if obj.kwargs:
            try:
                import json
                return json.loads(obj.kwargs)
            except:
                return obj.kwargs
        return {}

    def create(self, validated_data):
        crontab_id = validated_data.pop('crontab_id', None)
        interval_id = validated_data.pop('interval_id', None)
        kwargs = validated_data.pop('kwargs', {})
        
        if crontab_id:
            validated_data['crontab_id'] = crontab_id
        if interval_id:
            validated_data['interval_id'] = interval_id
        if kwargs:
            import json
            validated_data['kwargs'] = json.dumps(kwargs)
        
        return super().create(validated_data)

    def update(self, instance, validated_data):
        crontab_id = validated_data.pop('crontab_id', None)
        interval_id = validated_data.pop('interval_id', None)
        kwargs = validated_data.pop('kwargs', None)
        
        if crontab_id is not None:
            validated_data['crontab_id'] = crontab_id
        if interval_id is not None:
            validated_data['interval_id'] = interval_id
        if kwargs is not None:
            import json
            validated_data['kwargs'] = json.dumps(kwargs)
        
        return super().update(instance, validated_data)

