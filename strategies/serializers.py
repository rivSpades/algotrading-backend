"""
Serializers for Strategy models
"""

from rest_framework import serializers
from .models import StrategyDefinition, StrategyAssignment
from market_data.serializers import SymbolListSerializer


class StrategyDefinitionSerializer(serializers.ModelSerializer):
    """Serializer for StrategyDefinition"""
    
    class Meta:
        model = StrategyDefinition
        fields = [
            'id', 'name', 'description_short', 'description_long',
            'default_parameters', 'analytic_tools_used', 'required_tool_configs',
            'example_code', 'globally_enabled', 'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']


class StrategyAssignmentSerializer(serializers.ModelSerializer):
    """Serializer for StrategyAssignment"""
    symbol_ticker = serializers.CharField(write_only=True, required=False)
    strategy_name = serializers.CharField(source='strategy.name', read_only=True)
    strategy_id = serializers.IntegerField(source='strategy.id', read_only=True)
    symbol_info = SymbolListSerializer(source='symbol', read_only=True)
    
    class Meta:
        model = StrategyAssignment
        fields = [
            'id', 'symbol', 'symbol_ticker', 'symbol_info',
            'strategy', 'strategy_id', 'strategy_name',
            'parameters', 'enabled', 'created_at', 'updated_at',
            'effective_parameters'
        ]
        read_only_fields = ['created_at', 'updated_at']
    
    effective_parameters = serializers.SerializerMethodField()
    
    def get_effective_parameters(self, obj):
        """Return merged parameters (default + overrides)"""
        return obj.get_effective_parameters()
    
    def create(self, validated_data):
        symbol_ticker = validated_data.pop('symbol_ticker', None)
        strategy_id = validated_data.pop('strategy_id', None)
        strategy_name = validated_data.pop('strategy_name', None)
        
        # Get or create strategy
        if strategy_id:
            from .models import StrategyDefinition
            strategy = StrategyDefinition.objects.get(id=strategy_id)
        elif strategy_name:
            from .models import StrategyDefinition
            strategy = StrategyDefinition.objects.get(name=strategy_name)
        else:
            strategy = validated_data.pop('strategy')
        
        validated_data['strategy'] = strategy
        
        # Handle symbol
        if symbol_ticker:
            from market_data.models import Symbol
            symbol = Symbol.objects.get(ticker=symbol_ticker)
            validated_data['symbol'] = symbol
        
        return super().create(validated_data)
    
    def update(self, instance, validated_data):
        symbol_ticker = validated_data.pop('symbol_ticker', None)
        strategy_id = validated_data.pop('strategy_id', None)
        strategy_name = validated_data.pop('strategy_name', None)
        
        # Handle strategy update
        if strategy_id:
            from .models import StrategyDefinition
            strategy = StrategyDefinition.objects.get(id=strategy_id)
            validated_data['strategy'] = strategy
        elif strategy_name:
            from .models import StrategyDefinition
            strategy = StrategyDefinition.objects.get(name=strategy_name)
            validated_data['strategy'] = strategy
        
        # Handle symbol update
        if symbol_ticker:
            from market_data.models import Symbol
            symbol = Symbol.objects.get(ticker=symbol_ticker)
            validated_data['symbol'] = symbol
        
        return super().update(instance, validated_data)

