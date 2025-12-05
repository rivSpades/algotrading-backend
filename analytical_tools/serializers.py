"""
Serializers for Analytical Tools API
"""

from rest_framework import serializers
from .models import ToolDefinition, ToolAssignment, IndicatorValue
from market_data.serializers import SymbolListSerializer
from .utils import ensure_tool_definitions_exist
from .config import get_indicator_definition


class ToolDefinitionSerializer(serializers.ModelSerializer):
    """Serializer for ToolDefinition"""
    class Meta:
        model = ToolDefinition
        fields = [
            'id', 'name', 'description', 'default_parameters',
            'category', 'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']


class ToolAssignmentSerializer(serializers.ModelSerializer):
    """Serializer for ToolAssignment"""
    tool = ToolDefinitionSerializer(read_only=True)
    tool_id = serializers.IntegerField(write_only=True, required=False)
    tool_name = serializers.CharField(write_only=True, required=False)
    symbol = SymbolListSerializer(read_only=True)
    symbol_ticker = serializers.CharField(write_only=True, required=False)

    class Meta:
        model = ToolAssignment
        fields = [
            'id', 'symbol', 'symbol_ticker', 'tool', 'tool_id', 'tool_name',
            'parameters', 'enabled', 'subchart', 'style', 'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']

    def create(self, validated_data):
        symbol_ticker = validated_data.pop('symbol_ticker', None)
        tool_id = validated_data.pop('tool_id', None)
        tool_name = validated_data.pop('tool_name', None)
        
        # Ensure tool definitions exist in database
        ensure_tool_definitions_exist()
        
        if symbol_ticker:
            from market_data.models import Symbol
            symbol = Symbol.objects.get(ticker=symbol_ticker)
            validated_data['symbol'] = symbol
        
        # Get tool by ID or name, auto-create from config if needed
        if tool_id:
            tool = ToolDefinition.objects.get(id=tool_id)
        elif tool_name:
            # Ensure definitions exist, then get or create
            indicator_def = get_indicator_definition(tool_name)
            if not indicator_def:
                raise serializers.ValidationError(f"Unknown indicator: {tool_name}")
            tool, _ = ToolDefinition.objects.get_or_create(
                name=tool_name,
                defaults={
                    'description': indicator_def['description'],
                    'category': indicator_def['category'],
                    'default_parameters': indicator_def['default_parameters'],
                }
            )
        else:
            raise serializers.ValidationError("Either tool_id or tool_name must be provided")
        
        validated_data['tool'] = tool
        
        return super().create(validated_data)

    def update(self, instance, validated_data):
        symbol_ticker = validated_data.pop('symbol_ticker', None)
        tool_id = validated_data.pop('tool_id', None)
        tool_name = validated_data.pop('tool_name', None)
        
        if symbol_ticker:
            from market_data.models import Symbol
            symbol = Symbol.objects.get(ticker=symbol_ticker)
            validated_data['symbol'] = symbol
        
        if tool_id:
            tool = ToolDefinition.objects.get(id=tool_id)
            validated_data['tool'] = tool
        elif tool_name:
            # Ensure definitions exist, then get or create
            indicator_def = get_indicator_definition(tool_name)
            if indicator_def:
                tool, _ = ToolDefinition.objects.get_or_create(
                    name=tool_name,
                    defaults={
                        'description': indicator_def['description'],
                        'category': indicator_def['category'],
                        'default_parameters': indicator_def['default_parameters'],
                    }
                )
                validated_data['tool'] = tool
        
        return super().update(instance, validated_data)


class IndicatorValueSerializer(serializers.ModelSerializer):
    """Serializer for IndicatorValue"""
    assignment = ToolAssignmentSerializer(read_only=True)
    assignment_id = serializers.IntegerField(write_only=True)

    class Meta:
        model = IndicatorValue
        fields = [
            'id', 'assignment', 'assignment_id', 'timestamp',
            'value', 'metadata', 'created_at'
        ]
        read_only_fields = ['created_at']

    def create(self, validated_data):
        assignment_id = validated_data.pop('assignment_id')
        assignment = ToolAssignment.objects.get(id=assignment_id)
        validated_data['assignment'] = assignment
        return super().create(validated_data)

