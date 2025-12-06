"""
API Views for Analytical Tools
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.filters import SearchFilter, OrderingFilter
from django.shortcuts import get_object_or_404
from .models import ToolAssignment, IndicatorValue
from .serializers import (
    ToolAssignmentSerializer,
    IndicatorValueSerializer
)
from market_data.models import Symbol, OHLCV
from .config import get_all_indicator_definitions, get_indicator_definition
import pandas as pd
from datetime import datetime
from rest_framework.views import APIView


class ToolDefinitionViewSet(viewsets.ViewSet):
    """
    ViewSet for hardcoded Tool Definitions
    Returns indicator definitions from config.py (not from database)
    """
    
    def list(self, request):
        """Return all hardcoded indicator definitions"""
        definitions = get_all_indicator_definitions()
        # Ensure definitions exist in DB for backward compatibility
        from .utils import ensure_tool_definitions_exist
        ensure_tool_definitions_exist()
        
        # Return hardcoded definitions with virtual IDs
        result = []
        for idx, definition in enumerate(definitions, start=1):
            result.append({
                'id': idx,  # Virtual ID for frontend
                'name': definition['name'],
                'description': definition['description'],
                'category': definition['category'],
                'default_parameters': definition['default_parameters'],
            })
        return Response(result)
    
    def retrieve(self, request, pk=None):
        """Return a specific indicator definition by name or virtual ID"""
        definitions = get_all_indicator_definitions()
        try:
            # Try to find by virtual ID first
            idx = int(pk) - 1
            if 0 <= idx < len(definitions):
                definition = definitions[idx]
                return Response({
                    'id': idx + 1,
                    'name': definition['name'],
                    'description': definition['description'],
                    'category': definition['category'],
                    'default_parameters': definition['default_parameters'],
                })
        except (ValueError, IndexError):
            pass
        
        # Try to find by name
        definition = get_indicator_definition(pk)
        if definition:
            definitions = get_all_indicator_definitions()
            idx = next(i for i, d in enumerate(definitions) if d['name'] == pk)
            return Response({
                'id': idx + 1,
                'name': definition['name'],
                'description': definition['description'],
                'category': definition['category'],
                'default_parameters': definition['default_parameters'],
            })
        
        return Response({'error': 'Tool definition not found'}, status=status.HTTP_404_NOT_FOUND)


class ToolAssignmentViewSet(viewsets.ModelViewSet):
    """ViewSet for ToolAssignment model"""
    queryset = ToolAssignment.objects.all()
    serializer_class = ToolAssignmentSerializer
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['symbol__ticker', 'tool__name']
    ordering_fields = ['created_at', 'enabled']
    ordering = ['-created_at']

    def get_queryset(self):
        """Filter by symbol if symbol_ticker is provided, or show global if requested"""
        queryset = super().get_queryset()
        symbol_ticker = self.request.query_params.get('symbol_ticker', None)
        show_global = self.request.query_params.get('global', 'false').lower() == 'true'
        
        if symbol_ticker:
            # Show both symbol-specific and global assignments for this symbol
            from django.db.models import Q
            queryset = queryset.filter(Q(symbol__ticker=symbol_ticker) | Q(symbol__isnull=True))
        elif show_global:
            # Show only global assignments
            queryset = queryset.filter(symbol__isnull=True)
        # If neither, show all assignments
        
        return queryset

    def perform_create(self, serializer):
        """Save assignment - indicators computed on-the-fly when fetching OHLCV"""
        serializer.save()

    def perform_update(self, serializer):
        """Save assignment - indicators computed on-the-fly when fetching OHLCV"""
        serializer.save()

    @action(detail=False, methods=['get'], url_path='symbol/(?P<symbol_ticker>[^/.]+)')
    def by_symbol(self, request, symbol_ticker=None):
        """Get all tool assignments for a specific symbol (includes global assignments)"""
        from django.db.models import Q
        assignments = self.queryset.filter(Q(symbol__ticker=symbol_ticker) | Q(symbol__isnull=True))
        serializer = self.get_serializer(assignments, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['post'], url_path='compute')
    def compute(self, request, pk=None):
        """Trigger synchronous computation of indicator values for this assignment"""
        assignment = self.get_object()
        
        if not assignment.enabled:
            return Response(
                {'error': 'Tool assignment is disabled'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Compute synchronously
        result = compute_indicator_sync(assignment)
        
        if result['status'] == 'error':
            return Response(
                {'error': result['message']},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        return Response({
            'status': result['status'],
            'message': result['message'],
            'values_created': result.get('values_created', 0)
        })

    @action(detail=False, methods=['post'], url_path='symbol/(?P<symbol_ticker>[^/.]+)/compute')
    def compute_all_for_symbol(self, request, symbol_ticker=None):
        """Compute all enabled indicators synchronously for a symbol"""
        symbol = get_object_or_404(Symbol, ticker=symbol_ticker)
        assignments = self.queryset.filter(symbol=symbol, enabled=True)
        
        if not assignments.exists():
            return Response(
                {'error': 'No enabled tool assignments found for this symbol'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        results = []
        for assignment in assignments:
            result = compute_indicator_sync(assignment)
            results.append(result)
        
        success_count = sum(1 for r in results if r['status'] == 'completed')
        
        return Response({
            'status': 'completed',
            'results': results,
            'success_count': success_count,
            'total_count': len(results),
            'message': f'Computed {success_count} out of {len(results)} indicators for {symbol_ticker}'
        })


class IndicatorValueViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for IndicatorValue model (read-only)"""
    queryset = IndicatorValue.objects.all()
    serializer_class = IndicatorValueSerializer
    filter_backends = [OrderingFilter]
    ordering_fields = ['timestamp']
    ordering = ['-timestamp']

    def get_queryset(self):
        """Filter by assignment if provided"""
        queryset = super().get_queryset()
        assignment_id = self.request.query_params.get('assignment_id', None)
        symbol_ticker = self.request.query_params.get('symbol_ticker', None)
        tool_name = self.request.query_params.get('tool_name', None)
        
        if assignment_id:
            queryset = queryset.filter(assignment_id=assignment_id)
        
        if symbol_ticker:
            queryset = queryset.filter(assignment__symbol__ticker=symbol_ticker)
        
        if tool_name:
            queryset = queryset.filter(assignment__tool__name=tool_name)
        
        return queryset

    @action(detail=False, methods=['get'], url_path='symbol/(?P<symbol_ticker>[^/.]+)/tool/(?P<tool_name>[^/.]+)')
    def by_symbol_and_tool(self, request, symbol_ticker=None, tool_name=None):
        """Get indicator values for a specific symbol and tool"""
        values = self.queryset.filter(
            assignment__symbol__ticker=symbol_ticker,
            assignment__tool__name=tool_name
        )
        serializer = self.get_serializer(values, many=True)
        return Response(serializer.data)
