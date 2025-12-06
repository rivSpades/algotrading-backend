"""
API Views for Strategies
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.filters import SearchFilter, OrderingFilter
from django.db.models import Q
from .models import StrategyDefinition, StrategyAssignment
from .serializers import StrategyDefinitionSerializer, StrategyAssignmentSerializer
from market_data.models import Symbol


class StrategyDefinitionViewSet(viewsets.ModelViewSet):
    """ViewSet for StrategyDefinition"""
    queryset = StrategyDefinition.objects.all()
    serializer_class = StrategyDefinitionSerializer
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['name', 'description_short', 'description_long']
    ordering_fields = ['name', 'created_at', 'updated_at']
    ordering = ['name']


class StrategyAssignmentViewSet(viewsets.ModelViewSet):
    """ViewSet for StrategyAssignment"""
    queryset = StrategyAssignment.objects.all().select_related('strategy', 'symbol')
    serializer_class = StrategyAssignmentSerializer
    filter_backends = [SearchFilter, OrderingFilter]
    ordering_fields = ['created_at', 'updated_at']
    ordering = ['-created_at']
    
    def get_queryset(self):
        queryset = super().get_queryset()
        symbol_ticker = self.request.query_params.get('symbol_ticker', None)
        
        if symbol_ticker:
            # Filter for symbol-specific assignments OR global assignments
            queryset = queryset.filter(Q(symbol__ticker=symbol_ticker) | Q(symbol__isnull=True))
        else:
            # If no symbol_ticker, only show global assignments
            queryset = queryset.filter(symbol__isnull=True)
        
        return queryset
    
    @action(detail=False, methods=['get'], url_path='symbol/(?P<ticker>[^/.]+)')
    def by_symbol(self, request, ticker=None):
        """Get all strategy assignments for a specific symbol (including global)"""
        try:
            symbol = Symbol.objects.get(ticker=ticker)
            assignments = StrategyAssignment.objects.filter(
                Q(symbol=symbol) | Q(symbol__isnull=True)
            ).select_related('strategy', 'symbol')
            serializer = self.get_serializer(assignments, many=True)
            return Response(serializer.data)
        except Symbol.DoesNotExist:
            return Response(
                {'error': f'Symbol {ticker} not found'},
                status=status.HTTP_404_NOT_FOUND
            )
