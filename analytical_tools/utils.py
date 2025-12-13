"""
Utility functions for analytical tools
"""

from .models import ToolDefinition
from .config import get_all_indicator_definitions


def ensure_tool_definitions_exist():
    """
    Ensure all indicator definitions from config exist in the database.
    This is called automatically when needed.
    """
    definitions = get_all_indicator_definitions()
    
    for definition in definitions:
        ToolDefinition.objects.update_or_create(
            name=definition['name'],
            defaults={
                'description': definition['description'],
                'category': definition['category'],
                'default_parameters': definition['default_parameters'],
            }
        )







