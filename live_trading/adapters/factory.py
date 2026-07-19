"""
Broker Adapter Factory
Factory for creating broker adapter instances based on broker code
"""

from typing import Optional
from .base import BaseBrokerAdapter
from .alpaca import AlpacaBrokerAdapter
from ..models import Broker


# Registry of broker adapter classes
# Format: {broker_code: adapter_class}
_broker_adapters: dict[str, type[BaseBrokerAdapter]] = {}


def register_broker_adapter(broker_code: str, adapter_class: type[BaseBrokerAdapter]) -> None:
    """Register a broker adapter class.

    Args:
        broker_code: Broker code (e.g., 'ALPACA', 'IB')
        adapter_class: Class that inherits from BaseBrokerAdapter
    """
    if not issubclass(adapter_class, BaseBrokerAdapter):
        raise TypeError("Adapter must inherit from BaseBrokerAdapter")
    _broker_adapters[broker_code.upper()] = adapter_class


def get_broker_adapter(broker: Broker, paper_trading: bool = True) -> Optional[BaseBrokerAdapter]:
    """Get a broker adapter instance for the given broker.

    Args:
        broker: Broker model instance
        paper_trading: Whether to use paper trading credentials (True) or real money (False)

    Returns:
        BrokerAdapter instance or None if adapter not found
    """
    broker_code = broker.code.upper()

    if broker_code not in _broker_adapters:
        return None

    adapter_class = _broker_adapters[broker_code]

    try:
        return adapter_class(broker, paper_trading=paper_trading)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(
            "Error creating broker adapter for %s: %s", broker_code, e,
        )
        return None


# Register broker adapters
register_broker_adapter('ALPACA', AlpacaBrokerAdapter)
