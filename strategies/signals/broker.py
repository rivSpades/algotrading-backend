"""Broker capability helpers shared by backtest and live runtimes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from market_data.models import Symbol


def resolve_broker_side_capabilities(
    symbol: 'Symbol',
    broker,
) -> Tuple[bool, bool]:
    """
    Return (long_allowed, short_allowed) for a symbol on a broker.

    Policy: no SymbolBrokerAssociation → both sides disabled (explicit linkage required).
    """
    if broker is None or symbol is None:
        return True, True

    from live_trading.models import SymbolBrokerAssociation

    try:
        assoc = SymbolBrokerAssociation.objects.get(symbol=symbol, broker=broker)
    except SymbolBrokerAssociation.DoesNotExist:
        return False, False

    return bool(assoc.long_active), bool(assoc.short_active)
