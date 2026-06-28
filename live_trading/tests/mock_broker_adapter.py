"""Minimal broker adapter for unit tests (no network)."""

from __future__ import annotations

from decimal import Decimal
from typing import Dict, List, Optional

from django.utils import timezone

from live_trading.adapters.base import BaseBrokerAdapter, OrderResult, PositionInfo


class PaperTestBrokerAdapter(BaseBrokerAdapter):
    """Returns synthetic fills so order_service can open/close `LiveTrade` rows."""

    def __init__(
        self,
        broker,
        *,
        quote: Decimal = Decimal('100'),
        fill_price_sequence: Optional[list] = None,
    ):
        super().__init__(broker, paper_trading=True)
        self._quote = quote
        self._fill_price_sequence = list(fill_price_sequence) if fill_price_sequence else []
        self._seq = 0
        #: Long qty per ticker for position checks (manual close).
        self._long_qty: dict[str, Decimal] = {}

    def _next_fill_price(self) -> Decimal:
        if self._fill_price_sequence:
            return Decimal(str(self._fill_price_sequence.pop(0)))
        return Decimal(self._quote)

    def _next_order(self, symbol: str, side: str, quantity: Decimal, price: Decimal) -> OrderResult:
        self._seq += 1
        oid = str(self._seq)
        now = timezone.now()
        qty = Decimal(quantity)
        px = Decimal(price)

        ticker = symbol.strip().upper()
        if side.lower() == 'buy':
            self._long_qty[ticker] = self._long_qty.get(ticker, Decimal('0')) + qty
        elif side.lower() == 'sell':
            prev = self._long_qty.get(ticker, Decimal('0'))
            self._long_qty[ticker] = prev - qty
            if self._long_qty[ticker] <= 0:
                self._long_qty.pop(ticker, None)

        return OrderResult(
            order_id=oid,
            symbol=ticker,
            side=side,
            quantity=qty,
            filled_quantity=qty,
            price=px,
            status='filled',
            timestamp=now,
            broker_order_id=f'b{oid}',
        )

    def connect(self) -> bool:
        return True

    def disconnect(self):
        return None

    def verify_credentials(self) -> bool:
        return True

    def cancel_order(self, order_id: str) -> bool:
        return False

    def get_order_status(self, order_id: str) -> OrderResult:
        now = timezone.now()
        q = Decimal('1')
        return OrderResult(
            order_id=str(order_id),
            symbol='X',
            side='buy',
            quantity=q,
            filled_quantity=q,
            price=self._quote,
            status='filled',
            timestamp=now,
            broker_order_id=str(order_id),
        )

    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        raw = (symbol or '').strip().upper()
        qty = self._long_qty.get(raw)
        if qty is None or qty <= 0:
            return None
        now = timezone.now()
        px = self._quote
        return PositionInfo(
            symbol=raw,
            quantity=qty,
            average_price=px,
            current_price=px,
            unrealized_pnl=Decimal('0'),
            position_type='long',
            timestamp=now,
        )

    def get_all_positions(self) -> List[PositionInfo]:
        out: List[PositionInfo] = []
        now = timezone.now()
        px = self._quote
        for raw, qty in self._long_qty.items():
            if qty and qty > 0:
                out.append(
                    PositionInfo(
                        symbol=raw,
                        quantity=qty,
                        average_price=px,
                        current_price=px,
                        unrealized_pnl=Decimal('0'),
                        position_type='long',
                        timestamp=now,
                    )
                )
        return out

    def get_account_balance(self) -> Decimal:
        return Decimal('100000')

    def get_account_equity(self) -> Decimal:
        return Decimal('100000')

    def is_symbol_tradable(self, symbol: str) -> bool:
        return True

    def get_symbol_capabilities(self, symbol: str) -> Dict:
        return {
            'long_supported': True,
            'short_supported': False,
            'fractionable': True,
            'min_order_size': Decimal('0'),
            'max_order_size': Decimal('999999999'),
        }

    def get_current_price(self, symbol: str) -> Optional[Decimal]:
        return self._quote

    def get_market_data(
        self,
        symbol: str,
        start_date=None,
        end_date=None,
        timeframe: str = '1min',
    ) -> List[Dict]:
        return []

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        order_type: str = 'market',
        limit_price=None,
        stop_price=None,
    ) -> OrderResult:
        px = self._next_fill_price()
        return self._next_order(symbol, side, Decimal(quantity), px)
