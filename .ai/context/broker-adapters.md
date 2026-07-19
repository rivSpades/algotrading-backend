# Broker adapters (blackbox)

## Principle

Each broker implementation is an **API-specific blackbox**. Callers always use the
same exterior surface — they must not branch on Alpaca vs a future IB adapter.

Complexity (auth headers, REST paths, paper vs live base URLs, fill parsing) stays
inside `live_trading/adapters/{broker}.py`.

## Exterior (uniform)

| Step | API |
|------|-----|
| Resolve adapter | `get_broker_adapter(broker, paper_trading=True)` from `live_trading.adapters.factory` |
| Place order | `adapter.place_order(symbol, side, quantity, order_type=..., ...)` → `OrderResult` |
| Cancel / status | `cancel_order`, `get_order_status` |
| Positions | `get_position`, `get_all_positions`, `get_position_resolved` |
| Account | `get_account_balance`, `get_account_equity` |
| Symbol / market | `is_symbol_tradable`, `get_symbol_capabilities`, `get_current_price`, `get_market_data` |

Contract: `BaseBrokerAdapter` in `live_trading/adapters/base.py`.

Orchestration (LiveTrade lifecycle, sizing, audit events) lives in
`live_trading/services/order_service.py` — it types `BaseBrokerAdapter` and calls
`place_order`; it must not embed broker-vendor HTTP.

## Interior (blackbox)

- `live_trading/adapters/alpaca.py` — Alpaca REST (paper/real credentials from `Broker` model)
- Future brokers: one module per vendor, same ABC methods

Blackboxes must **not** import Celery tasks, views, or order-service orchestration.

## Adding a new broker

1. Subclass `BaseBrokerAdapter` in `live_trading/adapters/{code}.py`
2. Implement all abstract methods; map vendor payloads to `OrderResult` / `PositionInfo`
3. `register_broker_adapter('CODE', YourAdapter)` in `factory.py` (or import-time register)
4. Callers keep using `get_broker_adapter` — no changes to engines/views for vendor APIs

## Sensible exceptions

- **Backtest** simulates fills without a real broker adapter
- Paper vs real credentials are selected via `paper_trading=` on the factory, not via
  different exterior method names

## Don't

- Import `AlpacaBrokerAdapter` (or vendor SDKs/HTTP) from views, tasks, or engines
- Duplicate Alpaca (or other) request logic in `order_service` or live engines
- Expose broker-specific order shapes to the rest of the codebase — normalize in the adapter

## Related paths

- `live_trading/adapters/base.py` — ABC + result types
- `live_trading/adapters/factory.py` — registry + `get_broker_adapter`
- `live_trading/adapters/alpaca.py` — reference implementation
- `live_trading/services/order_service.py` — order/trade orchestration
- `live_trading/tests/mock_broker_adapter.py` — test double implementing the ABC
