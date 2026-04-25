import hashlib
import json
from datetime import datetime, timezone as dt_timezone
from decimal import Decimal
from typing import Any, Dict, Optional

from django.utils import timezone


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if timezone.is_naive(dt):
        dt = timezone.make_aware(dt, timezone=dt_timezone.utc)
    return dt.astimezone(dt_timezone.utc).isoformat().replace('+00:00', 'Z')


def _normalize(obj: Any) -> Any:
    """
    Canonicalize a JSON-serializable structure so semantically-equal payloads hash identically.
    """
    if obj is None:
        return None
    if isinstance(obj, Decimal):
        # Keep full decimal representation; calling code may already quantize.
        return str(obj)
    if isinstance(obj, float):
        # Avoid float noise; keep stable precision
        return round(obj, 12)
    if isinstance(obj, (int, str, bool)):
        return obj
    if isinstance(obj, datetime):
        return _iso(obj)
    if isinstance(obj, dict):
        return {str(k): _normalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    # Fallback: stringify unknown types
    return str(obj)


def build_symbol_run_parameter_payload(
    *,
    strategy_id: int,
    broker_id: Optional[int],
    start_date: datetime,
    end_date: datetime,
    split_ratio: float,
    initial_capital: Any,
    bet_size_percentage: float,
    strategy_parameters: Dict[str, Any],
    position_modes: Any,
    hedge_enabled: bool,
    run_strategy_only_baseline: bool,
    hedge_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the canonical payload used for signature generation.
    Excludes symbol and run name by design (so many symbols share one global identifier).
    """
    # Sort position modes deterministically (order should not matter)
    modes = list(position_modes or [])
    modes = [m for m in modes if m in ('long', 'short')]
    modes = sorted(set(modes))

    payload = {
        'strategy_id': int(strategy_id),
        'broker_id': int(broker_id) if broker_id is not None else None,
        'start_date': _iso(start_date),
        'end_date': _iso(end_date),
        'split_ratio': float(split_ratio),
        'initial_capital': str(initial_capital),
        'bet_size_percentage': float(bet_size_percentage),
        'strategy_parameters': strategy_parameters or {},
        'position_modes': modes,
        'hedge_enabled': bool(hedge_enabled),
        'run_strategy_only_baseline': bool(run_strategy_only_baseline),
        'hedge_config': hedge_config or {},
    }
    return _normalize(payload)


def signature_for_payload(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

