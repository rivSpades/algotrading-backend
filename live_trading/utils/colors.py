"""Color / tier classification helpers shared by frontend and backend.

The color palette must match the scatter chart used in
`algo-trading-frontend/src/pages/StrategyDetail.jsx` (`bucketColor`).

Buckets:
    Sharpe:
      S3: > 2.0
      S2: 1.0 .. 2.0
      S1: 0   .. 1.0
      S0: < 0
    Max-drawdown (absolute %):
      D1: <= 20
      D2: <= 40
      D3: <= 60
      D4: > 60

Color grid (rows S3..S0, cols D1..D4):
    S3:  green  green  yellow  orange
    S2:  green  yellow orange  red
    S1:  yellow orange red     red
    S0:  red    red    red     black
"""

from __future__ import annotations

from typing import Optional

GREEN = 'green'
YELLOW = 'yellow'
ORANGE = 'orange'
RED = 'red'
BLACK = 'black'
GRAY = 'gray'

TIER_GT_50 = 'gt50'
TIER_GT_20 = 'gt20'
TIER_GT_10 = 'gt10'
TIER_GT_0 = 'gt0'
TIER_NONE = 'none'


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN check
        return None
    return result


def bucket_color(sharpe, max_drawdown) -> str:
    """Return the discrete color for a (sharpe, max_drawdown) pair.

    `max_drawdown` is interpreted as an absolute percentage; signs are ignored.
    Returns one of `GREEN`, `YELLOW`, `ORANGE`, `RED`, `BLACK`, or `GRAY`.
    """

    sh = _safe_float(sharpe)
    dd_raw = _safe_float(max_drawdown)
    if sh is None or dd_raw is None:
        return GRAY

    dd = abs(dd_raw)

    if sh > 2:
        if dd <= 20:
            return GREEN
        if dd <= 40:
            return GREEN
        if dd <= 60:
            return YELLOW
        return ORANGE
    if sh >= 1:
        if dd <= 20:
            return GREEN
        if dd <= 40:
            return YELLOW
        if dd <= 60:
            return ORANGE
        return RED
    if sh >= 0:
        if dd <= 20:
            return YELLOW
        if dd <= 40:
            return ORANGE
        return RED
    if dd > 60:
        return BLACK
    return RED


def is_green(sharpe, max_drawdown) -> bool:
    """Strict green-cell rule shared with the deployment selector.

    GREEN = `Sharpe > 2 & |DD| <= 40` OR `Sharpe in [1, 2] & |DD| <= 20`.
    """

    return bucket_color(sharpe, max_drawdown) == GREEN


def trade_count_tier(total_trades) -> str:
    """Classify a closed-trade count into ordering tiers.

    Returns one of `gt50`, `gt20`, `gt10`, `gt0`, or `none`.
    """

    n = _safe_float(total_trades)
    if n is None:
        return TIER_NONE
    if n > 50:
        return TIER_GT_50
    if n > 20:
        return TIER_GT_20
    if n > 10:
        return TIER_GT_10
    if n > 0:
        return TIER_GT_0
    return TIER_NONE


# Order tiers from best to worst for sorting purposes.
TIER_ORDER = [TIER_GT_50, TIER_GT_20, TIER_GT_10, TIER_GT_0, TIER_NONE]
TIER_RANK = {tier: rank for rank, tier in enumerate(TIER_ORDER)}
