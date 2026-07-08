"""Symbol priority order permutations for order-variance Monte Carlo."""

from __future__ import annotations

import itertools
import math
import random
from typing import List, Tuple

# Enumerate exactly when n! is at most this (7! = 5040).
MAX_EXACT_ENUMERATION = 5040


def permutation_count(symbol_count: int) -> int:
    if symbol_count < 0:
        return 0
    return math.factorial(symbol_count)


def max_variant_runs(symbol_count: int) -> int:
    """Distinct variant orders excluding the reference run (Run 0)."""
    total = permutation_count(symbol_count)
    return max(0, total - 1)


def cap_variant_request(symbol_count: int, requested: int) -> Tuple[int, int]:
    """
    Cap requested variant count to unique orders available.

    Returns (capped_request, max_unique_variants).
    """
    max_unique = max_variant_runs(symbol_count)
    req = max(0, int(requested or 0))
    return min(req, max_unique), max_unique


def variant_symbol_orders(
    tickers: List[str],
    reference_order: List[str],
    requested: int,
) -> Tuple[List[List[str]], int]:
    """
    Return up to `requested` unique symbol orders, excluding reference_order.

    When n! is small, returns a deterministic exhaustive list (lex order).
    Otherwise samples without replacement up to the unique cap.
    """
    ref = list(reference_order)
    n = len(tickers)
    max_unique = max_variant_runs(n)
    take = min(max(0, int(requested or 0)), max_unique)
    if take == 0:
        return [], max_unique

    total = permutation_count(n)
    if total <= MAX_EXACT_ENUMERATION:
        others = [list(p) for p in itertools.permutations(tickers) if list(p) != ref]
        return others[:take], max_unique

    seen = {tuple(ref)}
    orders: List[List[str]] = []
    attempts = 0
    max_attempts = max(take * 50, 1000)
    while len(orders) < take and attempts < max_attempts:
        attempts += 1
        order = list(tickers)
        random.shuffle(order)
        key = tuple(order)
        if key in seen:
            continue
        seen.add(key)
        orders.append(order)
    return orders, max_unique
