"""Normalize which backtest execution modes (long / short) to run."""

ALLOWED = frozenset({'long', 'short'})
DEFAULT_MODES = ['long', 'short']


def default_position_modes_list():
    """JSONField default (new list each row in Django — use as default= callable)."""
    return ['long', 'short']


def normalize_position_modes(value):
    """
    Return ordered unique subset of long/short (long first, then short).
    None, empty, or invalid entries fall back to DEFAULT_MODES.
    """
    if value is None:
        return list(DEFAULT_MODES)
    if not isinstance(value, list):
        return list(DEFAULT_MODES)
    seen = set()
    out = []
    for m in value:
        if m in ALLOWED and m not in seen:
            seen.add(m)
            out.append(m)
    if not out:
        return list(DEFAULT_MODES)
    order = {'long': 0, 'short': 1}
    out.sort(key=lambda x: order[x])
    return out
