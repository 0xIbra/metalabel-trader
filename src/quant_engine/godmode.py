import numpy as np
import pandas as pd

def detect_liquidity_grab(high, low, close, window=20):
    """
    Detects 'Liquidity Grabs' (Stop Hunts).

    Pattern:
    - Bullish Grab: Price dips below a recent significant Low (Support) but closes ABOVE it.
    - Bearish Grab: Price spikes above a recent significant High (Resistance) but closes BELOW it.

    Args:
        high (np.array): High prices
        low (np.array): Low prices
        close (np.array): Close prices
        window (int): Lookback period for Support/Resistance

    Returns:
        int: 1 (Bullish Grab), -1 (Bearish Grab), 0 (None)
    """
    if len(close) < window + 1:
        return 0

    # Current bar (latest completed bar)
    curr_high = high[-1]
    curr_low = low[-1]
    curr_close = close[-1]

    # Previous bars for context
    prev_highs = high[-(window+1):-1]
    prev_lows = low[-(window+1):-1]

    # Find recent significant levels (Swing High/Low)
    # Simple approach: Min/Max of the window
    support_level = np.min(prev_lows)
    resistance_level = np.max(prev_highs)

    # Check Bullish Grab (Fake-out below support)
    # 1. Low went below support
    # 2. Close is back above support
    if curr_low < support_level and curr_close > support_level:
        return 1

    # Check Bearish Grab (Fake-out above resistance)
    # 1. High went above resistance
    # 2. Close is back below resistance
    if curr_high > resistance_level and curr_close < resistance_level:
        return -1

    return 0

def get_swap_bias(symbol):
    """
    Returns the directional bias based on interest rate differentials (Carry Trade).

    1 = Long Bias (Base currency yields more)
    -1 = Short Bias (Quote currency yields more)
    0 = Neutral / Unknown

    NOTE: These values are approximations based on typical 2024/2025 rate environments.
    Ideally, this should be fetched from a live calendar/central bank API.
    """
    # Map of Currency -> Interest Rate (Approx %)
    # Example Scenario: USD (5.5%), EUR (4.0%), JPY (0.0%), AUD (4.35%), GBP (5.25%)
    rates = {
        'USD': 5.50,
        'EUR': 4.00,
        'GBP': 5.25,
        'JPY': 0.10,
        'AUD': 4.35,
        'NZD': 5.50,
        'CAD': 5.00,
        'CHF': 1.75
    }

    if len(symbol) != 6:
        return 0

    base = symbol[:3]
    quote = symbol[3:]

    if base not in rates or quote not in rates:
        return 0

    base_rate = rates[base]
    quote_rate = rates[quote]

    diff = base_rate - quote_rate

    # Significant differential threshold (e.g., 1.0%)
    if diff > 1.0:
        return 1  # Long Base
    elif diff < -1.0:
        return -1 # Short Base (Long Quote)

    return 0
