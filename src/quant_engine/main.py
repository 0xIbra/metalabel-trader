import numpy as np
from collections import deque
from src.shared.datatypes import TickData
from src.quant_engine.indicators import (
    calculate_log_returns,
    calculate_volatility,
    calculate_z_score,
    calculate_rsi
)
import logging

logger = logging.getLogger("QuantEngine")

class QuantEngine:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.prices = deque(maxlen=window_size * 2) # Keep a bit more history
        self.timestamps = deque(maxlen=window_size * 2)

    def on_tick(self, tick: TickData):
        """
        Process a new tick and return calculated features.
        """
        # Use bid price for calculation for now (or mid)
        price = tick.bid
        self.prices.append(price)
        self.timestamps.append(tick.timestamp)

        if len(self.prices) < self.window_size:
            return None

        # Convert to numpy array for Numba functions
        price_arr = np.array(self.prices)

        # Calculate features
        # 1. Log Returns
        log_returns = calculate_log_returns(price_arr)

        # 2. Volatility (StdDev of Log Returns)
        volatility = calculate_volatility(log_returns, self.window_size)

        # 3. Z-Score
        z_score = calculate_z_score(price, price_arr[-self.window_size:])

        # 4. RSI
        rsi = calculate_rsi(price_arr, window=14)

        features = {
            "timestamp": tick.timestamp,
            "price": price,
            "volatility": volatility,
            "z_score": z_score,
            "rsi": rsi
        }

        return features
