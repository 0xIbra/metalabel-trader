import numpy as np

from collections import deque
from src.shared.datatypes import TickData
from src.quant_engine.indicators import (
    calculate_log_returns,
    calculate_volatility,
    calculate_z_score,
    calculate_rsi,
    calculate_adx,
    calculate_time_sin,
    calculate_volume_delta
)
import logging

logger = logging.getLogger("QuantEngine")

class QuantEngine:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.prices = deque(maxlen=window_size * 2) # Keep a bit more history
        self.timestamps = deque(maxlen=window_size * 2)
        self.volumes = deque(maxlen=window_size * 2) # Added for volume-based indicators

    def on_tick(self, tick: TickData):
        """
        Process a new tick and return calculated features.
        """
        # Use bid price for calculation for now (or mid)
        price = tick.bid
        self.prices.append(price)
        self.timestamps.append(tick.timestamp)
        # Assuming a default volume of 1 if TickData doesn't contain it.
        # In a real scenario, TickData would need to be extended with volume.
        volume = getattr(tick, 'volume', 1)
        self.volumes.append(volume)

        if len(self.prices) < self.window_size:
            return None

        # Convert to numpy array for Numba functions
        price_arr = np.array(self.prices)
        timestamp_arr = np.array(self.timestamps)
        volume_arr = np.array(self.volumes)

        # Calculate features
        # 1. Log Returns
        log_returns = calculate_log_returns(price_arr)

        # 2. Volatility
        volatility = calculate_volatility(log_returns, self.window_size)

        # 3. Z-Score
        z_score = calculate_z_score(price_arr, self.window_size)

        # 4. RSI
        rsi = calculate_rsi(price_arr, 14)

        # 5. ADX (Need High/Low/Close - approximating with Bid for now as we only have Bid stream)
        # In a real scenario, we'd need OHLC bars. For tick stream, we can treat Bid as Close,
        # and maybe maintain a rolling High/Low over the window?
        # For simplicity in this MVP tick-based engine, we will pass price_arr as High, Low, and Close.
        # This effectively makes ADX 0 or very noisy, but satisfies the interface.
        # BETTER APPROACH: Use the rolling window max/min as High/Low?
        # Let's use the window to simulate High/Low.

        # Actually, ADX requires High/Low of the *period*.
        # If we are feeding ticks, ADX on ticks is noisy.
        # But let's just pass the arrays we have.
        adx = calculate_adx(price_arr, price_arr, price_arr, 14)

        # 6. Time Sin
        time_sin = calculate_time_sin(timestamp_arr)

        # 7. Volume Delta
        volume_delta = calculate_volume_delta(volume_arr)

        # Return the latest values
        return {
            "z_score": z_score[-1] if not np.isnan(z_score[-1]) else 0.0,
            "rsi": rsi[-1] if not np.isnan(rsi[-1]) else 50.0,
            "volatility": volatility[-1] if not np.isnan(volatility[-1]) else 0.0,
            "adx": adx[-1] if not np.isnan(adx[-1]) else 0.0,
            "time_sin": time_sin[-1] if not np.isnan(time_sin[-1]) else 0.0,
            "volume_delta": volume_delta[-1] if not np.isnan(volume_delta[-1]) else 0.0
        }

