import pandas as pd
import numpy as np
from numba import jit

@jit(nopython=True)
def apply_triple_barrier(prices, take_profit_pips=2, stop_loss_pips=1, timeout_bars=10, pip_value=0.0001):
    """
    Triple Barrier Method for labeling

    For each bar:
    - Check which happens FIRST in next 'timeout_bars':
      1. Price hits take-profit (+take_profit_pips) → Label = 1 (BUY signal)
      2. Price hits stop-loss (-stop_loss_pips) → Label = -1 (SELL signal / avoid)
      3. Timeout reached → Label = 0 (NO_ACTION)

    Args:
        prices: numpy array of close prices
        take_profit_pips: pips for take-profit threshold
        stop_loss_pips: pips for stop-loss threshold
        timeout_bars: maximum bars to wait
        pip_value: value of 1 pip (0.0001 for EURUSD)

    Returns:
        labels: numpy array of labels (1, 0, or -1)
    """
    n = len(prices)
    labels = np.zeros(n, dtype=np.int32)

    take_profit = take_profit_pips * pip_value
    stop_loss = stop_loss_pips * pip_value

    for i in range(n - timeout_bars):
        entry_price = prices[i]
        tp_price = entry_price + take_profit
        sl_price = entry_price - stop_loss

        # Look forward up to timeout
        for j in range(1, timeout_bars + 1):
            if i + j >= n:
                break

            future_price = prices[i + j]

            # Check if take-profit hit
            if future_price >= tp_price:
                labels[i] = 1  # BUY signal (profitable move detected)
                break

            # Check if stop-loss hit
            elif future_price <= sl_price:
                labels[i] = -1  # SELL signal (or avoid)
                break

        # If loop completes without break, timeout → label = 0 (already initialized)

    return labels

def create_triple_barrier_labels(df, take_profit_pips=2, stop_loss_pips=1, timeout_bars=10):
    """
    Apply triple barrier labeling to dataframe

    Returns:
        df with 'target' column added (-1, 0, or 1)
    """
    prices = df['close'].values
    labels = apply_triple_barrier(prices, take_profit_pips, stop_loss_pips, timeout_bars)

    df['target'] = labels

    # Statistics
    label_counts = pd.Series(labels).value_counts()
    print("\nTriple Barrier Label Distribution:")
    print(f"  BUY signals (1):       {label_counts.get(1, 0):6d} ({label_counts.get(1, 0)/len(labels)*100:5.2f}%)")
    print(f"  NO_ACTION (0):         {label_counts.get(0, 0):6d} ({label_counts.get(0, 0)/len(labels)*100:5.2f}%)")
    print(f"  SELL/Avoid (-1):       {label_counts.get(-1, 0):6d} ({label_counts.get(-1, 0)/len(labels)*100:5.2f}%)")

    return df

if __name__ == "__main__":
    # Test with sample data
    print("Testing Triple Barrier Labeling...")

    # Create sample data
    test_prices = np.array([1.0500, 1.0502, 1.0505, 1.0498, 1.0501, 1.0503, 1.0507, 1.0504])
    labels = apply_triple_barrier(test_prices, take_profit_pips=2, stop_loss_pips=1, timeout_bars=5)

    print("\nTest Results:")
    print(f"Prices: {test_prices}")
    print(f"Labels: {labels}")
    print("\nLabel meaning:")
    print("  1  = Take-profit hit (BUY signal)")
    print("  0  = Timeout (NO_ACTION)")
    print(" -1  = Stop-loss hit (SELL/Avoid)")
