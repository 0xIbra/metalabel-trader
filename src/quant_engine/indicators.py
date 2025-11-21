import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def calculate_log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Calculate log returns: ln(P_t / P_{t-1})
    """
    if len(prices) < 2:
        return np.zeros(len(prices))

    res = np.zeros(len(prices))
    # res[0] is 0 because no previous price
    for i in range(1, len(prices)):
        if prices[i-1] > 0:
            res[i] = np.log(prices[i] / prices[i-1])
    return res

@jit(nopython=True, cache=True)
def calculate_volatility(log_returns: np.ndarray, window: int) -> float:
    """
    Calculate standard deviation of log returns over the last 'window' periods.
    Returns the volatility of the *last* window.
    """
    if len(log_returns) < window:
        return 0.0

    # Take the last 'window' elements
    slice_arr = log_returns[-window:]
    return np.std(slice_arr)

@jit(nopython=True, cache=True)
def calculate_z_score(price: float, window_prices: np.ndarray) -> float:
    """
    Calculate Z-Score: (Price - MA) / StdDev
    """
    if len(window_prices) < 2:
        return 0.0

    mean = np.mean(window_prices)
    std = np.std(window_prices)

    if std == 0:
        return 0.0

    return (price - mean) / std

@jit(nopython=True, cache=True)
def calculate_ema(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average.
    """
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()

    # Simple convolution for demonstration, or iterative for true EMA
    # For true EMA with Numba:
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    return ema

@jit(nopython=True, cache=True)
def calculate_rsi(prices: np.ndarray, window: int = 14) -> float:
    """
    Calculate RSI for the last point.
    """
    if len(prices) <= window:
        return 50.0

    deltas = np.diff(prices)
    seed = deltas[:window]
    up = seed[seed >= 0].sum() / window
    down = -seed[seed < 0].sum() / window

    if down != 0:
        rs = up / down
    else:
        rs = 0.0 # Avoid div by zero, though technically RSI is 100 if down is 0
        if up > 0: return 100.0

    rsi = 100. - (100. / (1. + rs))

    # Calculate for the rest
    for i in range(window, len(deltas)):
        delta = deltas[i]
        if delta > 0:
            up_val = delta
            down_val = 0.
        else:
            up_val = 0.
            down_val = -delta

        up = (up * (window - 1) + up_val) / window
        down = (down * (window - 1) + down_val) / window

        if down == 0:
            rsi = 100.0
        else:
            rs = up / down
            rsi = 100. - (100. / (1. + rs))

    return rsi
