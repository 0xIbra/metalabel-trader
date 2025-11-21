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
def calculate_volatility(log_returns: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate rolling standard deviation of log returns.
    """
    n = len(log_returns)
    vol = np.zeros(n)
    if n < window:
        return vol

    for i in range(window, n):
        slice_arr = log_returns[i-window+1:i+1]
        vol[i] = np.std(slice_arr)
    return vol


@jit(nopython=True, cache=True)
def calculate_z_score(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate rolling Z-Score: (Price - MA) / StdDev
    """
    n = len(prices)
    z_scores = np.zeros(n)
    if n < window:
        return z_scores

    for i in range(window, n):
        slice_arr = prices[i-window+1:i+1]
        mean = np.mean(slice_arr)
        std = np.std(slice_arr)

        if std != 0:
            z_scores[i] = (prices[i] - mean) / std

    return z_scores


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

@jit(nopython=True)
def calculate_rsi(prices, period=14):
    """
    Calculate RSI using Numba.
    """
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period

    if down == 0:
        rs = 100.0  # If no downward movement, RSI is 100
    else:
        rs = up / down

    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)

    for i in range(period, len(prices)):
        delta = deltas[i - 1]  # diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period

        if down == 0:
            rsi[i] = 100.0
        else:
            rs = up / down
            rsi[i] = 100. - 100. / (1. + rs)

    return rsi

@jit(nopython=True)
def calculate_adx(high, low, close, period=14):
    """
    Calculate ADX using Numba.
    """
    n = len(close)
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    # Calculate TR and DM
    for i in range(1, n):
        h_l = high[i] - low[i]
        h_pc = abs(high[i] - close[i-1])
        l_pc = abs(low[i] - close[i-1])
        tr[i] = max(h_l, max(h_pc, l_pc))

        delta_h = high[i] - high[i-1]
        delta_l = low[i-1] - low[i]

        if (delta_h > delta_l) and (delta_h > 0):
            plus_dm[i] = delta_h
        else:
            plus_dm[i] = 0.0

        if (delta_l > delta_h) and (delta_l > 0):
            minus_dm[i] = delta_l
        else:
            minus_dm[i] = 0.0

    # Smooth TR, +DM, -DM (Wilder's Smoothing)
    # First value is simple sum
    tr_smooth = np.zeros(n)
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)

    tr_smooth[period] = np.sum(tr[1:period+1])
    plus_dm_smooth[period] = np.sum(plus_dm[1:period+1])
    minus_dm_smooth[period] = np.sum(minus_dm[1:period+1])

    for i in range(period + 1, n):
        tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / period) + tr[i]
        plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period) + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period) + minus_dm[i]

    # Calculate DI and DX
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)

    for i in range(period, n):
        if tr_smooth[i] != 0:
            plus_di[i] = 100 * (plus_dm_smooth[i] / tr_smooth[i])
            minus_di[i] = 100 * (minus_dm_smooth[i] / tr_smooth[i])

        sum_di = plus_di[i] + minus_di[i]
        if sum_di != 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / sum_di

    # Calculate ADX (Smooth DX)
    adx = np.zeros(n)
    adx[2*period - 1] = np.mean(dx[period:2*period])

    for i in range(2*period, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period

    return adx

@jit(nopython=True)
def calculate_time_sin(timestamps):
    """
    Calculate Sine of time of day.
    """
    n = len(timestamps)
    result = np.zeros(n)
    seconds_in_day = 24 * 60 * 60

    for i in range(n):
        # Assuming timestamp is unix epoch
        # We want time within the day.
        # timestamp % seconds_in_day gives seconds since 00:00 UTC (roughly)
        # Ideally we use struct_time but numba doesn't like it.
        # Simple modulo is fast and works for periodicity.
        time_of_day = timestamps[i] % seconds_in_day
        result[i] = np.sin(2 * np.pi * time_of_day / seconds_in_day)

    return result

@jit(nopython=True)
def calculate_volume_delta(volumes):
    """
    Calculate Log Change in Volume.
    """
    n = len(volumes)
    result = np.zeros(n)

    for i in range(1, n):
        if volumes[i-1] > 0 and volumes[i] > 0:
            result[i] = np.log(volumes[i] / volumes[i-1])
        else:
            result[i] = 0.0

    return result

@jit(nopython=True)
def calculate_bollinger_bands(prices, window=20, num_std=2):
    """
    Calculate Bollinger Bands (Upper, Middle, Lower) and Bandwidth.
    Returns: (upper, middle, lower, bandwidth)
    """
    n = len(prices)
    upper = np.zeros(n)
    middle = np.zeros(n)
    lower = np.zeros(n)
    bandwidth = np.zeros(n)

    if n < window:
        return upper, middle, lower, bandwidth

    for i in range(window, n):
        slice_arr = prices[i-window+1:i+1]
        mean = np.mean(slice_arr)
        std = np.std(slice_arr)

        middle[i] = mean
        upper[i] = mean + (num_std * std)
        lower[i] = mean - (num_std * std)

        if middle[i] != 0:
            bandwidth[i] = (upper[i] - lower[i]) / middle[i]

    return upper, middle, lower, bandwidth

@jit(nopython=True)
def calculate_atr(high, low, close, window=14):
    """
    Calculate Average True Range (ATR).
    """
    n = len(close)
    tr = np.zeros(n)
    atr = np.zeros(n)

    # Calculate TR
    for i in range(1, n):
        h_l = high[i] - low[i]
        h_pc = abs(high[i] - close[i-1])
        l_pc = abs(low[i] - close[i-1])
        tr[i] = max(h_l, max(h_pc, l_pc))

    # First ATR is simple average of TR
    if n > window:
        atr[window] = np.mean(tr[1:window+1])

        # Subsequent ATR: (Prior ATR * (n-1) + Current TR) / n
        for i in range(window + 1, n):
            atr[i] = (atr[i-1] * (window - 1) + tr[i]) / window

    return atr

@jit(nopython=True)
def calculate_pivot_points(high, low, close):
    """
    Calculate Standard Pivot Points (P, R1, S1).
    Note: This is a rolling calculation based on previous bar,
    typically used for Daily/Weekly pivots.
    For H4, we treat the previous H4 bar as the reference for simplicity
    in this rolling context, or we'd need higher timeframe data.
    Here we calculate Pivot based on previous bar: (H+L+C)/3
    """
    n = len(close)
    pivot = np.zeros(n)
    r1 = np.zeros(n)
    s1 = np.zeros(n)

    for i in range(1, n):
        p = (high[i-1] + low[i-1] + close[i-1]) / 3
        pivot[i] = p
        r1[i] = (2 * p) - low[i-1]
        s1[i] = (2 * p) - high[i-1]

    return pivot, r1, s1
