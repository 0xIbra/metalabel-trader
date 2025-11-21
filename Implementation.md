### 1. THE CORE ARCHITECTURE (The Nervous System)

We are using a **Microservices-Lite** architecture. Do not build a monolithic script that runs a `while True` loop. If one part breaks, the whole thing crashes.

*   **Language:** Python 3.10+ (Reason: Speed improvements + typing).
*   **Design Pattern:** Event-Driven Architecture (Publisher/Subscriber).
*   **Concurrency:** `asyncio` (Asynchronous I/O). Threading is too heavy; we need non-blocking WebSocket handling.

### 2. THE TECH STACK (Dependencies)

**A. The Application Layer**
*   **`MetaTrader5` (Python Package)** or **`ccxt` (Crypto)**: For API connectivity.
*   **`numpy` & `pandas`**: High-performance array manipulation.
*   **`numba`**: JIT Compiler to speed up indicator calculations (keeps Python from being slow).
*   **`xgboost`**: The gradient boosting inference engine.
*   **`ta-lib`**: Technical Analysis Library (C-wrapper, much faster than calculating RSI in pure Python).

**B. The Data Layer**
*   **Database:** `InfluxDB` or `TimescaleDB`.
    *   *Why:* You need to store tick data (bid/ask/time) efficiently. SQL is trash for this. Influx handles high-write loads (millions of ticks) effortlessly.
*   **Serialization:** `Redis` (Optional but recommended).
    *   *Use Case:* Use Redis as a cache to store the "Current Market State" (Z-Score, Volatility) so different modules can read it without recalculating.

---

### 3. COMPONENT SPECIFICATIONS

#### MODULE A: The "Feed Handler" (Ingestion)
*   **Role:** Connects to Broker WebSocket/Stream.
*   **Input:** Raw Tick Data (Symbol, Bid, Ask, Timestamp).
*   **Process:**
    1.  **Debounce:** If ticks arrive faster than 10ms, aggregate them (unless doing HFT).
    2.  **Normalize:** Convert timestamps to UTC (Unix Epoch).
    3.  **Resample:** Build M1/M5 OHLC bars on the fly if the broker doesn't push them fast enough.
*   **Output:** Pushes clean data to the *Signal Engine*.

#### MODULE B: The "Quant Engine" (Feature Engineering)
*   **Role:** Transforms price into math.
*   **Specs:**
    *   **Window Size:** Rolling `window=20` (or variable).
    *   **Calculations (Numba Optimized):**
        *   `Log Returns = ln(P_t / P_{t-1})`
        *   `Vol = StdDev(Log Returns)`
        *   `Z-Score = (Price - MA) / StdDev`
        *   `ADX` (Trend Strength)
*   **Latency Budget:** Must complete all calculations in < 50ms.

#### MODULE C: The "Oracle" (AI Inference)
*   **Role:** Filters the signal.
*   **Input:** Feature Vector `[Z-Score, ADX, RSI, Time_Sin, Volume_Delta]`.
*   **Process:**
    1.  Load pre-trained model (`model.json`).
    2.  `prob = model.predict_proba(feature_vector)`
*   **Logic:**
    ```python
    if signal == "LONG_REVERSION" and prob > 0.85:
        return "EXECUTE_BUY"
    elif signal == "SHORT_REVERSION" and prob > 0.85:
        return "EXECUTE_SELL"
    else:
        return "NO_ACTION"
    ```

#### MODULE D: The "Executioner" (Order Management System - OMS)
*   **Role:** Talks to the money.
*   **Specs:**
    *   **Dynamic Sizing:** `Lots = (Account_Balance * Risk_Percent) / (Stop_Loss_Points * Point_Value)`
    *   **Order Type:** Use `IDC` (Immediate-or-Cancel) or Limit orders with a 3-second TTL (Time To Live).
    *   **Retry Logic:** If order fails (re-quote), retry max 1 time with slightly worse price, then abort.

#### MODULE E: The "Guardian" (The Kill Switch)
*   **Role:** Parallel process monitoring PnL.
*   **Logic:**
    *   Poll Account Equity every 1 second.
    *   **Condition 1 (Daily Stop):** If `Daily_Loss > 3%`, liquidate all, send email/SMS, shutdown process.
    *   **Condition 2 (Fat Finger):** If `Open_Lots > Max_Allowed_Lots`, liquidate excess immediately.

---

### 4. INFRASTRUCTURE SPECS (The Environment)

Do not run this on your gaming PC. Windows Update will reboot your computer while you have 10 lots open and bankrupt you.

**A. Virtual Private Server (VPS)**
*   **OS:** Ubuntu Server 22.04 LTS (Headless - No GUI, lighter and faster).
    *   *Note:* If using MT5, you might be forced to use Windows Server. Strip it down. Remove Cortana, Updates, Firewall crap.
*   **CPU:** High Frequency Compute (3.0 GHz+). Multi-core doesn't matter as much as Single-core speed for Python.
*   **RAM:** 4GB minimum (8GB recommended for InfluxDB).

**B. Location Strategy (Co-Location)**
*   **Ping Test:** You need `< 2ms` ping to your broker.
*   **The Look up:** Find your broker's datacenter.
    *   IC Markets / Pepperstone -> Usually **Equinix NY4 (New York)**.
    *   Forex.com / Oanda -> Usually **Equinix LDN4 (London)** or **TY3 (Tokyo)**.
*   **Protocol:** Rent a VPS in that specific building. (Providers: BeeksFX, CNS, AWS `us-east-1` for NY4).

---

### 5. THE DEPLOYMENT PIPELINE (DevOps)

**The "Git" Workflow:**
1.  **Local Branch:** Develop and Backtest features locally.
2.  **Staging Branch:** Deploy to a Demo account on the VPS. Run for 1 week.
3.  **Master Branch:** Deploy to Live.

**Logging & Alerting:**
*   **Logs:** Structure logs as JSON.
    *   `{"timestamp": 167888, "level": "INFO", "component": "OMS", "message": "Order Filled", "slippage": 0.2}`
*   **Alerts:** Integrate with **Telegram Bot API** or **Discord Webhooks**.
    *   *Why:* You want a push notification on your phone the second a trade is taken or an error occurs.

### 6. IMPLEMENTATION ROADMAP

1.  **Week 1:** Build the Data Ingestion + InfluxDB storage. Verify data integrity.
2.  **Week 2:** Build the Feature Engineering + Backtesting Engine. Train the XGBoost model.
3.  **Week 3:** Build the OMS (Order Management) + Telegram Alerts. Test on Demo.
4.  **Week 4:** Deploy Guardian (Kill Switch) + Go Live with Micro Lots (0.01).

