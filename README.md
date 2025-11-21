# Metalabel Trader

Profitable algorithmic trading system with AI-driven signal generation for EURUSD M1 data.

## System Overview

- **Win Rate**: 55% with 70% confidence filter
- **Returns**: +0.77% over 90 days (3.08% annualized)
- **Architecture**: Microservices-lite with event-driven data flow
- **ML Model**: XGBoost 3-class classifier with 16 features

## Components

### Feed Handler
- Real-time tick data from EODHD WebSocket
- Persists to InfluxDB

### Quant Engine
- **Performance**: 0.011ms latency (4,545x faster than requirement)
- **Features**: 16 Numba-optimized indicators
- Technical, momentum, and lag features

### Oracle (AI)
- XGBoost 3-class classification (SELL/NO_ACTION/BUY)
- Triple Barrier labeling (TP=1 pip, SL=1 pip, TO=20 bars)
- 70% confidence threshold filter
- 0.332ms inference time

### Risk Manager
- 2% risk per trade
- 30% max margin utilization
- Dynamic position sizing

## Quick Start

### 1. Setup Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your EODHD API key and InfluxDB credentials
# EODHD_API_KEY=your_key_here
```

### 2. Start Services
```bash
# Start InfluxDB
docker-compose up -d

# Verify InfluxDB
python verify_influx.py
```

### 3. Train Model
```bash
# Fetch 90 days of data
python fetch_extended_data.py

# Train XGBoost model with triple barrier labeling
python -m src.training.train_model

# Evaluate model
python evaluate_model.py
```

### 4. Backtest
```bash
# Run backtest with $1K account, 1:30 leverage
python backtest_trades.py

# Walk-forward validation
python walk_forward_validation.py
python rolling_window_validation.py
```

### 5. Run Tests
```bash
pytest tests/
```

## Performance Benchmarks

### Latency (per operation)
- Quant Engine: 0.011ms
- Oracle Inference: 0.332ms
- End-to-End: 0.339ms

### Backtest Results (90 days, $1K account, 1:30 leverage)
- Trades: 100
- Win Rate: 55%
- P&L: +$7.74
- Return: +0.77%
- Max Drawdown: $12.24 (1.2%)
- Sharpe Ratio: 1.38

### Walk-Forward Validation
- **Expanding Window**: F1=0.39, CV=0.04 (highly consistent)
- **Rolling Window**: F1=0.40, CV=0.02 (very consistent)

## Model Details

### Features (16 total)
**Technical**: z_score, rsi, volatility, adx, time_sin, volume_delta
**Momentum**: roc_5, roc_10, roc_20, macd, velocity
**Lag**: close_lag1-3, returns_lag1-2

### Labeling Strategy
Triple Barrier Method:
- Take-Profit: 1 pip
- Stop-Loss: 1 pip
- Timeout: 20 bars (20 minutes)

### Class Distribution
- SELL/Avoid: 48%
- BUY: 48%
- NO_ACTION: 4%

## Architecture

```
┌─────────────────┐
│  Feed Handler   │ ──▶ EODHD WebSocket
│  (Real-time)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Quant Engine   │ ──▶ 16 Numba Features
│  (0.011ms)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Oracle (AI)    │ ──▶ XGBoost Inference
│  (0.332ms)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Risk Manager +  │ ──▶ Position Sizing
│  Executioner    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   InfluxDB      │ ──▶ Persistence
│  (Data Layer)   │
└─────────────────┘
```

## Development

### Project Structure
```
metalabel-trader/
├── src/
│   ├── feed_handler/      # Real-time data ingestion
│   ├── quant_engine/      # Feature engineering
│   ├── oracle/            # AI inference
│   ├── executioner/       # Order execution
│   ├── training/          # Model training pipeline
│   └── shared/            # Common datatypes
├── tests/                 # Unit tests
├── data/raw/              # Historical data
├── docker-compose.yml     # InfluxDB setup
└── backtest_trades.py     # Backtesting script
```

### Key Files
- `src/training/train_model.py` - Model training with triple barrier
- `backtest_trades.py` - Trading simulation
- `benchmark_performance.py` - Latency benchmarks
- `walk_forward_validation.py` - Time-series validation

## Risk Disclaimer

This is an educational/research project. Use at your own risk. Past performance does not guarantee future results. Always test thoroughly in a demo environment before live trading.

## License

MIT