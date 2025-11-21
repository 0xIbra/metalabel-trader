# Metalabel Trader - XGBoost Forex Trading Bot

High-performance algorithmic trading system using XGBoost ML models with Multi-Symbol support and Live Paper Trading via MetaApi.

## 🎯 Performance Summary

**Strategy**: 5:1 Risk-Reward (TP=5 pips, SL=1 pip, 50% confidence threshold)

### Backtest Results (90 days, $1,000 account)

| Strategy | Symbols | Trades | Win Rate | P&L | Return | Annualized |
|----------|---------|--------|----------|-----|--------|------------|
| **Optimized Multi-Symbol** | EURUSD + AUDUSD | 43 | 34.9% | **+$39.96** | **+4.00%** | **16.2%** |
| Single Symbol | EURUSD only | 27 | 33.3% | +$40.77 | +4.08% | 16.3% |

**Key Metrics**:
- Profit Factor: 2.63
- Trade Frequency: ~0.5/day (1 every 2 days)
- Max Drawdown: <2%
- Sharpe Ratio: 8.85

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-live.txt  # For live trading
```

### 2. Set Up Environment
Create `.env` file:
```bash
# Data API
EODHD_API_KEY=your_key_here

# Live Trading (Optional)
METAAPI_TOKEN=your_metaapi_token
METAAPI_ACCOUNT_ID=your_account_id
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 3. Fetch Data & Train Models
```bash
# Fetch 90 days of M1 data
python3.10 fetch_multi_symbol_data.py

# Train models for EURUSD and AUDUSD
python3.10 -m src.training.train_model --symbol EURUSD
python3.10 -m src.training.train_model --symbol AUDUSD
```

### 4. Run Backtest
```bash
python3.10 backtest_multi_symbol.py
```

### 5. Live Paper Trading (Optional)
```bash
# Test connections
python3.10 test_telegram.py
python3.10 test_metaapi.py

# Start live bot
python3.10 live_trading_bot.py
```

See [LIVE_TRADING.md](LIVE_TRADING.md) for detailed setup.

---

## 📁 Project Structure

```
metalabel-trader/
├── src/
│   ├── oracle/              # Trained XGBoost models
│   │   ├── model.json       # EURUSD model
│   │   └── model_audusd.json # AUDUSD model
│   ├── quant_engine/        # Technical indicators
│   ├── training/            # Model training
│   │   ├── train_model.py   # Training script
│   │   └── triple_barrier.py # Labeling method
│   └── notifications/       # Telegram alerts
├── data/raw/                # M1 OHLCV data
├── backtest_multi_symbol.py # Multi-symbol backtest
├── live_trading_bot.py      # Live paper trading
└── LIVE_TRADING.md          # Live trading guide
```

---

## 🎓 Strategy Details

### Triple Barrier Labeling
- **Take Profit**: 5 pips (5:1 risk-reward)
- **Stop Loss**: 1 pip
- **Timeout**: 60 bars (1 hour)
- **Classes**: SELL (-1), NO_ACTION (0), BUY (1)

### Features (16 total)
**Technical Indicators**:
- Z-Score (price normalization)
- RSI (momentum)
- Volatility (20-period)
- ADX (trend strength)
- Time encoding (sin/cos)
- Volume delta

**Momentum**:
- ROC (5, 10, 20 periods)
- MACD
- Price velocity

**Lag Features**:
- Close price lags (1-3 bars)
- Returns lags (1-2 bars)

### Model Architecture
- **Algorithm**: XGBoost (Gradient Boosting)
- **Objective**: Multi-class classification (3 classes) or Binary (2 classes)
- **Trees**: 200
- **Learning Rate**: 0.05
- **Max Depth**: 4
- **Regularization**: L2 (gamma=0.1)

---

## 📊 Live Trading Features

- **MetaApi Integration**: Cloud-based MT4/MT5 connection
- **Telegram Alerts**: Real-time notifications for all events
- **Safety Limits**:
  - Max 2 concurrent positions (1 per symbol)
  - Daily loss limit: $20
  - Timeout: 60 minutes
- **Monitoring**: Hourly status updates

---

## 🛠️ Development

### Run Tests
```bash
python3.10 -m pytest tests/
```

### Retrain Models
```bash
# After fetching new data
python3.10 -m src.training.train_model --symbol EURUSD
python3.10 -m src.training.train_model --symbol AUDUSD
```

### Stress Test
```bash
python3.10 stress_test_backtest.py
```

---

## 📈 Expected Live Performance

Based on backtests:
- **Trades/Week**: ~3-4
- **Weekly P&L**: ~$3-4
- **Monthly Return**: ~1.3%
- **Alerts/Day**: 3-5 (trades + status)

---

## ⚠️ Disclaimer

**This system is for educational and paper trading purposes only.**

- Start with demo accounts
- Past performance does not guarantee future results
- Trading involves substantial risk of loss
- Never risk more than you can afford to lose

---

## 📚 Documentation

- [LIVE_TRADING.md](LIVE_TRADING.md) - Live trading setup guide
- [F1_IMPROVEMENT_PLAN.md](F1_IMPROVEMENT_PLAN.md) - Model optimization history
- [STRATEGY_COMPARISON.md](STRATEGY_COMPARISON.md) - Risk-reward analysis

---

## 🔧 Troubleshooting

### No Trades in Backtest
- Check confidence threshold (try lowering to 45%)
- Verify model files exist
- Ensure data has enough bars (>100)

### Live Bot Not Trading
- Bot needs ~100 minutes to warm up
- Check MetaApi connection
- Verify Telegram is receiving status updates
- Check logs: `logs/live_trading.log`

### MetaApi Connection Issues
- Ensure account is deployed
- Check token and account ID in `.env`
- Try restarting account in MetaApi dashboard

---

## 📞 Support

For issues:
1. Check logs in `logs/`
2. Review [LIVE_TRADING.md](LIVE_TRADING.md)
3. Verify environment variables in `.env`
4. Test components individually with `test_*.py` scripts

---

**Version**: 1.0.0
**Last Updated**: 2025-11-21
**License**: MIT