# Live Trading Setup Guide

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements-live.txt
```

### 2. Set Up Environment Variables

Add to your `.env` file:

```bash
# MetaApi Configuration
METAAPI_TOKEN=your_metaapi_token_here
METAAPI_ACCOUNT_ID=your_demo_account_id_here

# Telegram Notifications
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
```

### 3. Get MetaApi Credentials

1. Sign up at [https://metaapi.cloud](https://metaapi.cloud)
2. Create a **Demo Account** (free tier):
   - Platform: MT4 or MT5
   - Broker: IC Markets, Pepperstone, or similar
   - Balance: $1,000 (virtual)
3. Copy your API Token and Account ID

### 4. Create Telegram Bot

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` and follow prompts
3. Copy the Bot Token
4. Send a message to your bot
5. Get your Chat ID from: `https://api.telegram.org/bot<TOKEN>/getUpdates`

### 5. Run the Bot

```bash
python3.10 live_trading_bot.py
```

---

## 📊 Bot Features

- **Symbols**: EURUSD, AUDUSD
- **Strategy**: 5:1 Risk-Reward (TP=5 pips, SL=1 pip)
- **Confidence**: 50% threshold
- **Position Limit**: Max 2 concurrent (1 per symbol)
- **Daily Loss Limit**: $20
- **Hold Timeout**: 60 minutes

---

## 🔔 Telegram Alerts

### Trade Entry
```
🟢 TRADE OPENED

Symbol: EURUSD
Side: BUY
Entry: 1.08450
TP: 1.08500 (+5 pips)
SL: 1.08440 (-1 pip)
Lot: 0.09
Risk: $20
Confidence: 56.2%
```

### Trade Exit
```
✅ TRADE CLOSED - WINNER

Symbol: EURUSD
Exit: 1.08500 (TP Hit)
Pips: +5.0
P&L: +$4.50
Hold Time: 23 minutes

Account Balance: $1,004.50 (+0.45%)
```

### Hourly Status
```
📊 STATUS UPDATE

Balance: $1,015.32
Open Positions: 1
Trades Today: 3
P&L Today: +$15.32
Win Rate Today: 66.7%
```

---

## 🛡️ Safety Features

1. **Daily Loss Limit**: Stops trading if daily loss > $20
2. **Position Limits**: Max 2 concurrent positions
3. **Confidence Threshold**: Only trades signals > 50%
4. **Timeout**: Auto-closes after 60 minutes
5. **Error Notifications**: Immediate Telegram alerts on issues

---

## 🧪 Testing Strategy

### Phase 1: Dry Run (Recommended)
Before live deployment, test the bot logic without placing real orders.

### Phase 2: Paper Trading (Current)
- Run on MetaApi demo account
- Monitor for 7 days
- Compare with backtest results

### Phase 3: Live Trading (Future)
- Start with small account ($100-500)
- Monitor for 30 days
- Scale gradually

---

## 📈 Expected Performance

Based on 90-day backtest:

- **Trades/Day**: ~0.48 (1 trade every 2 days)
- **Win Rate**: ~35%
- **Monthly Return**: ~1.3%
- **Annualized**: ~16%
- **Max Drawdown**: <2%

---

## 🔧 Troubleshooting

### Bot Won't Connect to MetaApi
- Verify `METAAPI_TOKEN` and `METAAPI_ACCOUNT_ID` in `.env`
- Check demo account is deployed and active
- Ensure account has sufficient virtual balance

### No Telegram Notifications
- Verify `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`
- Send a test message to bot first
- Check bot token permissions

### No Trades Being Placed
- Bot needs 100 bars to warm up (~100 minutes)
- Check confidence threshold (50%)
- Verify models exist: `src/oracle/model.json` and `src/oracle/model_audusd.json`
- Check daily loss limit not exceeded

### High Frequency of Trades
- Expected: 0.48 trades/day (1 every 2 days)
- If much higher, check confidence threshold

---

## 🛑 Stopping the Bot

Press `Ctrl+C` or send SIGTERM. The bot will:
1. Send shutdown notification to Telegram
2. Log all open positions
3. Clean up connections

> **Note**: Open positions will remain open after shutdown. Close manually via MetaApi dashboard if needed.

---

## 📝 Logs

All activity is logged to `logs/live_trading.log`:
- Trade entries/exits
- P&L calculations
- Errors and exceptions
- Connection status

---

## ⚠️ Disclaimer

This bot is for **paper trading and educational purposes only**.

- Start with demo accounts
- Never risk more than you can afford to lose
- Past performance does not guarantee future results
- Trading involves substantial risk of loss

---

## 📞 Support

For issues:
1. Check logs in `logs/live_trading.log`
2. Verify environment variables
3. Test Telegram bot separately
4. Ensure internet connection stable
