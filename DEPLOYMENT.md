# VPS Deployment Guide with Watchdog

## 🛡️ Crash Protection Options

Choose **ONE** of these methods:

---

## Option 1: Systemd (Recommended for VPS) ✅

Systemd auto-restarts the bot if it crashes. Best for production.

### Setup:

```bash
# 1. Copy service files
sudo cp trading-bot.service /etc/systemd/system/
sudo cp watchdog.service /etc/systemd/system/

# 2. Update paths in service files (if needed)
sudo nano /etc/systemd/system/trading-bot.service
# Change User and WorkingDirectory to match your setup

# 3. Reload systemd
sudo systemctl daemon-reload

# 4. Enable auto-start on boot
sudo systemctl enable trading-bot.service
sudo systemctl enable watchdog.service

# 5. Start services
sudo systemctl start trading-bot.service
sudo systemctl start watchdog.service
```

### Management:

```bash
# Check status
sudo systemctl status trading-bot
sudo systemctl status watchdog

# View logs
sudo journalctl -u trading-bot -f
sudo journalctl -u watchdog -f

# Restart
sudo systemctl restart trading-bot

# Stop
sudo systemctl stop trading-bot
sudo systemctl stop watchdog
```

### What You Get:
- ✅ Auto-restart on crash (max 5 times in 10 minutes)
- ✅ Auto-start on VPS reboot
- ✅ Watchdog sends Telegram alerts if bot dies
- ✅ Watchdog can auto-restart bot (max 3 times/hour)
- ✅ Logs saved to files

---

## Option 2: Watchdog Script Only

Simple Python watchdog that monitors and restarts the bot.

### Setup:

```bash
# 1. Install psutil
pip install psutil

# 2. Start bot in background
nohup python live_trading_bot.py > logs/bot.log 2>&1 &

# 3. Start watchdog in background
nohup python watchdog.py > logs/watchdog.log 2>&1 &
```

### Check Status:

```bash
# View watchdog logs
tail -f logs/watchdog.log

# Check if running
ps aux | grep live_trading_bot
ps aux | grep watchdog
```

### What You Get:
- ✅ Telegram alert if bot crashes
- ✅ Auto-restart (max 3 times/hour)
- ✅ Rate-limited alerts (1 per 5 minutes)
- ❌ No auto-start on reboot

---

## Option 3: Screen/Tmux (Manual Monitoring)

Run bot in persistent terminal session.

### Setup:

```bash
# Start screen session
screen -S trading-bot
python live_trading_bot.py

# Detach: Ctrl+A then D
# Reattach: screen -r trading-bot
```

### What You Get:
- ✅ Simple, no dependencies
- ❌ No auto-restart
- ❌ No crash alerts
- ❌ Manual monitoring required

---

## Telegram Alerts Comparison

| Method | Bot Crash | Bot Restart | Heartbeat |
|--------|-----------|-------------|-----------|
| **Systemd + Watchdog** | ✅ Yes | ✅ Yes | ⚠️ Optional |
| **Watchdog Only** | ✅ Yes | ✅ Yes | ❌ No |
| **Screen** | ❌ No | ❌ No | ❌ No |

---

## Watchdog Features

The `watchdog.py` script:

- ✅ Checks bot every 5 minutes
- ✅ Sends Telegram alert if bot crashes
- ✅ Auto-restarts bot (max 3 times/hour)
- ✅ Rate-limited alerts (1 per 5 minutes)
- ✅ Critical alert if restart fails
- ✅ Logs all events

### Telegram Alerts:

**Bot Crashed**:
```
🚨 WATCHDOG ALERT

⚠️ Trading bot is not running!

Bot process not found.
Auto-restart: True

Time: 2025-11-21 05:45:00 UTC
```

**Bot Restarted**:
```
🔄 BOT RESTARTED

✅ Bot restarted successfully (PID: 12345)

Restarts in last hour: 1/3

Time: 2025-11-21 05:46:00 UTC
```

**Critical (Max Restarts)**:
```
🚨 CRITICAL: RESTART LIMIT REACHED

❌ Bot crashed but restart limit reached (3 restarts in last hour).

Manual intervention required!

Time: 2025-11-21 06:00:00 UTC
```

---

## Installation Commands

### On VPS:

```bash
# 1. Clone/copy repo
cd ~
git clone <your-repo> metalabel-trader
cd metalabel-trader

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-live.txt
pip install psutil  # For watchdog

# 4. Configure .env
nano .env
# Add your METAAPI_TOKEN, METAAPI_ACCOUNT_ID, TELEGRAM credentials

# 5. Create logs directory
mkdir -p logs

# 6. Test
python test_telegram.py
python test_metaapi.py

# 7. Choose deployment method above
```

---

## Recommended Setup

For **production VPS deployment**:

1. Use **Systemd + Watchdog** (Option 1)
2. Monitor Telegram for first 48 hours
3. Check logs daily: `sudo journalctl -u trading-bot -f`
4. Verify first trade P&L matches dashboard

---

## Emergency Procedures

### Stop Everything:
```bash
# Systemd
sudo systemctl stop trading-bot watchdog

# Manual
pkill -f live_trading_bot.py
pkill -f watchdog.py
```

### Check if Running:
```bash
ps aux | grep -E "(live_trading_bot|watchdog)"
```

### View All Logs:
```bash
tail -f logs/*.log
```

---

## FAQ

**Q: Will I get alerts for every crash?**
A: Yes, watchdog sends Telegram alert immediately when bot goes down.

**Q: Does watchdog restart the bot automatically?**
A: Yes, max 3 times per hour. After that, you get a critical alert.

**Q: What if watchdog crashes?**
A: If using systemd, watchdog auto-restarts. Otherwise, no monitoring.

**Q: Can I disable auto-restart?**
A: Yes, edit `watchdog.py` and set `AUTO_RESTART = False`

**Q: Does this work on Windows?**
A: Watchdog script works, but systemd is Linux-only. Use Task Scheduler on Windows.

---

**Recommended**: Use **Option 1 (Systemd + Watchdog)** for maximum reliability and peace of mind! 🛡️
