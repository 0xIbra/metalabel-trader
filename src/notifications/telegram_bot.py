"""
Telegram notification module for trading alerts
"""
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Telegram config
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Check if Telegram is configured
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

if TELEGRAM_ENABLED:
    try:
        import requests
    except ImportError:
        logger.warning("requests not installed, Telegram notifications disabled")
        TELEGRAM_ENABLED = False

def send_telegram(message, parse_mode='HTML'):
    """Send message to Telegram"""
    if not TELEGRAM_ENABLED:
        logger.info(f"[TELEGRAM] {message}")
        return False

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': parse_mode
        }
        response = requests.post(url, data=data, timeout=10)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False

def notify_trade_entry(symbol, side, entry_price, tp_price, sl_price, lot_size, risk_amount, confidence):
    """Notify trade entry"""
    message = f"""
🟢 <b>TRADE OPENED</b>

<b>Symbol:</b> {symbol}
<b>Side:</b> {side}
<b>Entry:</b> {entry_price:.5f}
<b>TP:</b> {tp_price:.5f} (+5 pips)
<b>SL:</b> {sl_price:.5f} (-1 pip)
<b>Lot:</b> {lot_size:.2f}
<b>Risk:</b> ${risk_amount:.2f}
<b>Confidence:</b> {confidence:.1f}%

<i>Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</i>
"""
    return send_telegram(message.strip())

def notify_trade_exit(symbol, exit_price, pnl, pips, reason, hold_time_minutes, balance):
    """Notify trade exit"""
    emoji = "✅" if pnl > 0 else "❌"
    result = "WINNER" if pnl > 0 else "LOSER"

    message = f"""
{emoji} <b>TRADE CLOSED - {result}</b>

<b>Symbol:</b> {symbol}
<b>Exit:</b> {exit_price:.5f} ({reason})
<b>Pips:</b> {pnl/0.9:+.1f}
<b>P&L:</b> ${pnl:+.2f}
<b>Hold Time:</b> {hold_time_minutes} minutes

<b>Account Balance:</b> ${balance:.2f} ({(pnl/1000)*100:+.2f}%)
"""
    return send_telegram(message.strip())

def notify_status(balance, open_positions, trades_today, pnl_today, win_rate_today):
    """Send status update"""
    message = f"""
📊 <b>STATUS UPDATE</b>

<b>Balance:</b> ${balance:.2f}
<b>Open Positions:</b> {open_positions}
<b>Trades Today:</b> {trades_today}
<b>P&L Today:</b> ${pnl_today:+.2f}
<b>Win Rate Today:</b> {win_rate_today:.1f}%

<i>Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</i>
"""
    return send_telegram(message.strip())

def notify_error(error_message, action="Investigating..."):
    """Notify error"""
    message = f"""
⚠️ <b>ERROR DETECTED</b>

<b>Message:</b> {error_message}
<b>Action:</b> {action}

<i>Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</i>
"""
    return send_telegram(message.strip())

def notify_startup(symbols, confidence_threshold, account_balance):
    """Notify bot startup"""
    message = f"""
🚀 <b>BOT STARTED</b>

<b>Symbols:</b> {', '.join(symbols)}
<b>Confidence:</b> {confidence_threshold*100:.0f}%
<b>Account:</b> ${account_balance:.2f}
<b>Strategy:</b> 5:1 RR (TP=5 pips, SL=1 pip)

<i>Paper Trading Mode</i>
"""
    return send_telegram(message.strip())

def notify_shutdown(reason="Manual stop"):
    """Notify bot shutdown"""
    message = f"""
🛑 <b>BOT STOPPED</b>

<b>Reason:</b> {reason}

<i>Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</i>
"""
    return send_telegram(message.strip())
