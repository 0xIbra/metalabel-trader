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

def notify_trade_entry(symbol, side, entry_price, sl_price, tp_price):
    """Notify trade entry"""
    # Calculate pips
    sl_pips = abs(entry_price - sl_price) * (100 if 'JPY' in symbol else 10000)
    tp_pips = abs(tp_price - entry_price) * (100 if 'JPY' in symbol else 10000)

    message = f"""
🚀 <b>ENTRY: {symbol} ({side})</b>
--------------------------------
<b>Price:</b> {entry_price:.5f}
<b>TP:</b>    {tp_price:.5f} (+{tp_pips:.1f} pips)
<b>SL:</b>    {sl_price:.5f} (-{sl_pips:.1f} pips)
--------------------------------
<i>Godmode Trend Strategy</i>
"""
    return send_telegram(message.strip())

def notify_trade_exit(symbol, pnl, reason):
    """Notify trade exit"""
    emoji = "💰" if pnl > 0 else "🛑"
    result = "PROFIT" if pnl > 0 else "LOSS"

    message = f"""
{emoji} <b>EXIT: {symbol} ({result})</b>
--------------------------------
<b>PnL:</b>    €{pnl:+.2f}
<b>Reason:</b> {reason}
--------------------------------
"""
    return send_telegram(message.strip())

def notify_status(balance, open_positions, trades_today, pnl_today, win_rate_today):
    """Send status update"""
    message = f"""
📊 <b>HOURLY STATUS</b>
--------------------------------
<b>Balance:</b>  €{balance:.2f}
<b>Open:</b>     {open_positions}
<b>Trades:</b>   {trades_today}
<b>PnL Today:</b> €{pnl_today:+.2f}
<b>Win Rate:</b> {win_rate_today:.1f}%
--------------------------------
"""
    return send_telegram(message.strip())

def notify_error(error_message, action="Investigating..."):
    """Notify error"""
    message = f"""
⚠️ <b>ERROR</b>
--------------------------------
<b>Message:</b> {error_message}
<b>Action:</b>  {action}
--------------------------------
"""
    return send_telegram(message.strip())

def notify_startup(symbols, confidence_threshold, account_balance):
    """Notify bot startup"""
    symbol_list = ', '.join(symbols[:5]) + f" +{len(symbols)-5} more" if len(symbols) > 5 else ', '.join(symbols)
    message = f"""
🚀 <b>GODMODE BOT ONLINE</b>
--------------------------------
<b>Strategy:</b> Godmode Trend (Long+Short)
<b>Symbols:</b>  {len(symbols)} pairs
<b>Balance:</b>  €{account_balance:.2f}
<b>Model:</b>    Strict OOS 2024
--------------------------------
<i>Live Trading Mode</i>
"""
    return send_telegram(message.strip())

def notify_shutdown(reason="Manual stop"):
    """Notify bot shutdown"""
    message = f"""
🛑 <b>BOT OFFLINE</b>
--------------------------------
<b>Reason:</b> {reason}
--------------------------------
"""
    return send_telegram(message.strip())
