"""
Test Telegram bot connection and notifications
"""
import sys
sys.path.append('.')

from src.notifications.telegram_bot import (
    send_telegram, notify_startup, notify_trade_entry,
    notify_trade_exit, notify_status, TELEGRAM_ENABLED
)

def main():
    print("="*60)
    print("TELEGRAM BOT TEST")
    print("="*60)

    if not TELEGRAM_ENABLED:
        print("\n❌ Telegram not configured!")
        print("Please check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        return

    print("\n✅ Telegram configured")
    print("\nSending test messages...\n")

    # Test 1: Simple message
    print("1. Sending simple test message...")
    success = send_telegram("🧪 <b>Test Message</b>\n\nTelegram bot is working!")
    if success:
        print("   ✅ Sent successfully")
    else:
        print("   ❌ Failed to send")

    # Test 2: Startup notification
    print("\n2. Sending startup notification...")
    success = notify_startup(['EURUSD', 'AUDUSD'], 0.50, 1000)
    if success:
        print("   ✅ Sent successfully")
    else:
        print("   ❌ Failed to send")

    # Test 3: Trade entry notification
    print("\n3. Sending trade entry notification...")
    success = notify_trade_entry(
        symbol='EURUSD',
        side='BUY',
        entry_price=1.08450,
        tp_price=1.08500,
        sl_price=1.08440,
        lot_size=0.09,
        risk_amount=20.0,
        confidence=56.2
    )
    if success:
        print("   ✅ Sent successfully")
    else:
        print("   ❌ Failed to send")

    # Test 4: Trade exit notification
    print("\n4. Sending trade exit notification...")
    success = notify_trade_exit(
        symbol='EURUSD',
        exit_price=1.08500,
        pnl=4.50,
        pips=5.0,
        reason='TP Hit',
        hold_time_minutes=23,
        balance=1004.50
    )
    if success:
        print("   ✅ Sent successfully")
    else:
        print("   ❌ Failed to send")

    # Test 5: Status update
    print("\n5. Sending status update...")
    success = notify_status(
        balance=1004.50,
        open_positions=1,
        trades_today=3,
        pnl_today=4.50,
        win_rate_today=66.7
    )
    if success:
        print("   ✅ Sent successfully")
    else:
        print("   ❌ Failed to send")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\nCheck your Telegram app for 5 test messages!")

if __name__ == "__main__":
    main()
