#!/usr/bin/env python3
"""
Watchdog script to monitor live trading bot
Sends Telegram alerts if bot crashes and optionally restarts it
"""

import os
import sys
import time
import psutil
import subprocess
from datetime import datetime
from dotenv import load_dotenv

# Add project to path
sys.path.append(os.path.dirname(__file__))

from src.notifications.telegram_bot import send_telegram

# Load environment
load_dotenv()

# Configuration
BOT_SCRIPT = "live_trading_bot.py"
CHECK_INTERVAL = 300  # Check every 5 minutes
AUTO_RESTART = True  # Auto-restart if bot crashes
MAX_RESTARTS = 3  # Max restarts per hour
RESTART_WINDOW = 3600  # 1 hour window for restart counting

# Tracking
restart_times = []
bot_process = None
last_alert_time = None


def find_bot_process():
    """Find the trading bot process"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and BOT_SCRIPT in ' '.join(cmdline):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def is_bot_running():
    """Check if bot is currently running"""
    global bot_process

    # Check if we have a cached process
    if bot_process and bot_process.is_running():
        return True

    # Search for process
    bot_process = find_bot_process()
    return bot_process is not None


def send_alert(message, title="🚨 WATCHDOG ALERT"):
    """Send Telegram alert (rate limited)"""
    global last_alert_time

    # Rate limit: don't spam alerts (min 5 minutes between alerts)
    now = time.time()
    if last_alert_time and (now - last_alert_time) < 300:
        print(f"[WATCHDOG] Alert suppressed (rate limit): {message}")
        return

    full_message = f"<b>{title}</b>\n\n{message}\n\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    send_telegram(full_message)
    last_alert_time = now
    print(f"[WATCHDOG] Alert sent: {message}")


def can_restart():
    """Check if we can restart (not exceeding restart limit)"""
    global restart_times

    # Clean old restart times outside window
    now = time.time()
    restart_times = [t for t in restart_times if now - t < RESTART_WINDOW]

    # Check if under limit
    if len(restart_times) >= MAX_RESTARTS:
        return False

    return True


def restart_bot():
    """Attempt to restart the bot"""
    global bot_process, restart_times

    if not can_restart():
        send_alert(
            f"❌ Bot crashed but restart limit reached ({MAX_RESTARTS} restarts in last hour).\n\n"
            "Manual intervention required!",
            "🚨 CRITICAL: RESTART LIMIT REACHED"
        )
        return False

    try:
        print(f"[WATCHDOG] Attempting to restart bot...")

        # Start bot in background
        process = subprocess.Popen(
            [sys.executable, BOT_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )

        # Wait a bit to see if it starts successfully
        time.sleep(5)

        if process.poll() is None:  # Still running
            bot_process = psutil.Process(process.pid)
            restart_times.append(time.time())

            send_alert(
                f"✅ Bot restarted successfully (PID: {process.pid})\n\n"
                f"Restarts in last hour: {len(restart_times)}/{MAX_RESTARTS}",
                "🔄 BOT RESTARTED"
            )
            print(f"[WATCHDOG] Bot restarted successfully (PID: {process.pid})")
            return True
        else:
            # Process died immediately
            stdout, stderr = process.communicate()
            error_msg = stderr.decode() if stderr else "Unknown error"

            send_alert(
                f"❌ Bot restart failed!\n\n"
                f"Error: {error_msg[:200]}",
                "🚨 RESTART FAILED"
            )
            print(f"[WATCHDOG] Restart failed: {error_msg}")
            return False

    except Exception as e:
        send_alert(
            f"❌ Failed to restart bot!\n\n"
            f"Error: {str(e)}",
            "🚨 RESTART FAILED"
        )
        print(f"[WATCHDOG] Restart exception: {e}")
        return False


def monitor():
    """Main monitoring loop"""
    print(f"[WATCHDOG] Starting watchdog for {BOT_SCRIPT}")
    print(f"[WATCHDOG] Check interval: {CHECK_INTERVAL}s")
    print(f"[WATCHDOG] Auto-restart: {AUTO_RESTART}")
    print(f"[WATCHDOG] Max restarts: {MAX_RESTARTS}/hour")

    # Send startup notification
    send_alert(
        f"🐕 Watchdog started\n\n"
        f"Monitoring: {BOT_SCRIPT}\n"
        f"Check interval: {CHECK_INTERVAL}s\n"
        f"Auto-restart: {AUTO_RESTART}",
        "🐕 WATCHDOG STARTED"
    )

    consecutive_failures = 0

    while True:
        try:
            if is_bot_running():
                if consecutive_failures > 0:
                    print(f"[WATCHDOG] Bot is back online")
                    consecutive_failures = 0
                else:
                    print(f"[WATCHDOG] Bot is running (PID: {bot_process.pid})")
            else:
                consecutive_failures += 1
                print(f"[WATCHDOG] ⚠️ Bot is NOT running! (Failure #{consecutive_failures})")

                if consecutive_failures == 1:
                    # First detection - send alert
                    send_alert(
                        f"⚠️ Trading bot is not running!\n\n"
                        f"Bot process not found.\n"
                        f"Auto-restart: {AUTO_RESTART}",
                        "🚨 BOT DOWN"
                    )

                if AUTO_RESTART and consecutive_failures >= 2:
                    # Second check confirms it's really down - attempt restart
                    print(f"[WATCHDOG] Attempting restart...")
                    if restart_bot():
                        consecutive_failures = 0
                    else:
                        # Restart failed, send critical alert
                        if consecutive_failures >= 3:
                            send_alert(
                                f"🚨 CRITICAL: Bot has been down for {consecutive_failures * CHECK_INTERVAL // 60} minutes!\n\n"
                                f"Restart attempts failed.\n"
                                f"Manual intervention required!",
                                "🚨 CRITICAL ALERT"
                            )

            # Sleep until next check
            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            print(f"\n[WATCHDOG] Shutting down...")
            send_alert(
                "🛑 Watchdog stopped manually",
                "🐕 WATCHDOG STOPPED"
            )
            break
        except Exception as e:
            print(f"[WATCHDOG] Error in monitor loop: {e}")
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    monitor()
