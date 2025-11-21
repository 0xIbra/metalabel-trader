#!/bin/bash
# Quick setup script for systemd services
# This automatically configures the paths based on your current directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_USER="${USER}"

echo "🔧 Configuring systemd services for MetaLabel Trading Bot"
echo "Directory: $SCRIPT_DIR"
echo "User: $SERVICE_USER"
echo ""

# Create temporary service files with correct paths
sed "s|/home/ubuntu/metalabel-trader|$SCRIPT_DIR|g; s|User=ubuntu|User=$SERVICE_USER|g" \
    trading-bot.service > /tmp/trading-bot.service

sed "s|/home/ubuntu/metalabel-trader|$SCRIPT_DIR|g; s|User=ubuntu|User=$SERVICE_USER|g" \
    watchdog.service > /tmp/watchdog.service

# Copy to systemd directory
echo "Installing service files..."
sudo cp /tmp/trading-bot.service /etc/systemd/system/
sudo cp /tmp/watchdog.service /etc/systemd/system/

# Reload systemd
echo "Reloading systemd..."
sudo systemctl daemon-reload

# Enable services
echo "Enabling services..."
sudo systemctl enable trading-bot.service
sudo systemctl enable watchdog.service

echo ""
echo "✅ Setup complete!"
echo ""
echo "Commands:"
echo "  Start:   sudo systemctl start trading-bot watchdog"
echo "  Status:  sudo systemctl status trading-bot"
echo "  Logs:    sudo journalctl -u trading-bot -f"
echo "  Stop:    sudo systemctl stop trading-bot watchdog"

# Clean up
rm /tmp/trading-bot.service /tmp/watchdog.service
