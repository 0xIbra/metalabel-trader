"""
Quick analysis of different Take-Profit levels
Tests how TP size affects win rate and profitability
"""

# EURUSD M1 Statistics (from our 90-day data)
avg_m1_range = 2.5  # pips per minute (typical)
avg_30min_range = 30 * 0.3  # ~9 pips in 30 minutes

# Theoretical Analysis
print("=" * 80)
print("TAKE-PROFIT LEVEL ANALYSIS")
print("=" * 80)

tp_scenarios = [
    {"tp": 1, "est_win_rate": 0.55, "timeout": 20},
    {"tp": 2, "est_win_rate": 0.40, "timeout": 30},
    {"tp": 5, "est_win_rate": 0.25, "timeout": 60},
    {"tp": 10, "est_win_rate": 0.15, "timeout": 120},
    {"tp": 20, "est_win_rate": 0.08, "timeout": 240},
]

print(f"\n{'TP (pips)':<12} {'Est. Win %':<15} {'RR Ratio':<12} {'Expectancy':<15} {'Timeout (bars)'}")
print("-" * 80)

for s in tp_scenarios:
    tp = s['tp']
    sl = 1  # Fixed 1 pip SL
    win_rate = s['est_win_rate']
    rr_ratio = tp / sl

    # Expectancy = (Win% × Win$) - (Loss% × Loss$)
    # Assuming $0.90 per pip for 0.09 lot
    pip_value = 0.90
    expectancy_per_trade = (win_rate * tp * pip_value) - ((1 - win_rate) * sl * pip_value)

    print(f"{tp:<12} {win_rate:<15.0%} {rr_ratio:<12.1f}:1 ${expectancy_per_trade:<14.2f} {s['timeout']}")

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

print("""
1. **Sweet Spot**: 2-5 pips seems optimal
   - TP=1: High win rate but low profit per win
   - TP=2: Balanced (current)
   - TP=5: Lower win rate but 5:1 RR
   - TP=10: Very rare to hit (only 15% of time)

2. **Volatility Constraint**:
   - EURUSD M1 avg range: ~2.5 pips/minute
   - 30-min window: ~9 pips typical max
   - 10-pip move = rare event (>1 std dev)

3. **Model Limitation**:
   - Our features (20-bar window) capture short-term patterns
   - Can't reliably predict 10-pip moves with M1 features
   - Would need higher timeframe features (M5, M15)

4. **Timeout Trade-off**:
   - TP=10 pips needs ~120 bars (2 hours) to develop
   - Longer timeouts = more capital tied up
   - More exposure to reversals

Recommendation: Test TP=3-5 pips with 60-bar timeout
This might give better balance than 2 pips.
""")

print("\n" + "=" * 80)
print("WANT TO TEST TP=5 PIPS?")
print("=" * 80)
print("Expected with TP=5, SL=1 (5:1 RR):")
print("  - Need only 17% win rate to break even")
print("  - Estimated 25% win rate = profitable")
print("  - Each win: $4.50, Each loss: $0.90")
print("  - 100 trades: 25 wins × $4.50 - 75 losses × $0.90 = +$44.25")
print("\nShall I retrain with TP=5 pips?")
