"""
Test MetaApi connection and trade execution
Places a dummy trade and immediately closes it
"""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

METAAPI_TOKEN = os.getenv('METAAPI_TOKEN')
METAAPI_ACCOUNT_ID = os.getenv('METAAPI_ACCOUNT_ID')

async def test_metaapi():
    """Test MetaApi connection and trade execution"""

    print("="*60)
    print("METAAPI CONNECTION TEST")
    print("="*60)

    if not METAAPI_TOKEN or not METAAPI_ACCOUNT_ID:
        print("\n❌ MetaApi not configured!")
        print("Please check METAAPI_TOKEN and METAAPI_ACCOUNT_ID in .env")
        return

    try:
        from metaapi_cloud_sdk import MetaApi

        print("\n✅ MetaApi SDK imported")
        print(f"\nAccount ID: {METAAPI_ACCOUNT_ID}")

        # Connect to MetaApi
        print("\n1. Connecting to MetaApi...")
        api = MetaApi(METAAPI_TOKEN)
        account = await api.metatrader_account_api.get_account(METAAPI_ACCOUNT_ID)
        print(f"   ✅ Account found: {account.name}")

        # Deploy account
        print("\n2. Deploying account...")
        await account.deploy()
        print("   ✅ Account deployed")

        # Wait for connection
        print("\n3. Waiting for connection...")
        await account.wait_connected()
        print("   ✅ Connected")

        # Get RPC connection
        print("\n4. Getting RPC connection...")
        connection = account.get_rpc_connection()
        await connection.connect()
        await connection.wait_synchronized()
        print("   ✅ Synchronized")

        # Get account info
        print("\n5. Getting account information...")
        account_info = await connection.get_account_information()
        print(f"   ✅ Balance: ${account_info['balance']:.2f}")
        print(f"   ✅ Equity: ${account_info['equity']:.2f}")
        print(f"   ✅ Leverage: 1:{account_info['leverage']}")
        print(f"   ✅ Currency: {account_info['currency']}")

        # Get current price
        print("\n6. Getting EURUSD price...")
        price = await connection.get_symbol_price('EURUSD')
        print(f"   ✅ Bid: {price['bid']:.5f}")
        print(f"   ✅ Ask: {price['ask']:.5f}")

        # Place dummy order
        print("\n7. Placing dummy BUY order (0.01 lot)...")
        result = await connection.create_market_buy_order(
            symbol='EURUSD',
            volume=0.01,  # Minimum lot size
            stop_loss=price['ask'] - 0.0010,  # 10 pips SL
            take_profit=price['ask'] + 0.0010  # 10 pips TP
        )

        order_id = result['orderId']
        print(f"   ✅ Order placed! ID: {order_id}")

        # Wait a moment
        print("\n8. Waiting 5 seconds...")
        await asyncio.sleep(5)

        # Get positions
        print("\n9. Getting open positions...")
        positions = await connection.get_positions()
        if positions:
            print(f"   ✅ Found {len(positions)} position(s)")
            for pos in positions:
                print(f"      - {pos['symbol']}: {pos['type']} {pos['volume']} lots @ {pos['openPrice']:.5f}")
        else:
            print("   ⚠️ No positions found (order may have hit TP/SL already)")

        # Close the position
        print("\n10. Closing position...")
        try:
            if positions:
                position_id = positions[0]['id']
                await connection.close_position(position_id)
                print("   ✅ Position closed successfully")
            else:
                print("   ⚠️ No position to close")
        except Exception as e:
            print(f"   ⚠️ Close failed (position may have already closed): {e}")

        # Final account info
        print("\n11. Final account information...")
        account_info = await connection.get_account_information()
        print(f"   ✅ Balance: ${account_info['balance']:.2f}")
        print(f"   ✅ Equity: ${account_info['equity']:.2f}")

        print("\n" + "="*60)
        print("TEST COMPLETE - ALL SYSTEMS OPERATIONAL! ✅")
        print("="*60)
        print("\nYour MetaApi account is ready for live trading!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Verify METAAPI_TOKEN and METAAPI_ACCOUNT_ID in .env")
        print("2. Check that account is a demo/paper account")
        print("3. Ensure account is deployed in MetaApi dashboard")
        print("4. Try restarting the account in MetaApi dashboard")

if __name__ == "__main__":
    asyncio.run(test_metaapi())
