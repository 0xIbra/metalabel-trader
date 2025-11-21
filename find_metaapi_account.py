"""
Helper script to list all MetaApi accounts and get the correct account ID
"""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

METAAPI_TOKEN = os.getenv('METAAPI_TOKEN')

async def list_accounts():
    """List all MetaApi accounts"""

    print("="*60)
    print("METAAPI ACCOUNT FINDER")
    print("="*60)

    if not METAAPI_TOKEN:
        print("\n❌ METAAPI_TOKEN not found in .env")
        return

    try:
        from metaapi_cloud_sdk import MetaApi

        print("\n✅ Connecting to MetaApi...")
        api = MetaApi(METAAPI_TOKEN)

        print("\n📋 Fetching your accounts...\n")
        accounts_api = api.metatrader_account_api
        accounts = await accounts_api.get_accounts_with_infinite_scroll_pagination()

        if not accounts:
            print("❌ No accounts found!")
            print("\nPlease create a demo account at:")
            print("https://app.metaapi.cloud/accounts")
            return

        print(f"Found {len(accounts)} account(s):\n")
        print("="*60)

        for i, account in enumerate(accounts, 1):
            print(f"\n{i}. {account.name}")
            print(f"   Account ID: {account.id}")
            print(f"   Type: {account.type}")

            # Handle optional attributes
            if hasattr(account, 'platform'):
                print(f"   Platform: {account.platform}")
            if hasattr(account, 'state'):
                print(f"   State: {account.state}")
            if hasattr(account, 'broker_name'):
                print(f"   Broker: {account.broker_name}")

            print(f"\n   ✅ FOUND - Use this account!")
            print(f"\n   Add to .env:")
            print(f"   METAAPI_ACCOUNT_ID={account.id}")

        print("\n" + "="*60)
        print("\n💡 TIP: Copy the Account ID of your demo account")
        print("   and update METAAPI_ACCOUNT_ID in .env file")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nCheck that METAAPI_TOKEN is correct in .env")

if __name__ == "__main__":
    asyncio.run(list_accounts())
