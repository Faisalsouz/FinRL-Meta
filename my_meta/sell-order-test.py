from dotenv import load_dotenv
import os
from binance.client import Client

# 1) pull variables from the .env file (or the shell)
load_dotenv()                       # does nothing if .env is absent
API_KEY    = os.getenv("BINANCE_TESTNET_KEY")
API_SECRET = os.getenv("BINANCE_TESTNET_SECRET")

# 2) point the SDK at the testnet endpoint
client = Client(API_KEY, API_SECRET, testnet=True)
# – or – (same result)
# client = Client(API_KEY, API_SECRET, base_url="https://testnet.binance.vision")

# 3) quick sanity check
info = client.get_account()         # authenticated call
print("Balances:", info["balances"][:3])
