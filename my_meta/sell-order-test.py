import numpy as np
import pandas as pd
from types import SimpleNamespace
from my_meta.custom_crypto_paper_trading import AlpacaPaperTradingCryptoLive

# 1) A fake fetch that accepts *any* signature and returns two bars:
def make_fake_fetch(prices, atrs):
    prices = list(prices)
    atrs   = list(atrs)
    def fake_fetch(*args, **kwargs):
        # Pop next bar price & ATR
        p = prices.pop(0)
        a = atrs.pop(0)
        # Build the single‐bar arrays
        price_arr = np.array([p], dtype=np.float32)
        tech_dim  = len(kwargs.get("tech_indicator_list", []))
        tech_arr  = np.zeros((1, tech_dim), dtype=np.float32)
        high_arr  = np.array([p], dtype=np.float32)
        low_arr   = np.array([p], dtype=np.float32)
        close_arr = np.array([p], dtype=np.float32)
        # Build a dummy hist_df with enough rows for ATR
        window = kwargs.get("atr_window", 1)
        dummy_hist = pd.DataFrame({
            "high":  np.full(window + 1, p, dtype=np.float32),
            "low":   np.full(window + 1, p, dtype=np.float32),
            "close": np.full(window + 1, p, dtype=np.float32),
        })
        return price_arr, tech_arr, high_arr, low_arr, close_arr, dummy_hist

    return fake_fetch

# 2) Dummy Alpaca client that just prints orders:
class DummyAccount:
    last_equity = "10000"
    cash        = "10000"

class DummyAlpaca:
    def list_orders(self, status): return []
    def cancel_order(self, oid): pass
    def submit_order(self, symbol, qty, side, order_type, time_in_force):
        print(f">>> SUBMIT {side.upper():<4} {qty:.6f} {symbol}")
    def get_account(self):     return DummyAccount()
    def list_positions(self):  return []

# 3) Build your tester
def make_tester():
    import my_meta.custom_crypto_paper_trading as mod

    # Monkey‐patch the fetch function
    mod.fetch_latest_data_crypto = make_fake_fetch(
        prices=[100.0, 110.0],   # first call → 100, second → 110
        atrs=[1.0, 1.0]          # ATR stays constant
    )

    trader = AlpacaPaperTradingCryptoLive(
        ticker_list=["BTCUSD"],
        time_interval="30Min",
        drl_lib="elegantrl",
        agent="ppo",
        cwd="/home/souz_wsl/finrl_proj/FinRL_Meta/api/papertrading_crypto",  # where best_actor.pth lives
        net_dim=[256,128,64,32],   # match your trained network
        state_dim=7,               # 1 price + 4 techs + 1 pct_change + 1 flag
        action_dim=1,
        API_KEY="PK0HYWTXV4JHULYVK2KY", API_SECRET="11ErKPAJ8JdCy6LbSYRWPcEzdzZGse4iJgSzGl7g", API_BASE_URL="https://paper-api.alpaca.markets",
        tech_indicator_list=["macd","rsi","cci","dx"],
        atr_window=1
    )

    # Swap out the real Alpaca client
    trader.alpaca = DummyAlpaca()

    # Force TP = entry + 1*ATR = 101, SL = entry - 1*ATR = 99
    trader.tp_multiplier = 1
    trader.sl_multiplier = 1
    return trader

if __name__ == "__main__":
    t = make_tester()

    print("=== BAR #1: should BUY at 100 ===")
    t.trade()

    print("\n=== BAR #2: should SELL at TP=101 (price=110) ===")
    t.trade()
