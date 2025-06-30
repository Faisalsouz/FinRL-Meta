from my_meta.custom_crypto_binance_trading import BinancePaperTradingCryptoLive
from dotenv import load_dotenv
import os
API_KEY    = os.getenv("BINANCE_TESTNET_KEY")
API_SECRET = os.getenv("BINANCE_TESTNET_SECRET")

paper_bot = BinancePaperTradingCryptoLive(
        ticker_list        = ["SOLUSDT"],
        time_interval      = "1min",
        drl_lib            = "elegantrl",
        agent              = "ppo",
        cwd                = "api/papertrading_crypto",
        net_dim            = [256,128,64,32],
        state_dim          = 1 + 4 + 1 + 1,        # 4 tech indicators
        action_dim         = 1,
        API_KEY            = API_KEY,
        API_SECRET         = API_SECRET,
        
        tech_indicator_list= ["macd","rsi","cci","dx"],
        initial_capital    = 10_000,
        log_file_path      = "api/papertrading_crypto/papertrading_binance.log",
        trade_budget= 5000,
)

paper_bot.run()

