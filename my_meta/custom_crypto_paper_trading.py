import time
import threading
import datetime as dt
import numpy as np
import pandas as pd
import torch
import alpaca_trade_api as tradeapi
import gym
import stockstats
import talib
import logging
from my_meta.single_asset_training_ppo import AgentPPO  # Adjust the import as needed

def convert_time_interval_to_timedelta(time_interval: str) -> dt.timedelta:
    """Convert a time_interval string (e.g., '1Day', '5Min', '15Min', '30s') into a timedelta."""
    time_interval = time_interval.lower()
    if time_interval.endswith('day'):
        value = int(time_interval.replace('day', ''))
        return dt.timedelta(days=value)
    elif time_interval.endswith('hour'):
        value = int(time_interval.replace('hour', ''))
        return dt.timedelta(hours=value)
    elif time_interval.endswith('min'):
        value = int(time_interval.replace('min', ''))
        return dt.timedelta(minutes=value)
    elif time_interval.endswith('s'):
        value = int(time_interval.replace('s', ''))
        return dt.timedelta(seconds=value)
    else:
        raise ValueError(f"Unsupported time_interval: {time_interval}")

def _calculate_indicators_generic(
    df: pd.DataFrame,
    data_source: str,
    tech_indicator_list: list,
    select_stockstats_talib=0
) -> pd.DataFrame:
    """
    Helper function to compute technical indicators using either stockstats or TA‑Lib.
    Expects columns: ['time','tic','open','high','low','close','volume'].
    """
    df.sort_values(by=['time','tic'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if select_stockstats_talib == 0:
        # Stockstats approach
        stock = stockstats.StockDataFrame.retype(df.copy(deep=True))
        unique_tic = stock['tic'].unique()
        for indicator in tech_indicator_list:
            indicator_df = pd.DataFrame()
            for tic in unique_tic:
                sub_df = stock[stock['tic'] == tic]
                if indicator not in sub_df.columns:
                    continue
                temp_vals = sub_df[indicator]
                tmp2 = pd.DataFrame({
                    'time': sub_df['time'],
                    'tic': tic,
                    indicator: temp_vals
                })
                indicator_df = pd.concat([indicator_df, tmp2], ignore_index=True)
            if not indicator_df.empty:
                df = df.merge(indicator_df, on=['time','tic'], how='left')
    else:
        # TA‑Lib approach
        final_df = pd.DataFrame()
        for tic_val in df['tic'].unique():
            tic_df = df[df['tic'] == tic_val].copy()
            if 'macd' in tech_indicator_list or 'macd_signal' in tech_indicator_list or 'macd_hist' in tech_indicator_list:
                macd, macd_signal, macd_hist = talib.MACD(
                    tic_df['close'], fastperiod=12, slowperiod=26, signalperiod=9
                )
                tic_df['macd'] = macd
                tic_df['macd_signal'] = macd_signal
                tic_df['macd_hist'] = macd_hist
            if 'rsi' in tech_indicator_list:
                tic_df['rsi'] = talib.RSI(tic_df["close"], timeperiod=14)
            if 'cci' in tech_indicator_list:
                tic_df['cci'] = talib.CCI(
                    tic_df["high"], tic_df["low"], tic_df["close"], timeperiod=14
                )
            if 'dx' in tech_indicator_list:
                tic_df['dx'] = talib.DX(
                    tic_df["high"], tic_df["low"], tic_df["close"], timeperiod=14
                )
            final_df = pd.concat([final_df, tic_df], ignore_index=True)
        df = final_df

    df.sort_values(by=['time','tic'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.dropna(axis=0, how='any').reset_index(drop=True)
    return df

def fetch_latest_data_crypto(
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    API_BASE_URL,
    ticker_list,
    time_interval,          # e.g., '1Day', '5Min', '15Min'
    tech_indicator_list,    # e.g., ['macd','rsi','cci','dx']
    start_date=None,
    end_date=None,
    exchanges: list = ['US'],
    select_stockstats_talib=0,  # 0 => stockstats, 1 => TA‑Lib
    data_source="alpaca",
    atr_window=14           # ATR window (in number of bars)
):
    """
    Fetch recent crypto bars from Alpaca, compute technical indicators,
    and return the latest bar’s data plus the full historical dataframe.
    """
    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, API_BASE_URL)

    # 1) Parse time_interval into Alpaca’s TimeFrame
    if time_interval.lower().endswith('day'):
        number_str = time_interval.lower().replace('day','')
        tf_value = int(number_str)
        tf_unit = tradeapi.TimeFrameUnit.Day
    elif time_interval.lower().endswith('min'):
        number_str = time_interval.lower().replace('min','')
        tf_value = int(number_str)
        tf_unit = tradeapi.TimeFrameUnit.Minute
    else:
        raise ValueError(f"Unsupported time_interval: {time_interval}")

    # 2) Determine history window based on bar resolution
    bar_delta = convert_time_interval_to_timedelta(time_interval)
    required_bars = 60 + atr_window  # e.g., 60 bars for indicators plus the ATR window
    if start_date is None:
        start_date = dt.datetime.now() - required_bars * bar_delta
    if end_date is None:
        end_date = dt.datetime.now()

    # Format start and end into RFC3339 strings (no microseconds, T separator, Z suffix)
    start_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    data_df_list = []
    # 3) Pull bars for each ticker
    for tic in ticker_list:
        barset = api.get_crypto_bars(
            symbol=tic,
            timeframe=tradeapi.TimeFrame(tf_value, tf_unit),
            start=start_str,
            end=end_str,
        )
        tmp_df = barset.df.reset_index()
        if tmp_df.empty:
            continue
        tmp_df['tic'] = tic
        tmp_df.rename(columns={'timestamp':'time'}, inplace=True)
        data_df_list.append(tmp_df)

    if len(data_df_list) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), pd.DataFrame()

    # 4) Concatenate all dataframes
    df = pd.concat(data_df_list, ignore_index=True)

    # 5) Aggregate to 30Min if needed (retaining your original aggregation logic)
    if time_interval.lower() == '30min':
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df = df.groupby('tic').resample('30T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()

    # 6) Compute technical indicators
    df = _calculate_indicators_generic(
        df=df,
        data_source=data_source,
        tech_indicator_list=tech_indicator_list,
        select_stockstats_talib=select_stockstats_talib
    )
    if df.empty:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), pd.DataFrame()

    # 7) Select the latest bar for each ticker
    latest_rows = df.groupby('tic', as_index=False).tail(1).copy()
    latest_rows['sort_order'] = latest_rows['tic'].apply(lambda x: ticker_list.index(x))
    latest_rows.sort_values('sort_order', inplace=True)

    latest_price = latest_rows['close'].values.astype(np.float32)
    latest_high = latest_rows['high'].values.astype(np.float32)
    latest_low = latest_rows['low'].values.astype(np.float32)
    latest_close = latest_rows['close'].values.astype(np.float32)
    tech_list_data = []
    for indicator in tech_indicator_list:
        tech_list_data.append(latest_rows[indicator].values.astype(np.float32))
    latest_tech = np.array(tech_list_data).T  # shape: (#tickers, #indicators)

    return latest_price, latest_tech, latest_high, latest_low, latest_close, df


##########################################################################
#  The Paper Trading Class (Refactored)
##########################################################################

class AlpacaPaperTradingCryptoLive:
    """
    Paper trading class for crypto on Alpaca using TA‑Lib for ATR calculations.
    Unnecessary functions (fetch_latest_bar_data, fetch_historical_data, custom ATR)
    have been removed. Data is fetched via fetch_latest_data_crypto.
    """

    def __init__(
        self,
        ticker_list,           # e.g., ["BTCUSD","ETHUSD","SOLUSD"]
        time_interval,         # e.g., '1Min' or '5Min'
        drl_lib,               # e.g., 'elegantrl'
        agent,                 # e.g., 'ppo'
        cwd,                   # directory with your actor.pth
        net_dim,               # e.g., [128, 64]
        state_dim,             # e.g., 1 + 3*action_dim + tech_dim
        action_dim,            # e.g., len(ticker_list)
        API_KEY,
        API_SECRET,
        API_BASE_URL,
        tech_indicator_list,   # e.g., ['macd','rsi','cci','dx']
        max_stock=1e2,         # scale factor for buy/sell
        tp_multiplier=2,       # TP multiplier
        sl_multiplier=1,       # SL multiplier
        atr_window=14,         # ATR window (in bars)
        max_trade_duration=20, # Max trade duration in steps
        initial_capital=10000, # Initial capital for paper trading
        log_file_path=None     # Path to the log file
    ):
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if log_file_path:
            handler = logging.FileHandler(log_file_path, mode='w')  # 'w' mode to replace contents
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        else:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info("Initializing AlpacaPaperTradingCryptoLive")
        self.API_SECRET = API_SECRET
        self.API_KEY = API_KEY

        # 1) Load the trained PPO actor (example using elegantrl)
        self.drl_lib = drl_lib
        if agent == 'ppo':
            if drl_lib == 'elegantrl':
                from torch import load
                # from __main__ import AgentPPO  # Adjust the import as needed
                tmp_agent = AgentPPO(net_dim, state_dim, action_dim)
                actor = tmp_agent.act
                try:
                    actor_path = f"{cwd}/best_actor.pth"
                    self.logger.info(f"| loading actor from: {actor_path}")
                    actor.load_state_dict(load(actor_path, map_location='cpu'))
                    self.act = actor
                    self.device = tmp_agent.device
                except Exception as e:
                    self.logger.error(f"Error loading actor: {e}")
                    raise ValueError("Fail to load agent!") from e
            else:
                raise ValueError("DRL library not supported in this snippet.")
        else:
            raise ValueError("Agent not supported yet.")

        # 2) Connect to Alpaca
        try:
            self.alpaca = tradeapi.REST(self.API_KEY, self.API_SECRET, self.API_BASE_URL)
            self.logger.info("Alpaca connected.")
        except Exception as e:
            self.logger.error(f"Fail to connect Alpaca. Error: {e}")
            raise ValueError(f"Fail to connect Alpaca. Error: {e}")

        self.time_interval = time_interval
        self.tech_indicator_list = tech_indicator_list
        self.max_stock = max_stock
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        self.atr_window = atr_window
        self.max_trade_duration = max_trade_duration

        self.stockUniverse = ticker_list
        self.stocks = np.zeros(len(ticker_list), dtype=float)
        self.stocks_cd = np.zeros_like(self.stocks)
        self.cash = initial_capital
        self.initial_cash = initial_capital
        self.price = np.zeros(len(ticker_list), dtype=float)
        self.equities = []
        self.current_step = 0
        self.in_position = False
        self.entry_price = None
        self.target_price = None
        self.stop_loss_price = None
        self.trade_entry_step = None

        self.stop_trading = False  # Flag to control the trading loop
        self.logger.info(f"PaperTradingCryptoLive with tickers: {ticker_list}")
        self.logger.info(f"Time interval = {self.time_interval}")

    def run(self):
        """Main loop: fetch data, trade, etc. (crypto is 24/7)."""
        # Cancel any open orders
        orders = self.alpaca.list_orders(status="open")
        for order in orders:
            self.alpaca.cancel_order(order.id)

        self.logger.info("Starting infinite paper trading loop for crypto.")
        self.logger.info(f'Initial cash balance: {self.cash}')
        while not self.stop_trading:  # Check the flag
            self.trade()
            self.current_step += 1
            last_equity = float(self.alpaca.get_account().last_equity)
            self.logger.info(f'Last equity: {last_equity}')
            self.equities.append((time.time(), last_equity))
            # Calculate sleep time based on time_interval
            if self.time_interval.endswith('min'):
                num_minutes = int(self.time_interval.replace('min', ''))
                sleep_time = num_minutes * 60
            elif self.time_interval.endswith('s'):
                sleep_time = int(self.time_interval.replace('s', ''))
            else:
                sleep_time = 60  # default fallback
            self.logger.info(f"Next bar data will be fetched in {sleep_time // 60} minutes.")
            time.sleep(sleep_time)

    def stop(self):
        """Stop the trading loop."""
        self.stop_trading = True

    def trade(self):
        """Fetch state, compute action, and place trades accordingly."""
        state = self.get_state()
        if self.drl_lib == 'elegantrl':
            with torch.no_grad():
                s_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                a_tensor = self.act(s_tensor)
                action = a_tensor.detach().cpu().numpy()[0]
                print('Action from actor (before scaling):', action)
        else:
            action = np.zeros(len(self.stockUniverse))

        self.stocks_cd += 1

        if not self.in_position:
            if action > 0:
                # Use fetch_latest_data_crypto to get both latest data and historical data
                price, tech, high, low, close, hist_df = fetch_latest_data_crypto(
                    ALPACA_API_KEY=self.API_KEY,
                    ALPACA_SECRET_KEY=self.API_SECRET,
                    API_BASE_URL=self.API_BASE_URL,
                    ticker_list=self.stockUniverse,
                    time_interval=self.time_interval,
                    tech_indicator_list=self.tech_indicator_list,
                    data_source="alpaca",
                    select_stockstats_talib=1,
                    atr_window=self.atr_window
                )
                if len(hist_df) >= self.atr_window:
                    # Calculate ATR using TA‑Lib
                    atr_values = talib.ATR(
                        hist_df['high'].values,
                        hist_df['low'].values,
                        hist_df['close'].values,
                        timeperiod=self.atr_window
                    )
                    atr = atr_values[-1]
                    self.logger.info(f"ATR value: {atr}")
                    self.entry_price = price[0]
                    self.target_price = self.entry_price + self.tp_multiplier * atr
                    self.stop_loss_price = max(self.entry_price - self.sl_multiplier * atr, 0.01)
                    self.logger.info(f"Entry Price: {self.entry_price}, TP: {self.target_price}, SL: {self.stop_loss_price}")

                    # Place a buy order using all available cash
                    qty = self.initial_cash / self.entry_price
                    self.logger.info(f"Calculated quantity to buy: {qty:.6f}")
                    if qty > 0:
                        respSO = []
                        self.submitOrder(qty, self.stockUniverse[0], 'buy', respSO)
                        self.stocks_cd[0] = 0
                        self.in_position = True
                        self.trade_entry_step = self.current_step
                        self.logger.info(f"Trade initiated: Entry={self.entry_price}, TP={self.target_price}, SL={self.stop_loss_price}")
                else:
                    self.logger.info("Not enough historical data for ATR calculation.")
        else:
            # Manage open trade
            current_price = self.price[0]
            self.logger.info(f"Trade open. Current price: {current_price}, TP: {self.target_price}, SL: {self.stop_loss_price}")
            if current_price >= self.target_price:
                qty = self.stocks[0]
                if qty > 0:
                    respSO = []
                    self.submitOrder(qty, self.stockUniverse[0], 'sell', respSO)
                    self.stocks_cd[0] = 0
                    self.in_position = False
                    self.logger.info(f"Take profit: Sold at {current_price}")
            elif current_price <= self.stop_loss_price:
                qty = self.stocks[0]
                if qty > 0:
                    respSO = []
                    self.submitOrder(qty, self.stockUniverse[0], 'sell', respSO)
                    self.stocks_cd[0] = 0
                    self.in_position = False
                    self.logger.info(f"Stop loss: Sold at {current_price}")
            elif (self.current_step - self.trade_entry_step) >= self.max_trade_duration:
                qty = self.stocks[0]
                if qty > 0:
                    respSO = []
                    self.submitOrder(qty, self.stockUniverse[0], 'sell', respSO)
                    self.stocks_cd[0] = 0
                    self.in_position = False
                    self.logger.info(f"Timeout exit: Sold at {current_price}")

        # Update cash balance
        self.cash = float(self.alpaca.get_account().cash)

    def get_state(self):
        """
        Build a state vector by fetching the latest data via fetch_latest_data_crypto.
        State includes a scaled price, technical indicators, the last percentage change, and an in-trade flag.
        """
        price, tech, high, low, close, _ = fetch_latest_data_crypto(
            ALPACA_API_KEY=self.API_KEY,
            ALPACA_SECRET_KEY=self.API_SECRET,
            API_BASE_URL=self.API_BASE_URL,
            ticker_list=self.stockUniverse,
            time_interval=self.time_interval,
            tech_indicator_list=self.tech_indicator_list,
            data_source="alpaca",
            select_stockstats_talib=1,
            atr_window=self.atr_window
        )

        if price.size == 0:
            total_length = 1 + len(tech.flatten()) + 2  # scaled_price + indicators + last_pct_change + in_trade_flag
            return np.zeros(total_length, dtype=np.float32)

        # Get current positions
        positions = self.alpaca.list_positions()
        stock_array = [0] * len(self.stockUniverse)
        for p in positions:
            sym = p.symbol
            if sym in self.stockUniverse:
                idx = self.stockUniverse.index(sym)
                stock_array[idx] = abs(int(float(p.qty)))
        self.stocks = np.array(stock_array, dtype=float)
        self.cash = float(self.alpaca.get_account().cash)
        self.price = price

        self.logger.info(f"State update - Price: {self.price}, Cash: {self.cash}, Stocks: {self.stocks}")

        if self.current_step == 0:
            self.last_pct_change = 0.0
        else:
            self.last_pct_change = (price[0] - self.price[0]) / self.price[0]

        scaled_price = price[0] / self.price[0]
        tech_features = tech.flatten() if len(tech.shape) == 2 else tech
        in_trade_flag = 1.0 if self.in_position else 0.0

        state = np.hstack([
            np.array([scaled_price], dtype=np.float32).flatten(),
            tech_features,
            np.array([self.last_pct_change], dtype=np.float32).flatten(),
            np.array([in_trade_flag], dtype=np.float32).flatten()
        ]).astype(np.float32)

        state[np.isnan(state)] = 0.0
        state[np.isinf(state)] = 0.0
        return state

    def submitOrder(self, qty, symbol, side, resp):
        """Place a MARKET order via Alpaca paper trading."""
        if qty > 0:
            try:
                self.alpaca.submit_order(symbol, qty, side, "market", time_in_force="gtc")
                self.logger.info(f"{side.upper()} {qty} {symbol} completed.")
                resp.append(True)
            except Exception as e:
                self.logger.error(f"{side.upper()} {qty} {symbol} failed. Error: {e}")
                resp.append(False)
        else:
            self.logger.info(f"Quantity=0, skipping order for {symbol} ({side}).")
            resp.append(True)
