import time
import threading
import datetime
import numpy as np
import pandas as pd
import torch
import alpaca_trade_api as tradeapi
import gym
import stockstats
import talib
import pandas as pd
import numpy as np
import datetime as dt

##########################################################
# 1) Function: fetch_latest_data_crypto
##########################################################

def fetch_latest_data_crypto(
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    API_BASE_URL,
    ticker_list,
    time_interval,          # e.g. '1Day', '5Min', '15Min'
    tech_indicator_list,    # e.g. ['macd','rsi','cci','dx']
    start_date=None,        # e.g. dt.date.today() - dt.timedelta(days=60)
    end_date=None,          # e.g. dt.date.today()
    exchanges: list = ['US'],
    select_stockstats_talib=0, # 0 => stockstats, 1 => talib
    data_source="alpaca",
):
    """
    Fetch recent crypto bars from Alpaca using official get_crypto_bars approach,
    parse the timeframe, optionally compute technical indicators,
    then return the last bar's price & technical arrays.

    Parameters
    ----------
    ALPACA_API_KEY : str
    ALPACA_SECRET_KEY : str
    API_BASE_URL : str
    ticker_list : list of str
        Crypto symbols, e.g. ["BTCUSD","ETHUSD","DOGEUSD"].
    time_interval : str
        E.g. '1Day', '5Min', '15Min'. We'll parse into tradeapi.TimeFrame().
    tech_indicator_list : list of str
        E.g. ['macd','rsi','cci','dx'] for either stockstats or talib approach.
    start_date : date or datetime
        Start date for fetching bars (inclusive).
    end_date : date or datetime
        End date for fetching bars (exclusive or inclusive depending on Alpaca).
    exchanges : list of str
        Alpaca crypto exchange(s). Default ['US'].
    select_stockstats_talib : int
        0 => use stockstats, 1 => use talib for computing indicators.
    data_source : str
        Just a label; "alpaca", "ccxt", etc.

    Returns
    -------
    latest_price : np.ndarray  (shape: (len(ticker_list),))
    latest_tech  : np.ndarray  (shape: (len(ticker_list), #indicators))
    latest_high  : np.ndarray  (shape: (len(ticker_list),))
    latest_low   : np.ndarray  (shape: (len(ticker_list),))
    latest_close : np.ndarray  (shape: (len(ticker_list),))

    If no data is found, returns two empty arrays.
    """
    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, API_BASE_URL)

    # 1) Parse time_interval into an Alpaca TimeFrame
    if time_interval.lower().endswith('day'):
        # e.g. '1Day'
        number_str = time_interval.lower().replace('day','')
        tf_value = int(number_str)
        tf_unit = tradeapi.TimeFrameUnit.Day
    elif time_interval.lower().endswith('min'):
        # e.g. '5Min','15Min'
        number_str = time_interval.lower().replace('min','')
        tf_value = int(number_str)
        tf_unit = tradeapi.TimeFrameUnit.Minute
    else:
        raise ValueError(f"Unsupported time_interval: {time_interval}")

    alpaca_tf = tradeapi.TimeFrame(tf_value, tf_unit)

    # 2) Default start/end if not provided
    if start_date is None:
        start_date = dt.date.today() - dt.timedelta(days=60)
    if end_date is None:
        end_date = dt.date.today()

    data_df_list = []

    # 3) Pull bars for each ticker
    for tic in ticker_list:
        barset = api.get_crypto_bars(
            symbol=tic,
            timeframe=alpaca_tf,
            start=start_date,
            end=end_date,
        )
        tmp_df = barset.df.reset_index()
        if tmp_df.empty:
            continue
        tmp_df['tic'] = tic
        tmp_df.rename(columns={'timestamp':'time'}, inplace=True)
        data_df_list.append(tmp_df)

    if len(data_df_list) == 0:
        # No data at all
        return np.array([]), np.array([])

    # 4) Concatenate into a single DataFrame
    df = pd.concat(data_df_list, ignore_index=True)

    # 5) Compute technical indicators
    df = _calculate_indicators_generic(
        df=df,
        data_source=data_source,
        tech_indicator_list=tech_indicator_list,
        select_stockstats_talib=select_stockstats_talib
    )
    if df.empty:
        return np.array([]), np.array([])

    # 6) Take the last row per ticker => "latest" bar
    latest_rows = df.groupby('tic', as_index=False).tail(1).copy()
    # Ensure the same order as ticker_list
    latest_rows['sort_order'] = latest_rows['tic'].apply(lambda x: ticker_list.index(x))
    latest_rows.sort_values('sort_order', inplace=True)

    # 7) Extract final arrays
    latest_price = latest_rows['close'].values.astype(np.float32)
    latest_high = latest_rows['high'].values.astype(np.float32)
    latest_low = latest_rows['low'].values.astype(np.float32)
    latest_close = latest_rows['close'].values.astype(np.float32)
    tech_list_data = []
    for indicator in tech_indicator_list:
        tech_list_data.append(latest_rows[indicator].values.astype(np.float32))
    latest_tech = np.array(tech_list_data).T  # shape (#tickers, #indicators)

    print(f"Latest Bars data: {latest_price}, {latest_tech}")
    return latest_price, latest_tech, latest_high, latest_low, latest_close


def _calculate_indicators_generic(
    df: pd.DataFrame,
    data_source: str,
    tech_indicator_list: list,
    select_stockstats_talib=0
) -> pd.DataFrame:
    """
    Helper function to compute indicators with either stockstats or talib.
    Expects columns: ['time','tic','open','high','low','close','volume'].
    Returns the same df with extra indicator columns appended.
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
                    # e.g. 'macd' might be auto-generated,
                    # or might need exact naming like 'rsi_14'.
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
        # TALib approach
        final_df = pd.DataFrame()
        for tic_val in df['tic'].unique():
            tic_df = df[df['tic'] == tic_val].copy()
            # Example for macd, rsi, cci, dx:
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

##########################################################
# 2) The Paper Trading Class
##########################################################

class AlpacaPaperTradingCryptoLive:
    """
    Paper trading class for crypto on Alpaca, calling fetch_latest_data_crypto
    directly from the same file. No turbulence logic, no market open logic.
    """

    def __init__(
        self,
        ticker_list,           # e.g. ["BTCUSD","ETHUSD","SOLUSD"]
        time_interval,         # e.g. '1Min' or '5Min'
        drl_lib,               # e.g. 'elegantrl'
        agent,                 # e.g. 'ppo'
        cwd,                   # directory with your actor.pth
        net_dim,               # e.g. [128, 64]
        state_dim,             # e.g. 1 + 3*action_dim + tech_dim
        action_dim,            # e.g. len(ticker_list)
        API_KEY,
        API_SECRET,
        API_BASE_URL,
        tech_indicator_list,   # e.g. ['macd','rsi','cci','dx']
        max_stock=1e2,         # scale factor for buy/sell
        tp_multiplier=2,       # TP multiplier
        sl_multiplier=1,       # SL multiplier
        atr_window=14,         # ATR window
        max_trade_duration=20  # Max trade duration
    ):
        self.API_BASE_URL = API_BASE_URL
        self.API_SECRET = API_SECRET
        self.API_KEY = API_KEY

        # 1) Load the trained PPO actor
        self.drl_lib = drl_lib
        if agent == 'ppo':
            if drl_lib == 'elegantrl':
                from torch import load
                from __main__ import AgentPPO  # or wherever AgentPPO is defined
                tmp_agent = AgentPPO(net_dim, state_dim, action_dim)
                actor = tmp_agent.act
                try:
                    actor_path = f"{cwd}/best_actor.pth"
                    print(f"| loading actor from: {actor_path}")
                    actor.load_state_dict(load(actor_path, map_location='cpu'))
                    self.act = actor
                    self.device = tmp_agent.device
                except BaseException:
                    raise ValueError("Fail to load agent!")
            else:
                raise ValueError("DRL library not supported in this snippet.")
        else:
            raise ValueError("Agent not supported yet.")

        # 2) Connect to Alpaca
        try:
            self.alpaca = tradeapi.REST(self.API_KEY, self.API_SECRET, self.API_BASE_URL)
            print("Alpaca connected.", self.alpaca)
        except Exception as e:
            raise ValueError(f"Fail to connect Alpaca. Error: {e}")

        # 3) Parse the time interval for the main loop
        if time_interval.lower() in ['1min', '5min', '15min']:
            num = int(time_interval.replace('min',''))
            self.time_interval = 60 * num
        else:
            if time_interval.endswith('s'):
                self.time_interval = int(time_interval[:-1])
            else:
                self.time_interval = 60

        self.tech_indicator_list = tech_indicator_list
        self.max_stock = max_stock

        # 4) Initialize account info
        self.stockUniverse = ticker_list
        self.stocks = np.zeros(len(ticker_list), dtype=float)
        self.stocks_cd = np.zeros_like(self.stocks)
        self.cash = float(self.alpaca.get_account().last_equity)
        self.price = np.zeros(len(ticker_list), dtype=float)
        self.equities = []

        self.current_step = 0
        self.in_position = False  # Initialize in_position
        self.entry_price = None
        self.target_price = None
        self.stop_loss_price = None
        self.trade_entry_step = None
        print(f"PaperTradingCryptoLive with tickers: {ticker_list}")
        print("Time interval (seconds) =", self.time_interval)

    def run(self):
        """
        Main loop:
        Sleep for self.time_interval seconds, fetch data, trade, etc.
        Crypto is 24/7, so no open/close logic.
        """
        # Cancel any open orders
        orders = self.alpaca.list_orders(status="open")
        for order in orders:
            self.alpaca.cancel_order(order.id)

        print("Starting infinite paper trading loop for crypto.")
        print('you have this much cash in account:', self.cash)
        while True:
            self.trade()
            self.current_step += 1
            last_equity = float(self.alpaca.get_account().last_equity)
            print('last_equity:', last_equity)
            self.equities.append((time.time(), last_equity))
            time.sleep(self.time_interval)

    def trade(self):
        """Fetch state, compute action, place trades accordingly."""
        state = self.get_state()
        if self.drl_lib == 'elegantrl':
            with torch.no_grad():
                s_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                a_tensor = self.act(s_tensor)
                action = a_tensor.detach().cpu().numpy()[0]
                print('Actions value at function trade before scaling,:', action)
        else:
            action = np.zeros(len(self.stockUniverse))
            if action > 0:
                # Calculate ATR
                atr = self._calculate_atr(self.price, self.price, self.price, self.atr_window)[0]
                self.entry_price = self.price[0]
                self.target_price = self.entry_price + self.tp_multiplier * atr
                self.stop_loss_price = self.entry_price - self.sl_multiplier * atr
                self.stop_loss_price = max(self.stop_loss_price, 0.01)  # Prevent negative SL

                # Place buy order
                qty = min(self.cash // self.price[0], abs(int(action * self.max_stock)))
                if qty > 0:
                    respSO = []
                    self.submitOrder(qty, self.stockUniverse[0], 'buy', respSO)
                    self.stocks_cd[0] = 0
                    self.in_position = True
                    self.trade_entry_step = self.current_step
                    print(f"Trade initiated: Entry={self.entry_price}, TP={self.target_price}, SL={self.stop_loss_price}")
            else:
                # Manage open trade
                current_price = self.price[0]
                if current_price >= self.target_price:
                    # Take profit
                    qty = self.stocks[0]
                    if qty > 0:
                        respSO = []
                        self.submitOrder(qty, self.stockUniverse[0], 'sell', respSO)
                        self.stocks_cd[0] = 0
                        self.in_position = False
                        print(f"Take profit: Sold at {current_price}")
                elif current_price <= self.stop_loss_price:
                    # Stop loss
                    qty = self.stocks[0]
                    if qty > 0:
                        respSO = []
                        self.submitOrder(qty, self.stockUniverse[0], 'sell', respSO)
                        self.stocks_cd[0] = 0
                        self.in_position = False
                        print(f"Stop loss: Sold at {current_price}")
                elif (self.current_step - self.trade_entry_step) >= self.max_trade_duration:
                    # Timeout exit
                    qty = self.stocks[0]
                    if qty > 0:
                        respSO = []
                        self.submitOrder(qty, self.stockUniverse[0], 'sell', respSO)
                        self.stocks_cd[0] = 0
                        self.in_position = False
                        print(f"Timeout exit: Sold at {current_price}")

            # update self.cash
            self.cash = float(self.alpaca.get_account().cash)
        self.cash = float(self.alpaca.get_account().cash)

    def get_state(self):
        """
        Fetch the latest bar or short history from fetch_latest_data_crypto
        (defined above in the same file).
        Build a state vector of [scaled_cash, price*scale, stocks*scale, stocks_cd, tech_indicators].
        """
        price, tech, high, low, close = fetch_latest_data_crypto(
            ALPACA_API_KEY= self.API_KEY,
            ALPACA_SECRET_KEY = self.API_SECRET,
            API_BASE_URL= self.API_BASE_URL,
            ticker_list=self.stockUniverse,
            time_interval='1Min',
            tech_indicator_list=self.tech_indicator_list,
            data_source="alpaca",
            select_stockstats_talib=1
        )

        if price.size == 0:
            return np.zeros(1 + 3*len(self.stockUniverse) + len(self.tech_indicator_list)*len(self.stockUniverse) + 2*len(self.stockUniverse), dtype=np.float32)

        # positions
        positions = self.alpaca.list_positions()
        stock_array = [0]*len(self.stockUniverse)
        for p in positions:
            sym = p.symbol
            if sym in self.stockUniverse:
                idx = self.stockUniverse.index(sym)
                stock_array[idx] = abs(int(float(p.qty)))
        self.stocks = np.array(stock_array, dtype=float)
        self.cash = float(self.alpaca.get_account().cash)
        self.price = price


        # Calculate percentage change in price
        if self.current_step == 0:
            self.last_pct_change = 0.0
        else:
            self.last_pct_change = (price[0] - self.price[0]) / self.price[0]

        # build state
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

    def _calculate_atr(self, high, low, close, window=14):
        """Compute ATR for each timestep using a rolling window."""
        n = close.shape[0]
        tr = np.zeros(n, dtype=np.float32)
        
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        atr = np.convolve(tr, np.ones(window)/window, mode='same')
        return atr.astype(np.float32)

    def submitOrder(self, qty, symbol, side, resp):
        """Place a MARKET order via Alpaca paper trading."""
        if qty > 0:
            try:
                self.alpaca.submit_order(symbol, qty, side, "market", time_in_force="gtc")
                print(f"{side.upper()} {qty} {symbol} completed.")
                resp.append(True)
            except Exception as e:
                print(f"{side.upper()} {qty} {symbol} failed. Error: {e}")
                resp.append(False)
        else:
            print(f"Quantity=0, skip order for {symbol}, side={side}")
            resp.append(True)
