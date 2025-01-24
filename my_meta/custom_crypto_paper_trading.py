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


##########################################################
# 1) Function: fetch_latest_data_crypto
##########################################################
def fetch_latest_data_crypto(
    api,
    ticker_list,
    time_interval,
    tech_indicator_list,
    limit=100,
    data_source="alpaca",    # e.g. "alpaca" or "ccxt"
    select_stockstats_talib=0
):
    """
    Fetch recent crypto bars from Alpaca, compute real indicators, return the last price & tech arrays.
    """
    data_df_list = []

    # 1) Pull bars from Alpaca
    for tic in ticker_list:
        barset = api.get_crypto_bars(
            symbol=tic,
            timeframe=time_interval,
            limit=limit,
            exchanges='US'
        )
        tmp_df = barset.df.reset_index()  # => columns: ['symbol','timestamp','open','high','low','close','volume','tic']
        tmp_df['tic'] = tic
        # rename 'timestamp' -> 'time'
        tmp_df.rename(columns={'timestamp':'time'}, inplace=True)
        data_df_list.append(tmp_df)

    if len(data_df_list)==0:
        return np.array([]), np.array([])

    df = pd.concat(data_df_list, ignore_index=True)
    # Now df columns might be: ['symbol','time','open','high','low','close','volume','tic']

    # 2) Actually compute real indicators
    df = calculate_technical_indicators(
        df=df,
        data_source=data_source,
        tech_indicator_list=tech_indicator_list,
        select_stockstats_talib=select_stockstats_talib,
        drop_na_timesteps=1
    )

    # 3) Take the last row per ticker => "latest" bar
    #    We want 1 row per ticker for final state
    latest_rows = df.groupby('tic').tail(1).copy()
    # match order
    latest_rows['sort_order'] = latest_rows['tic'].apply(lambda x: ticker_list.index(x))
    latest_rows.sort_values('sort_order', inplace=True)

    # 4) Extract final arrays:
    latest_price = latest_rows['close'].values.astype(np.float32)

    # gather indicator columns from tech_indicator_list
    # e.g. if stockstats approach => we might have them named exactly same
    # if talib approach => they might be 'macd','rsi','cci','dx' in df
    tech_list_data = []
    for indicator in tech_indicator_list:
        tech_list_data.append(latest_rows[indicator].values.astype(np.float32))

    latest_tech = np.array(tech_list_data).T  # shape (#tickers, #indicators)

    return latest_price, latest_tech


def calculate_technical_indicators(
    df: pd.DataFrame,
    data_source: str,
    tech_indicator_list: list,
    select_stockstats_talib: int = 0,
    drop_na_timesteps: int = 1
) -> pd.DataFrame:
    """
    Calculate technical indicators using either stockstats or talib,
    then return the enriched dataframe.

    df must have columns: ['time','tic','open','high','low','close','volume'].

    data_source : e.g. "ccxt","binance","alpaca", etc. for minor naming differences
    tech_indicator_list : e.g. ['macd','rsi','cci','dx']
    select_stockstats_talib : 0 => stockstats, 1 => talib
    drop_na_timesteps : if 1, remove any timesteps with NaN indicator values
    """

    # rename date -> time if needed
    if "date" in df.columns:
        df.rename(columns={"date": "time"}, inplace=True)

    # if coming from ccxt or other source might need minor renaming
    if data_source == "ccxt":
        if "index" in df.columns:
            df.rename(columns={"index":"time"}, inplace=True)

    df.reset_index(drop=False, inplace=True)
    # remove weird multi-level columns if they exist
    if "level_1" in df.columns:
        df.drop(columns=["level_1"], inplace=True)
    if "level_0" in df.columns and "tic" not in df.columns:
        df.rename(columns={"level_0": "tic"}, inplace=True)

    print("tech_indicator_list:", tech_indicator_list)

    # === Stockstats approach
    if select_stockstats_talib == 0:
        stock = stockstats.StockDataFrame.retype(df)
        unique_tickers = stock.tic.unique()
        for indicator in tech_indicator_list:
            print("Calculating indicator:", indicator)
            indicator_df = pd.DataFrame()
            for tic in unique_tickers:
                try:
                    temp_indicator = stock[stock.tic == tic][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = tic
                    # match times
                    temp_indicator["time"] = df[df.tic == tic]["time"].to_list()
                    indicator_df = pd.concat([indicator_df, temp_indicator],
                                             ignore_index=True)
                except Exception as e:
                    print("Indicator error:", e)
            if not indicator_df.empty:
                df = df.merge(indicator_df[["tic","time",indicator]],
                              on=["tic","time"], how="left")

    # === Talib approach
    else:
        final_df = pd.DataFrame()
        for tic in df.tic.unique():
            tic_df = df[df.tic == tic].copy()
            # Example: compute macd, rsi, cci, dx using talib
            macd, macd_signal, macd_hist = talib.MACD(
                tic_df["close"], fastperiod=12, slowperiod=26, signalperiod=9
            )
            rsi = talib.RSI(tic_df["close"], timeperiod=14)
            cci = talib.CCI(tic_df["high"], tic_df["low"], tic_df["close"], timeperiod=14)
            dx  = talib.DX(tic_df["high"], tic_df["low"], tic_df["close"], timeperiod=14)

            tic_df["macd"] = macd
            tic_df["macd_signal"] = macd_signal
            tic_df["macd_hist"] = macd_hist
            tic_df["rsi"] = rsi
            tic_df["cci"] = cci
            tic_df["dx"] = dx

            final_df = pd.concat([final_df, tic_df], axis=0)

        df = final_df

    # sort & optionally drop NaN timesteps
    df.sort_values(by=["time","tic"], inplace=True)
    if drop_na_timesteps == 1:
        na_times = df[df.isna().any(axis=1)].time.unique()
        df = df[~df.time.isin(na_times)]

    df.reset_index(drop=True, inplace=True)
    print("Successfully added technical indicators.")
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
    ):
        # 1) Load the trained PPO actor
        self.drl_lib = drl_lib
        if agent == 'ppo':
            if drl_lib == 'elegantrl':
                from torch import load
                # from your code base or local definitions:
                from __main__ import AgentPPO  # or wherever AgentPPO is defined
                tmp_agent = AgentPPO(net_dim, state_dim, action_dim)
                actor = tmp_agent.act
                # Load agent weights
                try:
                    actor_path = f"{cwd}/actor.pth"
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
            self.alpaca = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, 'v2')
        except Exception as e:
            raise ValueError(f"Fail to connect Alpaca. Error: {e}")

        # 3) Parse the time interval for the main loop
        if time_interval.lower() in ['1min', '5min', '15min']:
            num = int(time_interval.replace('min',''))
            self.time_interval = 60 * num
        else:
            # fallback if '1s','5s', etc.
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
        self.cash = None
        self.price = np.zeros(len(ticker_list), dtype=float)
        self.equities = []

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
        while True:
            self.trade()
            last_equity = float(self.alpaca.get_account().last_equity)
            self.equities.append((time.time(), last_equity))
            time.sleep(self.time_interval)

    def trade(self):
        """Fetch state, compute action, place trades accordingly."""
        state = self.get_state()
        if self.drl_lib == 'elegantrl':
            with torch.no_grad():
                s_tensor = torch.as_tensor([state], dtype=torch.float32, device=self.device)
                a_tensor = self.act(s_tensor)
                action = a_tensor.detach().cpu().numpy()[0]
            action = (action * self.max_stock).astype(int)
        else:
            # placeholder
            action = np.zeros(len(self.stockUniverse))

        self.stocks_cd += 1
        min_action = 10

        # SELL
        for idx in np.where(action < -min_action)[0]:
            sell_num_shares = min(self.stocks[idx], -action[idx])
            qty = abs(int(sell_num_shares))
            if qty > 0:
                respSO = []
                self.submitOrder(qty, self.stockUniverse[idx], 'sell', respSO)
                self.stocks_cd[idx] = 0

        # BUY
        for idx in np.where(action > min_action)[0]:
            if self.cash is None or self.cash <= 0:
                continue
            buy_num_shares = min(self.cash // self.price[idx], abs(int(action[idx])))
            if np.isnan(buy_num_shares):
                qty = 0
            else:
                qty = abs(int(buy_num_shares))
            if qty > 0:
                respSO = []
                self.submitOrder(qty, self.stockUniverse[idx], 'buy', respSO)
                self.stocks_cd[idx] = 0

        # update self.cash
        self.cash = float(self.alpaca.get_account().cash)

    def get_state(self):
        """
        Fetch the latest bar or short history from fetch_latest_data_crypto
        (defined above in the same file).
        Build a state vector of [scaled_cash, price*scale, stocks*scale, stocks_cd, tech_indicators].
        """
        price, tech = fetch_latest_data_crypto(
            api=self.alpaca,
            ticker_list=self.stockUniverse,
            time_interval='1Min',  # or '5Min'
            tech_indicator_list=self.tech_indicator_list,
            limit=100,  # enough bars for MACD etc.
            data_source="alpaca",
            select_stockstats_talib=1  # if you want the talib approach
        )

        if price.size == 0:
            # In case no data was returned
            return np.zeros(1 + 3*len(self.stockUniverse) + len(self.tech_indicator_list)*len(self.stockUniverse), dtype=np.float32)

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

        # build state
        amount_scaled = np.array(self.cash*(2**-12), dtype=np.float32)
        scale = np.array(2**-6, dtype=np.float32)

        # tech shape => (#tickers, #indicators). Flatten if needed
        # e.g. we do [price_i*scale for each i], [stock_i * scale], stocks_cd, then flatten tech
        state = []
        state.append(amount_scaled)
        # price
        state.extend(list(price * scale))
        # stocks scaled
        state.extend(list(self.stocks * scale))
        # cooldown
        state.extend(list(self.stocks_cd))

        # flatten tech => shape (#tickers * #indicators)
        if len(tech.shape) == 2:
            tech_flat = tech.flatten()
            state.extend(list(tech_flat))
        else:
            # if it's 1D, just extend
            state.extend(list(tech))

        state = np.array(state, dtype=np.float32)
        # fill any NaN
        state[np.isnan(state)] = 0.0
        state[np.isinf(state)] = 0.0
        return state

    def submitOrder(self, qty, symbol, side, resp):
        """Place a MARKET order via Alpaca paper trading."""
        if qty > 0:
            try:
                self.alpaca.submit_order(symbol, qty, side, "market", "day")
                print(f"{side.upper()} {qty} {symbol} completed.")
                resp.append(True)
            except Exception as e:
                print(f"{side.upper()} {qty} {symbol} failed. Error: {e}")
                resp.append(False)
        else:
            print(f"Quantity=0, skip order for {symbol}, side={side}")
            resp.append(True)
























#########################################################
#       old calss of papertrading for reference
########################################################
# class AlpacaPaperTrading():

#     def __init__(self,ticker_list, time_interval, drl_lib, agent, cwd, net_dim, 
#                  state_dim, action_dim, API_KEY, API_SECRET, 
#                  API_BASE_URL, tech_indicator_list, turbulence_thresh=30, 
#                  max_stock=1e2, latency = None):
#         #load agent
#         self.drl_lib = drl_lib
#         if agent =='ppo':
#             if drl_lib == 'elegantrl':              
#                 agent_class = AgentPPO
#                 agent = agent_class(net_dim, state_dim, action_dim)
#                 actor = agent.act
#                 # load agent
#                 try:  
#                     cwd = cwd + '/actor.pth'
#                     print(f"| load actor from: {cwd}")
#                     actor.load_state_dict(torch.load(cwd, map_location=lambda storage, loc: storage))
#                     self.act = actor
#                     self.device = agent.device
#                 except BaseException:
#                     raise ValueError("Fail to load agent!")
                        
#             elif drl_lib == 'rllib':
#                 from ray.rllib.agents import ppo
#                 from ray.rllib.agents.ppo.ppo import PPOTrainer
                
#                 config = ppo.DEFAULT_CONFIG.copy()
#                 config['env'] = StockEnvEmpty
#                 config["log_level"] = "WARN"
#                 config['env_config'] = {'state_dim':state_dim,
#                             'action_dim':action_dim,}
#                 trainer = PPOTrainer(env=StockEnvEmpty, config=config)
#                 trainer.restore(cwd)
#                 try:
#                     trainer.restore(cwd)
#                     self.agent = trainer
#                     print("Restoring from checkpoint path", cwd)
#                 except:
#                     raise ValueError('Fail to load agent!')
                    
#             elif drl_lib == 'stable_baselines3':
#                 from stable_baselines3 import PPO
                
#                 try:
#                     #load agent
#                     self.model = PPO.load(cwd)
#                     print("Successfully load model", cwd)
#                 except:
#                     raise ValueError('Fail to load agent!')
                    
#             else:
#                 raise ValueError('The DRL library input is NOT supported yet. Please check your input.')
               
#         else:
#             raise ValueError('Agent input is NOT supported yet.')
            
            
            
#         #connect to Alpaca trading API
#         try:
#             self.alpaca = tradeapi.REST(API_KEY,API_SECRET,API_BASE_URL, 'v2')
#         except:
#             raise ValueError('Fail to connect Alpaca. Please check account info and internet connection.')
        
#         #read trading time interval
#         if time_interval == '1s':
#             self.time_interval = 1
#         elif time_interval == '5s':
#             self.time_interval = 5
#         elif time_interval == '1Min':
#             self.time_interval = 60
#         elif time_interval == '5Min':
#             self.time_interval = 60 * 5
#         elif time_interval == '15Min':
#             self.time_interval = 60 * 15
#         else:
#             raise ValueError('Time interval input is NOT supported yet.')
        
#         #read trading settings
#         self.tech_indicator_list = tech_indicator_list
#         self.turbulence_thresh = turbulence_thresh
#         self.max_stock = max_stock 
        
#         #initialize account
#         self.stocks = np.asarray([0] * len(ticker_list)) #stocks holding
#         self.stocks_cd = np.zeros_like(self.stocks) 
#         self.cash = None #cash record 
#         self.stocks_df = pd.DataFrame(self.stocks, columns=['stocks'], index = ticker_list)
#         self.asset_list = []
#         self.price = np.asarray([0] * len(ticker_list))
#         self.stockUniverse = ticker_list
#         self.turbulence_bool = 0
#         self.equities = []
        
#     def test_latency(self, test_times = 10): 
#         total_time = 0
#         for i in range(0, test_times):
#             time0 = time.time()
#             self.get_state()
#             time1 = time.time()
#             temp_time = time1 - time0
#             total_time += temp_time
#         latency = total_time/test_times
#         print('latency for data processing: ', latency)
#         return latency
        
#     def run(self):
#         orders = self.alpaca.list_orders(status="open")
#         for order in orders:
#           self.alpaca.cancel_order(order.id)
    
#         # Wait for market to open.
#         print("Waiting for market to open...")
#         tAMO = threading.Thread(target=self.awaitMarketOpen)
#         tAMO.start()
#         tAMO.join()
#         print("Market opened.")
#         while True:

#           # Figure out when the market will close so we can prepare to sell beforehand.
#           clock = self.alpaca.get_clock()
#           closingTime = clock.next_close.replace(tzinfo=datetime.timezone.utc).timestamp()
#           currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
#           self.timeToClose = closingTime - currTime
    
#           if(self.timeToClose < (60)):
#             # Close all positions when 1 minutes til market close.
#             print("Market closing soon. Stop trading.")
#             break
            
#             '''# Close all positions when 1 minutes til market close.
#             print("Market closing soon.  Closing positions.")
    
#             positions = self.alpaca.list_positions()
#             for position in positions:
#               if(position.side == 'long'):
#                 orderSide = 'sell'
#               else:
#                 orderSide = 'buy'
#               qty = abs(int(float(position.qty)))
#               respSO = []
#               tSubmitOrder = threading.Thread(target=self.submitOrder(qty, position.symbol, orderSide, respSO))
#               tSubmitOrder.start()
#               tSubmitOrder.join()
    
#             # Run script again after market close for next trading day.
#             print("Sleeping until market close (15 minutes).")
#             time.sleep(60 * 15)'''
            
#           else:
#             trade = threading.Thread(target=self.trade)
#             trade.start()
#             trade.join()
#             last_equity = float(self.alpaca.get_account().last_equity)
#             cur_time = time.time()
#             self.equities.append([cur_time,last_equity])
#             time.sleep(self.time_interval)
            
#     def awaitMarketOpen(self):
#         isOpen = self.alpaca.get_clock().is_open
#         while(not isOpen):
#           clock = self.alpaca.get_clock()
#           openingTime = clock.next_open.replace(tzinfo=datetime.timezone.utc).timestamp()
#           currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
#           timeToOpen = int((openingTime - currTime) / 60)
#           print(str(timeToOpen) + " minutes til market open.")
#           time.sleep(60)
#           isOpen = self.alpaca.get_clock().is_open
    
#     def trade(self):
#         state = self.get_state()
        
#         if self.drl_lib == 'elegantrl':
#             with torch.no_grad():
#                 s_tensor = torch.as_tensor((state,), device=self.device)
#                 a_tensor = self.act(s_tensor)  
#                 action = a_tensor.detach().cpu().numpy()[0]  
#             action = (action * self.max_stock).astype(int)
            
#         elif self.drl_lib == 'rllib':
#             action = self.agent.compute_single_action(state)
        
#         elif self.drl_lib == 'stable_baselines3':
#             action = self.model.predict(state)[0]
            
#         else:
#             raise ValueError('The DRL library input is NOT supported yet. Please check your input.')
        
#         self.stocks_cd += 1
#         if self.turbulence_bool == 0:
#             min_action = 10  # stock_cd
#             for index in np.where(action < -min_action)[0]:  # sell_index:
#                 sell_num_shares = min(self.stocks[index], -action[index])
#                 qty =  abs(int(sell_num_shares))
#                 respSO = []
#                 tSubmitOrder = threading.Thread(target=self.submitOrder(qty, self.stockUniverse[index], 'sell', respSO))
#                 tSubmitOrder.start()
#                 tSubmitOrder.join()
#                 self.cash = float(self.alpaca.get_account().cash)
#                 self.stocks_cd[index] = 0

#             for index in np.where(action > min_action)[0]:  # buy_index:
#                 if self.cash < 0:
#                     tmp_cash = 0
#                 else:
#                     tmp_cash = self.cash
#                 buy_num_shares = min(tmp_cash // self.price[index], abs(int(action[index])))
#                 if (buy_num_shares != buy_num_shares): # if buy_num_change = nan
#                     qty = 0 # set to 0 quantity
#                 else:
#                     qty = abs(int(buy_num_shares))
#                 qty = abs(int(buy_num_shares))
#                 respSO = []
#                 tSubmitOrder = threading.Thread(target=self.submitOrder(qty, self.stockUniverse[index], 'buy', respSO))
#                 tSubmitOrder.start()
#                 tSubmitOrder.join()
#                 self.cash = float(self.alpaca.get_account().cash)
#                 self.stocks_cd[index] = 0
                
#         else:  # sell all when turbulence
#             positions = self.alpaca.list_positions()
#             for position in positions:
#                 if(position.side == 'long'):
#                     orderSide = 'sell'
#                 else:
#                     orderSide = 'buy'
#                 qty = abs(int(float(position.qty)))
#                 respSO = []
#                 tSubmitOrder = threading.Thread(target=self.submitOrder(qty, position.symbol, orderSide, respSO))
#                 tSubmitOrder.start()
#                 tSubmitOrder.join()
            
#             self.stocks_cd[:] = 0
            
    
#     def get_state(self):
#         alpaca = AlpacaProcessor(api=self.alpaca)
#         price, tech, turbulence = alpaca.fetch_latest_data(ticker_list = self.stockUniverse, time_interval='1Min',
#                                                      tech_indicator_list=self.tech_indicator_list)
#         turbulence_bool = 1 if turbulence >= self.turbulence_thresh else 0
        
#         turbulence = (self.sigmoid_sign(turbulence, self.turbulence_thresh) * 2 ** -5).astype(np.float32)
        
#         tech = tech * 2 ** -7
#         positions = self.alpaca.list_positions()
#         stocks = [0] * len(self.stockUniverse)
#         for position in positions:
#             ind = self.stockUniverse.index(position.symbol)
#             stocks[ind] = ( abs(int(float(position.qty))))
        
#         stocks = np.asarray(stocks, dtype = float)
#         cash = float(self.alpaca.get_account().cash)
#         self.cash = cash
#         self.stocks = stocks
#         self.turbulence_bool = turbulence_bool 
#         self.price = price
        
        
        
#         amount = np.array(self.cash * (2 ** -12), dtype=np.float32)
#         scale = np.array(2 ** -6, dtype=np.float32)
#         state = np.hstack((amount,
#                     turbulence,
#                     self.turbulence_bool,
#                     price * scale,
#                     self.stocks * scale,
#                     self.stocks_cd,
#                     tech,
#                     )).astype(np.float32)
#         state[np.isnan(state)] = 0.0
#         state[np.isinf(state)] = 0.0
#         print(len(self.stockUniverse))
#         return state
        
#     def submitOrder(self, qty, stock, side, resp):
#         if(qty > 0):
#           try:
#             self.alpaca.submit_order(stock, qty, side, "market", "day")
#             print("Market order of | " + str(qty) + " " + stock + " " + side + " | completed.")
#             resp.append(True)
#           except:
#             print("Order of | " + str(qty) + " " + stock + " " + side + " | did not go through.")
#             resp.append(False)
#         else:
#           print("Quantity is 0, order of | " + str(qty) + " " + stock + " " + side + " | not completed.")
#           resp.append(True)

#     @staticmethod
#     def sigmoid_sign(ary, thresh):
#         def sigmoid(x):
#             return 1 / (1 + np.exp(-x * np.e)) - 0.5

#         return sigmoid(ary / thresh) * thresh