import gymnasium as gym
import numpy as np

class CryptoTradingEnv(gym.Env):
    """
    A revised crypto trading environment with the following behavior:
      - The state vector consists of:
           • Scaled price (current price / initial price)
           • Standardized technical indicators (via z-score normalization)
           • Previous step’s percentage change in price
           • In-trade flag (1 if a trade is open, 0 otherwise)
      - The reward is defined as the percentage change in price (expressed as a fraction).
      - The network’s action (a scalar) is interpreted as a target percentage change signal.
          For example, an output of 0.02 means “expect a +2% move.”
      - If action > 0 and no trade is open, a trade is initiated:
             1. A market order is executed (using available cash).
             2. A target exit price and a stop-loss price are set.
      - Once a trade is open, at each new bar:
             • The reward is calculated as the current return relative to the entry price.
             • If the price reaches the target or stop loss, the trade is closed.
             • Otherwise, if the trade remains open and the number of bars since entry exceeds a
               maximum trade duration, the trade is forced to exit (a "timeout" exit).
    """

    def __init__(self, config, initial_account=10000, atr_window=14, tp_multiplier=2, sl_multiplier=1):
        """
        Parameters:
          config: dict with keys:
             - "price_array": np.ndarray of shape (timesteps, 1)
             - "tech_array": np.ndarray of shape (timesteps, tech_features)
             - "if_train": bool
             - Optionally, "max_trade_duration": int (max number of bars an open trade is allowed to remain open)
          initial_account: starting cash in dollars.
        """
        # Load data.
        self.price_ary = config["price_array"].astype(np.float32)
        self.tech_ary = config["tech_array"].astype(np.float32)
        self.if_train = config["if_train"]
        self.initial_capital = initial_account
        # Load high, low, close prices for ATR calculation.
        self.high_ary = config["high_array"].astype(np.float32)
        self.low_ary = config["low_array"].astype(np.float32)
        self.close_ary = config["close_array"].astype(np.float32)
        
        # Precompute ATR
        self.atr_ary = self._calculate_atr(atr_window)
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier

        # Standardize technical indicators using z-score normalization.
        self.tech_mean = np.mean(self.tech_ary, axis=0)
        self.tech_std = np.std(self.tech_ary, axis=0)
        self.tech_ary = (self.tech_ary - self.tech_mean) / (self.tech_std + 1e-8)

        self.timestep = self.price_ary.shape[0]
        self.current_step = 0
        self.env_name = "CryptoTradingEnv"
        # Allow user to set a maximum trade duration (in number of candles). Default: 20.
        self.max_trade_duration = config.get("max_trade_duration", 20)

        # State vector: [scaled price, tech_features, last_pct_change, in_trade_flag]
        tech_dim = self.tech_ary.shape[1]
        self.state_dim = 1 + tech_dim + 1 + 1
        self.action_dim = 1  # network outputs a single scalar signal

        self.initial_price = self.price_ary[0, 0]
        self.last_pct_change = 0.0

        # Trade-related variables.
        self.in_position = False
        self.entry_price = None
        self.target_price = None
        self.stop_loss_price = None
        self.trade_entry_step = None  # New: record the step when trade is initiated

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.reset()

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.last_pct_change = 0.0
        self.in_position = False
        self.entry_price = None
        self.target_price = None
        self.stop_loss_price = None
        self.trade_entry_step = None
        print(f"Resetting environment. Initial price: {self.initial_price}")
        return self._get_state(), {}

    def _get_state(self):
        """Construct the state vector."""
        current_price = self.price_ary[self.current_step, 0]
        scaled_price = current_price / self.initial_price
        tech_features = np.array(self.tech_ary[self.current_step], dtype=np.float32).flatten()
        pct_change = np.array([self.last_pct_change], dtype=np.float32).flatten()
        in_trade_flag = np.array([1.0], dtype=np.float32) if self.in_position else np.array([0.0], dtype=np.float32)
        state = np.hstack([
            np.array([scaled_price], dtype=np.float32).flatten(),
            tech_features,
            pct_change,
            in_trade_flag
        ]).astype(np.float32)
        print(f"State at step {self.current_step}: {state}")
        return state


    def step(self, action):
        done = False
        info = {}
        reward = 0.0
        current_price = self.price_ary[self.current_step, 0]

        if not self.in_position:
            # Not in trade - check if opening new position
            signal = action[0]
            if signal > 0:
                # ========== ATR-BASED TP/SL SETUP ========== #
                current_atr = self.atr_ary[self.current_step]
                
                # Calculate TP/SL using multipliers
                self.entry_price = current_price
                self.target_price = self.entry_price + self.tp_multiplier * current_atr
                self.stop_loss_price = self.entry_price - self.sl_multiplier * current_atr
                
                # Validate prices aren't negative (important for crypto)
                self.stop_loss_price = max(self.stop_loss_price, 0.01)  # Prevent negative SL
                
                self.in_position = True
                self.trade_entry_step = self.current_step
                info["trade"] = (
                    f"Opened: Entry={self.entry_price:.2f}, "
                    f"TP={self.target_price:.2f} (+{self.tp_multiplier}ATR), "
                    f"SL={self.stop_loss_price:.2f} (-{self.sl_multiplier}ATR)"
                )
            else:
                # No trade opened
                if self.current_step < self.timestep - 1:
                    prev_price = current_price
                    self.current_step += 1
                    new_price = self.price_ary[self.current_step, 0]
                    self.last_pct_change = (new_price - prev_price) / prev_price
                else:
                    done = True
        else:
            # Trade is active - check exit conditions
            self.current_step += 1
            
            if self.current_step >= self.timestep:
                # End of data - force exit
                exit_price = self.price_ary[-1, 0]
                reward = (exit_price - self.entry_price) / self.entry_price
                self._close_trade("forced exit (data end)", exit_price)
                done = True
            else:
                exit_price = self.price_ary[self.current_step, 0]
                current_return = (exit_price - self.entry_price) / self.entry_price

                # ========== CHECK EXIT CONDITIONS ========== #
                # 1. Take Profit Hit
                if exit_price >= self.target_price:
                    reward = (self.target_price - self.entry_price) / self.entry_price
                    info["trade_result"] = f"TP Hit: return={reward:.4f}"
                    info["exit_price"] = self.target_price
                    info["entry_price"] = self.entry_price
                    info["atr"] = self.atr_ary[self.current_step]
                    info["tp_mult"] = self.tp_multiplier
                    info["sl_mult"] = self.sl_multiplier
                    self._close_trade("TP hit", self.target_price)
      
                
                # 2. Stop Loss Hit
                elif exit_price <= self.stop_loss_price:
                    reward = (self.stop_loss_price - self.entry_price) / self.entry_price
                    info["trade_result"] = f"SL Hit: return={reward:.4f}"
                    info["exit_price"] = self.stop_loss_price
                    info["entry_price"] = self.entry_price
                    info["atr"] = self.atr_ary[self.current_step]
                    info["tp_mult"] = self.tp_multiplier
                    info["sl_mult"] = self.sl_multiplier
                    self._close_trade("SL hit", self.stop_loss_price)
                
                
                # 3. Timeout Exit
                elif (self.current_step - self.trade_entry_step) >= self.max_trade_duration:
                    reward = current_return
                    info["trade_result"] = f"Timeout: return={reward:.4f}"
                    info["exit_price"] = exit_price
                    info["entry_price"] = self.entry_price
                    info["atr"] = self.atr_ary[self.current_step]
                    info["tp_mult"] = self.tp_multiplier
                    info["sl_mult"] = self.sl_multiplier
                    self._close_trade("timeout", exit_price)
      
                
                # 4. Trade remains open
                else:
                    reward = current_return
                    info["trade_status"] = (
                        f"Open: Unrealized={reward:.4f}, "
                        f"Current Price={exit_price:.2f}, "
                        f"TP Remaining={self.target_price - exit_price:.2f}"
                    )

        # Episode termination check
        if self.current_step >= self.timestep - 1:
            done = True

        next_state = self._get_state()
        return next_state, reward, done, False, info

    def _close_trade(self, reason, exit_price):
        """Helper to reset trade-related variables"""
        self.in_position = False
        self.entry_price = None
        self.target_price = None
        self.stop_loss_price = None
        self.trade_entry_step = None
        self.last_pct_change = (exit_price - self.entry_price) / self.entry_price if self.entry_price else 0.0
        
    def _calculate_atr(self, window: int) -> np.ndarray:
        """Compute ATR for each timestep using a rolling window."""
        # Flatten the close, high, and low arrays to ensure they are 1D.
        close_flat = self.close_ary.flatten()
        high_flat = self.high_ary.flatten()
        low_flat = self.low_ary.flatten()
        
        n = close_flat.shape[0]
        tr = np.zeros(n, dtype=np.float32)
        
        for i in range(1, n):
            tr[i] = max(
                high_flat[i] - low_flat[i],
                abs(high_flat[i] - close_flat[i-1]),
                abs(low_flat[i] - close_flat[i-1])
            )
        atr = np.convolve(tr, np.ones(window)/window, mode='same')
        return atr.astype(np.float32)


