import gymnasium as gym
import numpy as np

class CryptoTradingEnv(gym.Env):
    """
    A revised crypto trading environment with the following behavior:
      - The state vector consists of:
           • Scaled price (current price / initial price)
           • Standardized technical indicators (via z-score normalization)
           • Previous step’s percentage change in price
           • In-trade flag (1 if a trade is open, else 0)
      - The reward is defined as:
           • If no trade is open: the natural percentage change from the previous bar.
           • If a trade is open: the current percentage change relative to the entry price.
      - The network’s action (a scalar) is interpreted as a target percentage change signal.
          For example, an output of +0.02 means “expect a +2% move.”
      - If action > 0 and no trade is open, a trade is initiated:
             1. A market order is executed to buy (using available cash).
             2. A target exit price and a stop-loss price are set.
      - Once a trade is open, at each new bar the environment:
             • Computes the current return relative to the entry price as the reward.
             • Checks whether the current price has reached the target or stop loss:
                  - If so, the trade is closed.
                  - Otherwise, the trade remains open (reward is the intermediate return).
    """

    def __init__(self, config, initial_account=10000, stop_loss_ratio=2.5):
        # Price array: shape (T, 1) (assumes one asset)
        self.price_ary = config["price_array"].astype(np.float32)
        # Technical indicators: shape (T, tech_features)
        self.tech_ary = config["tech_array"].astype(np.float32)
        self.if_train = config["if_train"]
        self.initial_capital = initial_account
        self.stop_loss_ratio = stop_loss_ratio

        # ***** Standardize Technical Indicators using z-score *****
        self.tech_mean = np.mean(self.tech_ary, axis=0)
        self.tech_std = np.std(self.tech_ary, axis=0)
        self.tech_ary = (self.tech_ary - self.tech_mean) / (self.tech_std + 1e-8)
        # ************************************************************

        self.timestep = self.price_ary.shape[0]
        self.current_step = 0
        self.env_name = "CryptoTradingEnv"

        # State vector: [scaled_price, tech_features, last_pct_change, in_trade_flag]
        tech_dim = self.tech_ary.shape[1]
        self.state_dim = 1 + tech_dim + 1 + 1
        self.action_dim = 1  # single scalar output

        self.initial_price = self.price_ary[0, 0]
        self.last_pct_change = 0.0

        # Trade-related variables.
        self.in_position = False
        self.entry_price = None
        self.target_price = None
        self.stop_loss_price = None

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
        return self._get_state(), {}

    def _get_state(self):
        """Construct the state vector."""
        current_price = self.price_ary[self.current_step, 0]
        scaled_price = current_price / self.initial_price
        tech_features = np.array(self.tech_ary[self.current_step], dtype=np.float32).flatten()
        pct_change = np.array([self.last_pct_change], dtype=np.float32).flatten()
        in_trade_flag = np.array([1.0], dtype=np.float32) if self.in_position else np.array([0.0], dtype=np.float32)
        return np.hstack([np.array([scaled_price], dtype=np.float32).flatten(),
                          tech_features,
                          pct_change,
                          in_trade_flag]).astype(np.float32)

    def step(self, action):
        """
        Process one time step.
        - If no trade is open and action > 0, initiate a trade.
        - If no trade is open and action <= 0, simply move to the next bar (reward = natural price change).
        - If a trade is open, at each new bar:
             • Compute reward = (current price - entry_price) / entry_price.
             • If current price >= target_price: close trade (target hit).
             • If current price <= stop_loss_price: close trade (stop loss hit).
             • Otherwise, leave the trade open (reward = intermediate return).
        """
        done = False
        info = {}
        reward = 0.0

        current_price = self.price_ary[self.current_step, 0]

        if not self.in_position:
            # Not in a trade.
            signal = action[0]
            if signal > 0:
                # Initiate trade.
                self.in_position = True
                self.entry_price = current_price
                self.target_price = current_price * (1 + signal)
                self.stop_loss_price = current_price * (1 - signal * self.stop_loss_ratio)
                info["trade"] = f"Trade initiated at {float(current_price):.2f}: target {float(self.target_price):.2f}, stop {float(self.stop_loss_price):.2f}"
                reward = 0.0
                self.last_pct_change = 0.0
            else:
                # No trade signal: move to next bar.
                if self.current_step < self.timestep - 1:
                    prev_price = current_price
                    self.current_step += 1
                    new_price = self.price_ary[self.current_step, 0]
                    reward = (new_price - prev_price) / prev_price
                    self.last_pct_change = reward
                else:
                    done = True
        else:
            # Trade is open.
            self.current_step += 1
            if self.current_step >= self.timestep:
                done = True
                exit_price = self.price_ary[-1, 0]
            else:
                exit_price = self.price_ary[self.current_step, 0]

            # Calculate the current trade return relative to entry.
            current_trade_return = (exit_price - self.entry_price) / self.entry_price

            # Check exit conditions:
            if exit_price >= self.target_price:
                # Target reached: trade closed.
                reward = (self.target_price - self.entry_price) / self.entry_price
                info["trade_result"] = f"Target hit: exit at {float(self.target_price):.2f}, reward {float(reward):.4f}"
                self.in_position = False
                self.entry_price = None
                self.target_price = None
                self.stop_loss_price = None
                self.last_pct_change = reward
            elif exit_price <= self.stop_loss_price:
                # Stop loss reached: trade closed.
                reward = (self.stop_loss_price - self.entry_price) / self.entry_price
                info["trade_result"] = f"Stop loss hit: exit at {float(self.stop_loss_price):.2f}, reward {float(reward):.4f}"
                self.in_position = False
                self.entry_price = None
                self.target_price = None
                self.stop_loss_price = None
                self.last_pct_change = reward
            else:
                # Trade remains open: reward is the current trade return.
                reward = current_trade_return
                info["trade_status"] = f"Trade open: current return {float(reward):.4f}"
                self.last_pct_change = reward

        if self.current_step >= self.timestep - 1:
            done = True

        next_state = self._get_state()
        return next_state, reward, done, False, info
