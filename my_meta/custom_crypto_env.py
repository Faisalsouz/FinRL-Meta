import gymnasium as gym
import numpy as np
from numpy import random as rd


class CryptoTradingEnv(gym.Env):
    """
    A simplified trading environment for crypto pairs that:
      - Removes VIX and turbulence logic
      - Retains price/tech-based observations
      - Allows continuous buy/sell actions

    Compatible with the same RL pipelines, but excludes turbulence & VIX.
    """

    def __init__(
        self,
        config,
        initial_account=1e6,
        gamma=0.99,
        min_stock_rate=0.1,
        max_stock=1e2,
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        reward_scaling=2**-11,
        initial_stocks=None,
    ):
        """
        Parameters
        ----------
        config: dict
            A dictionary that must contain:
              "price_array": shape = (timesteps, number_of_coins)
              "tech_array": shape = (timesteps, feature_dim)  # all technical indicators
              "if_train": bool (True: random initial, False: fixed initial)
        initial_account: float
            Initial capital (e.g. 1e6 = 1,000,000).
        gamma: float
            Discount factor for future rewards in final episodic return.
        min_stock_rate: float
            A fraction of max_stock used to define a threshold for ignoring small actions.
        max_stock: float
            Action scaling factor. Typically how many units (coins) max to buy/sell at once.
        buy_cost_pct: float
            A transaction fee ratio for buys.
        sell_cost_pct: float
            A transaction fee ratio for sells.
        reward_scaling: float
            Scales the raw change in portfolio value into a smaller or larger reward.
        initial_stocks: np.ndarray or None
            If provided, the initial number of coins for each ticker. Else zero.

        Note: We removed any references to turbulence, vix, or thresholds for them.
        """
        price_ary = config["price_array"]
        tech_ary = config["tech_array"]
        if_train = config["if_train"]

        self.price_ary = price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        self.if_train = if_train

        # Scale the technical indicators slightly (optional; adapt to your preference)
        self.tech_ary = self.tech_ary * 2**-7

        self.gamma = gamma
        self.max_stock = max_stock
        self.min_stock_rate = min_stock_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_account

        stock_dim = self.price_ary.shape[1]
        self.initial_stocks = (
            np.zeros(stock_dim, dtype=np.float32)
            if initial_stocks is None
            else initial_stocks
        )

        # --- Internal State ---
        self.day = None
        self.amount = None
        self.stocks = None
        self.stocks_cool_down = None
        self.total_asset = None
        self.gamma_reward = None
        self.initial_total_asset = None

        # --- Env Info ---
        self.env_name = "CryptoTradingEnv"
        # We REMOVE the turbulence dimension here:
        # Old: 1 + 2 + 3*stock_dim + tech_dim
        #     (amount) + (turbulence, turbulence_bool) + (price, stock, cooldown)*stock_dim + tech_dim
        #
        # Now: 1 + 3*stock_dim + tech_dim
        #     (amount) + (price, stock, cooldown)*stock_dim + tech_dim
        self.state_dim = 1 + 3 * stock_dim + self.tech_ary.shape[1]
        self.action_dim = stock_dim

        self.max_step = self.price_ary.shape[0] - 1
        self.if_discrete = False
        self.target_return = 10.0
        self.episode_return = 0.0

        # Gym spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        """Reset the environment to an initial state."""
        self.day = 0
        price = self.price_ary[self.day]

        if self.if_train:
            # Randomize initial holdings
            self.stocks = (
                self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)
            ).astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = (
                self.initial_capital * rd.uniform(0.95, 1.05) 
                - (self.stocks * price).sum()
            )
        else:
            # Fixed initial holdings for testing
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = self.initial_capital

        self.total_asset = self.amount + (self.stocks * price).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        return self.get_state(price), {}

    def step(self, actions):
        """
        actions: np.ndarray of shape (stock_dim,)
        Each action is in [-1, 1], scaled by self.max_stock to get #coins to buy/sell.
        """
        # Scale action
        actions = (actions * self.max_stock).astype(int)

        self.day += 1
        if self.day >= self.max_step:
            # If we exceed data length, treat it as done
            self.day = self.max_step
        price = self.price_ary[self.day]
        self.stocks_cool_down += 1

        min_action = int(self.max_stock * self.min_stock_rate)
        # ---- SELL ----
        for index in np.where(actions < -min_action)[0]:
            if price[index] > 0:  # Sell only if price > 0
                sell_num_shares = min(self.stocks[index], -actions[index])
                self.stocks[index] -= sell_num_shares
                self.amount += price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                self.stocks_cool_down[index] = 0

        # ---- BUY ----
        for index in np.where(actions > min_action)[0]:
            if price[index] > 0:
                buy_num_shares = min(self.amount // price[index], actions[index])
                self.stocks[index] += buy_num_shares
                self.amount -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)
                self.stocks_cool_down[index] = 0

        # Next state
        state = self.get_state(price)
        total_asset = self.amount + (self.stocks * price).sum()
        reward = (total_asset - self.total_asset) * self.reward_scaling
        self.total_asset = total_asset

        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = (self.day == self.max_step)
        if done:
            # Use discounted reward as final reward
            reward = self.gamma_reward
            self.episode_return = total_asset / self.initial_total_asset

        return state, reward, done, False, {}

    def get_state(self, price):
        """
        Returns the current observation as a 1D float32 array of length:
          1 + 3*stock_dim + tech_dim
        Where:
          [0]: scaled amount (cash)
          [1:1+stock_dim]: scaled price
          [1+stock_dim : 1+2*stock_dim]: scaled stocks
          [1+2*stock_dim : 1+3*stock_dim]: stocks_cool_down
          [1+3*stock_dim : end]: technical indicators
        """
        # Example scaling
        amount_scaled = np.array(self.amount * 2**-12, dtype=np.float32)
        price_scaled = price * 2**-6
        stocks_scaled = self.stocks * 2**-6

        return np.hstack((
            amount_scaled,
            price_scaled,
            stocks_scaled,
            self.stocks_cool_down,
            self.tech_ary[self.day]
        )).astype(np.float32)
