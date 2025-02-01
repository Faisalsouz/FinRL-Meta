import gymnasium as gym
import numpy as np
from numpy import random as rd

class CryptoTradingEnv(gym.Env):
    """
    A refactored crypto trading environment that uses technical indicators and a momentum signal
    (derived from historical price patterns) to drive the target allocation. It also uses log returns
    as rewards and optionally includes the momentum signal in the state.
    """

    def __init__(
        self,
        config,
        initial_account=10000,       # Starting capital
        gamma=0.99,
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        reward_scaling=1.0,          # Use 1.0 so that rewards (now log returns) are not squashed
        initial_stocks=None,
        max_position_pct=0.98,       # Maximum fraction of portfolio per asset
        min_trade_amount=20.0,       # Minimum trade value (in $) to trigger a trade
        momentum_window=5,           # Window length (in days) to compute moving averages
        include_momentum_in_state=True  # Optionally include the momentum signal in the state vector
    ):
        """
        Parameters:
          config: dict with keys:
            - "price_array": np.ndarray of shape (timesteps, number_of_assets)
            - "tech_array": np.ndarray of shape (timesteps, tech_feature_dim)
            - "if_train": bool, whether in training mode
        """
        price_ary = config["price_array"]
        tech_ary = config["tech_array"]
        self.if_train = config["if_train"]

        self.price_ary = price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        # (Optional) scale technical indicators
        self.tech_ary = self.tech_ary * 2**-7

        self.gamma = gamma
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling  # With log returns, scaling=1.0 is reasonable
        self.initial_capital = initial_account
        self.max_position_pct = max_position_pct
        self.momentum_window = momentum_window
        self.include_momentum_in_state = include_momentum_in_state

        stock_dim = self.price_ary.shape[1]  # number of assets; typically 1 for a single coin
        self.initial_stocks = (
            np.zeros(stock_dim, dtype=np.float32)
            if initial_stocks is None
            else initial_stocks
        )

        # --- Internal State ---
        self.day = None
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.gamma_reward = None
        self.initial_total_asset = None
        self.min_trade_amount = min_trade_amount

        # Track current position values and ratios
        self.position_values = None
        self.position_ratios = None

        self.env_name = "CryptoTradingEnv"

        # --- State Dimension ---
        # We include:
        #   1) Cash ratio (scaled amount)
        #   2) Current price (normalized)
        #   3) Stocks scaled (as current position value / portfolio)
        #   4) Position ratios (allocation per asset)
        #   5) Technical indicators from tech_ary at the current day
        #   6) Optionally, the momentum signal (1 extra feature)
        tech_dim = self.tech_ary.shape[1]
        base_dim = 1 + 3 * stock_dim + tech_dim
        if self.include_momentum_in_state:
            base_dim += 1
        self.state_dim = base_dim
        self.action_dim = stock_dim

        print(f"Environment state_dim: {self.state_dim}")
        print("State components:")
        print("- Cash ratio (scaled amount): 1")
        print("- Price scaled: ", stock_dim)
        print("- Stocks scaled: ", stock_dim)
        print("- Position ratios: ", stock_dim)
        print("- Technical indicators: ", tech_dim)
        if self.include_momentum_in_state:
            print("- Momentum signal: 1")

        self.max_step = self.price_ary.shape[0] - 1
        self.if_discrete = False
        self.target_return = 10.0
        self.episode_return = 0.0

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        """Reset the environment with proper initialization."""
        self.day = 0
        price = self.price_ary[self.day]

        if self.if_train:
            # For training, start with a small random position
            self.stocks = (self.initial_stocks * rd.uniform(0, 0.1, size=self.initial_stocks.shape)).astype(np.float32)
            self.amount = self.initial_capital * rd.uniform(0.95, 1.0)
        else:
            # For testing, start fully in cash
            self.stocks = np.zeros_like(self.initial_stocks, dtype=np.float32)
            self.amount = self.initial_capital

        self.position_values = self.stocks * price
        self.total_asset = self.amount + self.position_values.sum()
        self.position_ratios = (self.position_values / self.total_asset) if self.total_asset > 0 else np.zeros_like(self.stocks)
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0

        return self.get_state(price), {}

    def _compute_momentum(self):
        """Compute a simple momentum signal based on moving averages.
           Returns a value in [-1, 1]. If there isn’t enough history, return 0."""
        if self.day < self.momentum_window:
            return 0.0
        # Use the last 'momentum_window' days for short-term average
        short_ma = np.mean(self.price_ary[self.day - self.momentum_window + 1 : self.day + 1])
        # Use a longer window for long-term average (e.g., 2 * momentum_window)
        long_window = min(2 * self.momentum_window, self.day + 1)
        long_ma = np.mean(self.price_ary[self.day - long_window + 1 : self.day + 1])
        # Normalize the difference and squash with tanh
        momentum = np.tanh((short_ma - long_ma) / long_ma)
        return momentum

    def calculate_reward(self, old_total_asset, new_total_asset):
        """Calculate reward as a (scaled) log-return."""
        # Use log returns to emphasize percentage changes even if small
        log_return = np.log(new_total_asset / old_total_asset)
        reward = log_return * self.reward_scaling
        return reward, {
            'log_return': log_return,
            'portfolio_value': new_total_asset
        }

    def step(self, actions):
        """
        Execute one time step.
        :param actions: np.ndarray of shape (asset_dim,) with values in [-1,1]
                        representing the network’s output.
        """
        self.day += 1
        if self.day >= self.max_step:
            self.day = self.max_step

        current_price = self.price_ary[self.day]
        actions = np.clip(actions, -1, 1)

        # Compute a momentum signal from historical prices:
        momentum_signal = self._compute_momentum()

        # Combine the network action with the momentum signal.
        # For example, add the momentum signal (possibly with a weight) so that strong trends modify the action.
        weight = 0.5  # Adjust this weight as needed.
        combined_action = np.clip(actions + weight * momentum_signal, -1, 1)

        # Map the (combined) action to a target portfolio percentage in [0, max_position_pct]
        target_position_pcts = (combined_action + 1) * 0.5 * self.max_position_pct

        # Compute current portfolio value
        portfolio_value = self.amount + (self.stocks * current_price).sum()
        # Compute desired position values (in dollars)
        desired_position_values = target_position_pcts * portfolio_value
        current_position_values = self.stocks * current_price
        # The difference tells us how much value to trade for each asset
        position_changes = desired_position_values - current_position_values

        trades_log = []

        # Process SELL orders first (freeing up cash)
        for index in np.where(position_changes < 0)[0]:
            if current_price[index] > 0:
                current_value = current_position_values[index]
                desired_sell_value = abs(position_changes[index])
                if current_value >= self.min_trade_amount:
                    sell_value = min(desired_sell_value, current_value)
                    sell_units = sell_value / current_price[index]
                    self.stocks[index] -= sell_units
                    self.amount += sell_value * (1 - self.sell_cost_pct)
                    trades_log.append({
                        'step': self.day,
                        'asset_index': index,
                        'type': 'SELL',
                        'units': sell_units,
                        'price': current_price[index],
                        'value': sell_value,
                        'fees': sell_value * self.sell_cost_pct,
                        'portfolio_value_before': portfolio_value,
                        'portfolio_value_after': self.amount + (self.stocks * current_price).sum()
                    })

        # Update portfolio value after sells
        current_position_values = self.stocks * current_price
        portfolio_value = self.amount + current_position_values.sum()

        # Process BUY orders
        for index in np.where(position_changes > 0)[0]:
            if current_price[index] > 0:
                available_cash = self.amount * 0.98  # Use most of available cash
                desired_buy_value = position_changes[index]
                buy_value = min(desired_buy_value, available_cash / (1 + self.buy_cost_pct))
                if buy_value >= self.min_trade_amount:
                    buy_units = buy_value / current_price[index]
                    self.stocks[index] += buy_units
                    self.amount -= buy_value * (1 + self.buy_cost_pct)
                    trades_log.append({
                        'step': self.day,
                        'asset_index': index,
                        'type': 'BUY',
                        'units': buy_units,
                        'price': current_price[index],
                        'value': buy_value,
                        'fees': buy_value * self.buy_cost_pct,
                        'portfolio_value_before': portfolio_value,
                        'portfolio_value_after': self.amount + (self.stocks * current_price).sum()
                    })

        # Recompute portfolio values and update position ratios
        self.position_values = self.stocks * current_price
        new_total_asset = self.amount + self.position_values.sum()
        self.position_ratios = (self.position_values / new_total_asset) if new_total_asset > 0 else np.zeros_like(self.stocks)

        # Calculate reward using log-return
        reward, reward_info = self.calculate_reward(self.total_asset, new_total_asset)
        self.total_asset = new_total_asset
        self.gamma_reward = self.gamma_reward * self.gamma + reward

        state = self.get_state(current_price, momentum_signal)
        done = self.day == self.max_step

        info = {
            'portfolio_value': new_total_asset,
            'position_ratios': self.position_ratios,
            'trades': trades_log,
            'reward_info': reward_info
        }

        if done:
            reward = self.gamma_reward
            self.episode_return = new_total_asset / self.initial_total_asset

        return state, reward, done, False, info

    def get_state(self, price, momentum_signal=0.0):
        """
        Build the state vector.
          - Cash ratio: self.amount / initial_capital
          - Price scaled: current price normalized by initial price
          - Stocks scaled: (stocks * current_price) / total_asset
          - Position ratios: current allocation per asset
          - Technical indicators: from tech_ary at the current day
          - (Optional) Momentum signal: computed from historical prices
        """
        amount_scaled = np.array(self.amount / self.initial_capital, dtype=np.float32)
        price_scaled = price / self.price_ary[0]
        stocks_scaled = (self.stocks * price) / self.total_asset if self.total_asset > 0 else np.zeros_like(self.stocks)
        pos_ratios = self.position_ratios
        tech_features = self.tech_ary[self.day]

        state_components = [amount_scaled, price_scaled, stocks_scaled, pos_ratios, tech_features]
        if self.include_momentum_in_state:
            state_components.append(np.array([momentum_signal], dtype=np.float32))
        state = np.hstack(state_components).astype(np.float32)

        # Check the dimension
        expected_dim = 1 + 3 * len(self.stocks) + self.tech_ary.shape[1]
        if self.include_momentum_in_state:
            expected_dim += 1
        assert len(state) == expected_dim, f"State dimension mismatch. Expected {expected_dim}, got {len(state)}"
        return state
