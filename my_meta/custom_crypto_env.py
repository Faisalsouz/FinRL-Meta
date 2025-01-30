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
        initial_account=10000,  # Changed default to 10K
        gamma=0.99,
        min_stock_rate=0.001,  # Reduced for fractional trading
        max_stock=1.0,         # Changed to represent maximum position size as fraction of portfolio
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        reward_scaling=2**-11,
        initial_stocks=None,
        max_position_pct=0.5,  # New: maximum position size as % of portfolio
        min_trade_amount=10.0, # New: minimum USD trade size
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
        self.reward_scaling = 1.0  # Remove scaling initially for debugging
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
        # Position and trade limits
        self.max_position_pct = 0.95  # Allow larger positions
        self.min_trade_amount = 50.0  # Lower minimum trade size
        self.position_values = None
        self.position_ratios = None
        self.debug_mode = False  # Debug mode disabled by default
        self.debug_step_interval = 1000  # Only print debug info every 1000 steps

        # --- Env Info ---
        self.env_name = "CryptoTradingEnv"
        # We REMOVE the turbulence dimension here:
        # Old: 1 + 2 + 3*stock_dim + tech_dim
        #     (amount) + (turbulence, turbulence_bool) + (price, stock, cooldown)*stock_dim + tech_dim
        #
        # Now: 1 + 3*stock_dim + tech_dim
        #     (amount) + (price, stock, cooldown)*stock_dim + tech_dim
        # Calculate state dimension components
        tech_dim = self.tech_ary.shape[1]
        self.state_dim = 1 + 4 * stock_dim + tech_dim  # Updated to include position_ratios
        self.action_dim = stock_dim
        
        # Debug prints
        print(f"Environment state_dim: {self.state_dim}")
        print(f"State components:")
        print(f"- Amount: 1")
        print(f"- Price scaled: {stock_dim}")
        print(f"- Stocks scaled: {stock_dim}")
        print(f"- Stocks cooldown: {stock_dim}")
        print(f"- Position ratios: {stock_dim}")
        print(f"- Technical indicators: {tech_dim}")

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

    def reset(self, *, seed=None, options=None):
        """Enhanced reset with better initial state handling"""
        self.day = 0
        price = self.price_ary[self.day]

        if self.if_train:
            # Initialize with small random positions for training
            self.stocks = (self.initial_stocks * rd.uniform(0, 0.1, size=self.initial_stocks.shape)).astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = self.initial_capital * rd.uniform(0.95, 1.0)
        else:
            # Start with cash for testing
            self.stocks = np.zeros_like(self.initial_stocks, dtype=np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = self.initial_capital

        # Initialize position tracking
        self.position_values = self.stocks * price
        self.total_asset = self.amount + self.position_values.sum()
        self.position_ratios = self.position_values / self.total_asset if self.total_asset > 0 else np.zeros_like(self.stocks)
        
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0

        return self.get_state(price), {}

    def calculate_reward(self, old_total_asset, new_total_asset):
        """Calculate reward based on portfolio growth with appropriate scaling"""
        returns = (new_total_asset - old_total_asset) / old_total_asset
        
        # Scale returns back to smaller values for better learning
        reward = returns * self.reward_scaling  # Remove the *100.0 multiplier
        
        return reward, {
            'returns_pct': returns * 100,
            'reward': reward,
            'portfolio_value': new_total_asset
        }

    def step(self, actions):
        """
        Enhanced step function with better position sizing and fractional trading
        actions: np.ndarray of shape (stock_dim,)
        Each action in [-1, 1] represents target position changes as % of portfolio
        """
        self.day += 1
        if self.day >= self.max_step:
            self.day = self.max_step

        current_price = self.price_ary[self.day]
        
        # Simplify position changes calculation
        actions = np.clip(actions, -1, 1)
        desired_position_values = (actions + 1) * 0.5 * self.total_asset  # Convert to [0,1] range
        current_position_values = self.stocks * current_price
        position_changes = desired_position_values - current_position_values
        
        # Calculate current portfolio value
        portfolio_value = self.amount + self.position_values.sum()
        
        # Controlled debug printing
        if self.debug_mode and (self.day % self.debug_step_interval == 0):
            print(f"\n=== Debug Info for Step {self.day} ===")
            print(f"Portfolio value: ${portfolio_value:.2f}")
            print(f"Current positions: {self.position_values / portfolio_value}")
            print(f"Desired positions: {desired_position_values / portfolio_value}")
            print(f"Position changes: {position_changes}")
            print("=" * 40)
        
        # Add trade logging
        trades_log = []
        
        # Process sells first (to free up cash)
        for index in np.where(position_changes < 0)[0]:
            if current_price[index] > 0:  # Valid price check
                current_position_value = self.position_values[index]
                desired_sell_value = abs(position_changes[index])
                
                if current_position_value >= self.min_trade_amount:
                    sell_value = min(desired_sell_value, current_position_value)
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
        self.position_values = self.stocks * current_price
        portfolio_value = self.amount + self.position_values.sum()

        # Process buys
        for index in np.where(position_changes > 0)[0]:
            if current_price[index] > 0:  # Valid price check
                max_position_value = portfolio_value * self.max_position_pct
                current_position_value = self.position_values[index]
                available_position_value = max_position_value - current_position_value
                available_cash = self.amount / (1 + self.buy_cost_pct)
                
                buy_value = min(
                    position_changes[index],
                    available_position_value,
                    available_cash
                )
                
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

        # Calculate returns and reward
        self.position_values = self.stocks * current_price
        new_total_asset = self.amount + sum(self.position_values)
        
        # Calculate reward using dedicated method
        reward, reward_info = self.calculate_reward(self.total_asset, new_total_asset)
        
        self.total_asset = new_total_asset
        self.gamma_reward = self.gamma_reward * self.gamma + reward

        # State, done, and info
        state = self.get_state(current_price)
        done = self.day == self.max_step
        
        info = {
            'portfolio_value': new_total_asset,
            'position_ratios': self.position_ratios,
            'cash_ratio': self.amount / new_total_asset,
            'trades': trades_log,
            'reward_info': reward_info
        }

        if done:
            reward = self.gamma_reward
            self.episode_return = new_total_asset / self.initial_total_asset

        return state, reward, done, False, info

    def _calculate_volatility(self):
        """Calculate rolling volatility for each asset"""
        if self.day < 10:  # Need some history
            return np.zeros(self.action_dim)
        
        lookback = 10
        price_window = self.price_ary[self.day-lookback:self.day+1]
        returns = np.diff(price_window, axis=0) / price_window[:-1]
        volatility = np.std(returns, axis=0)
        return volatility

    def get_state(self, price):
        """Enhanced state representation with better normalization"""
        amount_scaled = np.array(self.amount / self.initial_capital, dtype=np.float32)
        price_scaled = price / self.price_ary[0]  # Normalize to initial price
        stocks_scaled = self.stocks * price / self.total_asset  # Position sizes as ratios
        
        # Add price changes
        price_change = np.zeros_like(price)
        if self.day > 0:
            price_change = (price - self.price_ary[self.day-1]) / self.price_ary[self.day-1]
            
        # Calculate volatility
        volatility = self._calculate_volatility()

        state = np.hstack((
            amount_scaled,           # 1
            price_scaled,           # action_dim
            stocks_scaled,          # action_dim
            price_change,           # action_dim
            volatility,             # action_dim
            self.tech_ary[self.day] # tech_indicators * action_dim
        )).astype(np.float32)
        
        # Add dimension check
        expected_dim = 1 + 4 * len(self.stocks) + self.tech_ary.shape[1]
        assert len(state) == expected_dim, f"State dimension mismatch. Expected {expected_dim}, got {len(state)}"
        
        return state
