import gymnasium as gym
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """A custom stock trading environment for reinforcement learning."""
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df
        self.initial_balance = 10000
        self.look_back_window = 5
        self.shares_to_trade = 10

        self.action_space = gym.spaces.Discrete(3)

        # Observation space: 5 features (close, rsi, macd, macd_signal, macd_hist)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.look_back_window * 5 + 2,), dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.current_step = np.random.randint(
            self.look_back_window, len(self.df) - 1
        )
        self.action = None  # To store the last action
        return self._next_observation(), {}

    def _next_observation(self):
        # Use the correct lowercase column names
        frame = self.df.iloc[
            self.current_step - self.look_back_window : self.current_step
        ][['close', 'rsi', 'macd', 'macd_signal', 'macd_hist']].values

        obs = np.append(frame.flatten(), [self.shares_held, self.balance])
        return obs.astype(np.float32)

    def step(self, action):
        action = action.item()
        self.action = action  # Store the action for rendering
        current_price = self.df['close'].iloc[self.current_step]
        
        cost = self.shares_to_trade * current_price
        if action == 1 and self.balance > cost:
            self.shares_held += self.shares_to_trade
            self.balance -= cost
        elif action == 2 and self.shares_held >= self.shares_to_trade:
            self.shares_held -= self.shares_to_trade
            self.balance += self.shares_to_trade * current_price

        new_net_worth = self.balance + self.shares_held * current_price
        reward = new_net_worth - self.net_worth
        self.net_worth = new_net_worth

        # Penalty for holding cash
        if action == 0 and self.shares_held == 0:
            reward -= 2

        self.current_step += 1
        done = self.net_worth <= 0 or self.current_step >= len(self.df) - 1
        obs = self._next_observation()
        
        return obs, reward, done, False, {}

    def render(self):
        action_map = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
        action_str = action_map.get(self.action, 'N/A')
        print(f'Step: {self.current_step}, Action: {action_str}, Net Worth: {self.net_worth:.2f}, Shares: {self.shares_held}, Balance: {self.balance:.2f}')