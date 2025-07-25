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

        # Define action space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = gym.spaces.Discrete(3)

        # Define observation space (prices + RSI for look_back_window days + holdings + balance)
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(self.look_back_window * 2 + 2,), dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.current_step = np.random.randint(
            self.look_back_window, len(self.df) - 1
        )

        return self._next_observation(), {}

    def _next_observation(self):
        # Get the price and RSI data for the look-back window
        frame = self.df.iloc[
            self.current_step - self.look_back_window : self.current_step
        ][['Close', 'RSI']].values

        # Flatten the frame and append current holdings and balance
        obs = np.append(frame.flatten(), [self.shares_held, self.balance])
        return obs.astype(np.float32)

    def step(self, action):
        action = action.item()
        
        # This is the correct, future-proof way to get the current price
        current_price = self.df['Close'].iloc[:, 0].iloc[self.current_step]
        
        trade_executed = False
        
        # Execute the action
        if action == 1:  # Buy
            cost = self.shares_to_trade * current_price
            if self.balance > cost:
                self.shares_held += self.shares_to_trade
                self.balance -= cost
                trade_executed = True

        elif action == 2:  # Sell
            # CORRECTED: Use >= to allow selling all shares
            if self.shares_held >= self.shares_to_trade:
                self.balance += self.shares_to_trade * current_price
                self.shares_held -= self.shares_to_trade
                trade_executed = True

        # Calculate the net worth after the potential trade
        new_net_worth = self.balance + self.shares_held * current_price
        
        # The base reward is the change in net worth
        reward = new_net_worth - self.net_worth
        self.net_worth = new_net_worth

        # Apply penalties for ineffective actions
        if not trade_executed:
            if action == 1 or action == 2:
                reward -= 10  # Penalize trying to buy/sell but failing
            elif action == 0 and self.shares_held == 0:
                reward -= 1 # Small penalty for holding cash

        # Move to the next time step
        self.current_step += 1
        
        # Check if the episode is done
        done = self.net_worth <= 0 or self.current_step >= len(self.df) - 1
        obs = self._next_observation()
        
        terminated = done
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Shares: {self.shares_held}, Balance: {self.balance:.2f}')