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
        self.look_back_window = 5 # Number of previous days to use as state

        # Define action space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = gym.spaces.Discrete(3)

        # Define observation space (prices for look_back_window days + RSI + holdings + balance)
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(self.look_back_window * 2 + 2,), dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        # Start at a random point in the data, ensuring enough previous data exists
        self.current_step = np.random.randint(
            self.look_back_window, len(self.df) - 1
        )

        return self._next_observation(), {}

    def _next_observation(self):
        # Get the price data for the look-back window
        # Get the price and RSI data for the look-back window
        frame = self.df.iloc[
            self.current_step - self.look_back_window : self.current_step
        ][['Close', 'RSI']].values

        # Flatten the frame and append current holdings and balance
        obs = np.append(frame.flatten(), [self.shares_held, self.balance])
        return obs.astype(np.float32)


    def step(self, action):
        
        action = action.item()

        # Get current price
        current_price = self.df['Close'].iloc[:, 0].iloc[self.current_step]
        
        shares_to_trade = 10
        cost = current_price * shares_to_trade
        print(f"Step: {self.current_step} | Action: {action} | Price: {current_price:.2f} | Balance: {self.balance:.2f} | Cost: {cost:.2f} | Shares: {self.shares_held}")
        # Execute the action
        if action == 1: # Buy
            # Buy multiple shares
            if self.balance > cost:
                self.shares_held += shares_to_trade
                self.balance -= cost
        elif action == 2: # Sell
            # Sell multiple shares
            if self.shares_held > shares_to_trade:
                self.shares_held -= shares_to_trade
                self.balance += shares_to_trade * current_price

        # New reward logic with penalties
        # Start with the default reward, which is the change in net worth
        new_net_worth = self.balance + self.shares_held * current_price
        reward = new_net_worth - self.net_worth
        self.net_worth = new_net_worth

        # Now, add penalties for invalid actions
        if action == 0 and self.shares_held == 0:
            reward -= 1  # Penalize for doing nothing when uninvested

        if action == 2 and self.shares_held < shares_to_trade:
            reward -= 10  # Heavily penalize trying to sell shares you don't hav
                
        # Move to the next time step
        self.current_step += 1
        
        # Check if the episode is done
        done = self.net_worth <= 0 or self.current_step >= len(self.df) - 1
        
        # Get the next observation
        obs = self._next_observation()
        
        # For compatibility with Gymnasium, step returns 5 values
        terminated = done
        truncated = False # We don't have a time limit that truncates the episode
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        # Optional: For visualizing the agent's performance
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Shares: {self.shares_held}, Balance: {self.balance:.2f}')