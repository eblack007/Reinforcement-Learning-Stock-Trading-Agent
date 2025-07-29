import pandas as pd
from stable_baselines3 import DQN, PPO
from src.data_handler import fetch_data
from src.environment import TradingEnv

# --- 1. Load Data ---
# I'll use data up to the end of 2022 for training.
# The data from 2023 will be the "out-of-sample" test set.
df = fetch_data(ticker='SPY', start_date='2020-01-01', end_date='2022-12-31')

if df.empty:
    print("Failed to load data. Exiting.")
    exit()

# --- 2. Create the Environment ---
env = TradingEnv(df)

# --- 3. Instantiate the Model ---
# I am using a PPO model with a Multi-Layer Perceptron (MLP) policy.
# 'verbose=1' will print out training progress.
model = PPO("MlpPolicy", env, verbose=1)

# --- 4. Train the Model ---
# We'll train it for 20,000 timesteps. This is a starting point and
# can be increased for better performance.
model.learn(total_timesteps=100000)

# --- 5. Save the Model ---
# The trained model is saved for later evaluation.
model.save("saved_models/ppo_spy_trading")

print("\nTraining complete. Model saved to saved_models/ppo_spy_trading.zip")