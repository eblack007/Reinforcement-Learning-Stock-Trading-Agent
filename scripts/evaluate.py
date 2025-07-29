from stable_baselines3 import DQN, PPO
from src.data_handler import fetch_data
from src.environment import TradingEnv
import numpy as np

# --- 1. Load Test Data ---
# This is the "out-of-sample" data the model has never seen.
df_eval = fetch_data(ticker='SPY', start_date='2023-01-01', end_date='2023-12-31')

if df_eval.empty:
    print("Failed to load evaluation data. Exiting.")
    exit()

# --- 2. Create the Evaluation Environment ---
eval_env = TradingEnv(df_eval)

# --- 3. Load the Trained Model ---
model = PPO.load("saved_models/ppo_spy_trading.zip")

n_episodes = 10
episode_profits = []

# --- 4. Run the Evaluation Loop ---
for episode in range(n_episodes):
    obs, info = eval_env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
    
    # Store the profit from this episode
    profit = eval_env.net_worth - eval_env.initial_balance
    episode_profits.append(profit)
    print(f"Episode {episode + 1}/{n_episodes} - PnL: ${profit:.2f}")


avg_pnl = np.mean(episode_profits)
avg_pnl_percent = (avg_pnl / eval_env.initial_balance) * 100

print("\n--- Average Evaluation Complete ---")
print(f"Ran {n_episodes} episodes.")
print(f"Average Profit/Loss (PnL): ${avg_pnl:.2f} ({avg_pnl_percent:.2f}%)")
print("---------------------------------")