from stable_baselines3 import DQN, PPO
from src.data_handler import fetch_data
from src.environment import TradingEnv

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

# --- 4. Run the Evaluation Loop ---
obs, info = eval_env.reset()
done = False
while not done:
    # Use deterministic=True for evaluation to get the best action
    action, _states = model.predict(obs, deterministic=True)
    
    # Take a step in the environment
    obs, reward, terminated, truncated, info = eval_env.step(action)
    
    # The 'done' flag is a combination of terminated and truncated
    done = terminated or truncated
    
    # Optional: Render the environment to see progress
    eval_env.render()

print("\n--- Evaluation Complete ---")
print(f"Initial Balance: ${eval_env.initial_balance:.2f}")
print(f"Final Net Worth: ${eval_env.net_worth:.2f}")

pnl = eval_env.net_worth - eval_env.initial_balance
pnl_percent = (pnl / eval_env.initial_balance) * 100
print(f"Profit/Loss (PnL): ${pnl:.2f} ({pnl_percent:.2f}%)")
print("--------------------------")