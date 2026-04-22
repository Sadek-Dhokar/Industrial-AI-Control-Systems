"""
Smart Grid Energy Management System
RL Agent Training Script

Trains a PPO Agent on the SmartGridEnv using Stable-Baselines3.

Author: Sadek Dhokar
Date: April 2026
"""

import pandas as pd
import numpy as np
import os
import joblib

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Import our custom environment class
from src.env import SmartGridEnv

def train_agent():
    print("Loading training data for RL...")
    train_df = pd.read_csv("data/smart_grid_train.csv")
    
    print("Creating SmartGrid Environment...")
    # Initialize the base environment
    env = SmartGridEnv(train_df)
    
    # Optional: check the environment with SB3's checker
    try:
        check_env(env, warn=True)
        print("Environment check passed!")
    except Exception as e:
        print(f"Environment check failed: {e}")
        return

    # Wrap the environment
    vec_env = DummyVecEnv([lambda: env])
    
    print("\n--- Initializing PPO Agent ---")
    # PPO hyperparams are relatively standard
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1, 
        learning_rate=0.0003, 
        n_steps=2048, 
        batch_size=64, 
        n_epochs=10, 
        gamma=0.99, 
        gae_lambda=0.95, 
        clip_range=0.2, 
        ent_coef=0.01,
        tensorboard_log="./logs/"
    )
    
    print("\n--- Training PPO Agent (this may take a few minutes) ---")
    # Train for a few thousand timesteps to demonstrate learning capabilities
    total_timesteps = 100000 
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    print("\n--- Saving the Model ---")
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_smart_grid_agent")
    
    print("Training finished successfully.")

if __name__ == "__main__":
    train_agent()
