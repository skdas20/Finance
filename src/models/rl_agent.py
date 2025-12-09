from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

def train_rl_agent(env, total_timesteps=10000, model_path="models/ppo_trading_agent"):
    """
    Trains a PPO agent on the given environment.
    """
    # Wrap in DummyVecEnv if not already
    # env = DummyVecEnv([lambda: env]) 
    # Actually SB3 handles unwrapped envs usually, but recommended to wrap.
    # For now let's pass env directly, PPO handles it.
    
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model

def load_rl_agent(model_path, env):
    return PPO.load(model_path, env=env)
