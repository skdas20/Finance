import numpy as np
import pandas as pd
import torch

def explain_decision(model, env, obs):
    """
    Explain the agent's decision using Feature Importance Perturbation.
    
    Args:
        model: Trained PPO model
        env: Trading environment
        obs: Current observation
        
    Returns:
        dict: Feature importance scores
    """
    # 1. Get original action probability distribution
    # PPO's actor network outputs logits/action distribution
    # We want to see how much the action probability changes when we perturb a feature.
    
    # Convert obs to tensor if needed
    obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)
    
    # Get original action (deterministic)
    original_action, _ = model.predict(obs, deterministic=True)
    
    # Get original value estimation (critic)
    with torch.no_grad():
        original_value = model.policy.predict_values(obs_tensor)
        
    feature_names = env.feature_cols # e.g. ['Close', 'SMA_20', 'RSI', ...]
    # Account features are first, then market features.
    # We focus on market features for explainability.
    
    account_features_len = 5 # Balance, NetWorth, Shares, CostBasis, MaxNetWorth
    market_features_start_idx = account_features_len
    
    importances = {}
    
    # Perturbation Analysis
    # We slightly modify each feature and check the change in Value Function (Critic)
    # Why Critic? Because it represents the "expected future reward" of the state.
    # If changing RSI drops the Value drastically, RSI is crucial.
    
    base_obs = obs.copy()
    
    for i, feature_name in enumerate(feature_names):
        idx = market_features_start_idx + i
        if idx >= len(base_obs): break
        
        # Perturb feature by +10% standard deviation (or simply +1% value)
        # Since our data is not standardized in the Env (we assumed raw), let's add 1% relative.
        perturbed_obs = base_obs.copy()
        if perturbed_obs[idx] != 0:
            perturbed_obs[idx] *= 1.05 # 5% perturbation
        else:
            perturbed_obs[idx] = 0.01
            
        perturbed_tensor = torch.as_tensor(perturbed_obs).float().unsqueeze(0)
        
        with torch.no_grad():
            perturbed_value = model.policy.predict_values(perturbed_tensor)
            
        # Importance = Abs(Change in Value)
        impact = float(abs(perturbed_value - original_value))
        importances[feature_name] = impact
        
    # Normalize importance
    total_impact = sum(importances.values())
    if total_impact > 0:
        for k in importances:
            importances[k] /= total_impact
            
    # Sort
    sorted_importance = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_importance, original_action
