import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

from src.feature_engineering import add_technical_indicators, add_sentiment_score
from src.models.lstm import LSTMModel, TimeSeriesDataset, create_sequences
from src.env.trading_env import StockTradingEnv
from src.models.rl_agent import train_rl_agent # I need to create this file first!

# Fix for src import if needed, assuming running from root
import sys
sys.path.append('.')

def load_and_process_data(filepath):
    print(f"Loading data from {filepath}...")
    try:
        # Try reading with multi-header (Price, Ticker)
        df = pd.read_csv(filepath, header=[0, 1], index_col=0, parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    except:
        # Fallback to standard reading if single header
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # Feature Engineering
    print("Generating technical indicators...")
    df = add_technical_indicators(df)
    
    # Sentiment
    print("Adding sentiment scores...")
    df = add_sentiment_score(df)
    
    return df

def train_lstm_model(df, target_col='Close', seq_length=60, epochs=10):
    print("\nPreparing data for LSTM...")
    data = df.values
    
    # Scaling
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    target_idx = df.columns.get_loc(target_col)
    
    X, y = create_sequences(data_scaled, seq_length, target_idx)
    
    # Split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Dataset
    train_dataset = TimeSeriesDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Model
    input_dim = X.shape[2]
    model = LSTMModel(input_dim, hidden_dim=64, num_layers=2, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Training LSTM...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
            
    # Predictions
    print("Generating LSTM predictions...")
    model.eval()
    with torch.no_grad():
        all_X_tensor = torch.tensor(X, dtype=torch.float32)
        predictions_scaled = model(all_X_tensor).numpy()
    
    # Inverse transform predictions
    # We normalized all features, so we need to inverse transform carefully.
    # The scaler worked on (samples, features). Predictions are (samples, 1).
    # We need to construct a dummy array to inverse transform.
    dummy = np.zeros((len(predictions_scaled), data.shape[1]))
    dummy[:, target_idx] = predictions_scaled.flatten()
    predictions_original = scaler.inverse_transform(dummy)[:, target_idx]
    
    # Add predictions to dataframe
    # Note: LSTM consumes seq_length rows to produce 1 prediction.
    # So the first seq_length rows have no prediction.
    # The prediction at index i (in X) corresponds to data[i+seq_length].
    # So we should align them.
    
    # Create a Series with NaNs for the first seq_length rows
    pred_series = pd.Series(np.nan, index=df.index)
    
    # The X[0] predicts y[0] which corresponds to df.iloc[seq_length]
    # So predictions start at seq_length
    start_idx = seq_length
    pred_series.iloc[start_idx:start_idx+len(predictions_original)] = predictions_original
    
    df['LSTM_Prediction'] = pred_series
    
    # Drop rows without prediction (the first seq_length rows) for RL training
    df.dropna(inplace=True)
    
    return df, model

def main():
    # 1. Load Data (Research Mode: Multi-Stock)
    data_dir = 'data'
    if not os.path.exists(data_dir):
        print("Data directory not found.")
        return

    files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'benchmark' not in f]
    if not files:
        print("No data files found. Run download_data.py")
        return

    print(f"Found {len(files)} datasets. Splitting into Train/Test for Generalization Research.")
    
    # Split: Train on 80% of stocks, Test on 20% unseen stocks
    # This proves the model learns "Trading" not "Memorizing AAPL"
    train_files = files[:int(len(files)*0.8)]
    test_files = files[int(len(files)*0.8):]
    
    print(f"Training on: {train_files}")
    print(f"Testing on: {test_files}")

    # Helper to process a list of files
    def process_files(file_list):
        processed_dfs = []
        for f in file_list:
            path = os.path.join(data_dir, f)
            try:
                df = load_and_process_data(path)
                # Add LSTM predictions (Simulate "Forecast" signal)
                # Ideally we train one LSTM per stock or one giant LSTM. 
                # For simplicity here, we train a quick LSTM per stock to generate the 'LSTM_Prediction' feature.
                df, _ = train_lstm_model(df, epochs=2) # Quick train for feature generation
                processed_dfs.append(df)
            except Exception as e:
                print(f"Skipping {f}: {e}")
        return processed_dfs

    print("\n--- Phase 1: Preprocessing & Feature Engineering ---")
    train_dfs = process_files(train_files)
    
    if not train_dfs:
        print("No valid training data.")
        return

    # 3. Train RL Agent (Vectorized for Generalization)
    print("\n--- Phase 2: Training RL Agent (PPO) ---")
    from stable_baselines3.common.vec_env import DummyVecEnv
    from src.models.rl_agent import train_rl_agent
    
    # Create a vectorized environment that cycles through different stocks
    # In a real paper, we might use SubprocVecEnv for parallel training
    
    # We concatenate all training data into one giant timeline for the 'Dummy' env
    # Or better: Create a custom VecEnv. 
    # For this script, we will train on the first stock, then continue training on the second, etc.
    # (Curriculum Learning approach)
    
    model_path = "models/ppo_research_agent"
    model = None
    
    total_timesteps_per_stock = 10000 # Increase for real research
    
    for i, df in enumerate(train_dfs):
        print(f"\nTraining iteration {i+1}/{len(train_dfs)}...")
        env = StockTradingEnv(df)
        
        if model is None:
            # First initialization
            from stable_baselines3 import PPO
            model = PPO('MlpPolicy', env, verbose=1, ent_coef=0.01) # ent_coef encourage exploration
        else:
            # Swap environment and continue learning
            model.set_env(env)
            
        model.learn(total_timesteps=total_timesteps_per_stock, reset_num_timesteps=False)
    
    model.save(model_path)
    print(f"Research Agent saved to {model_path}")
    
    # 4. Out-of-Sample Testing
    print("\n--- Phase 3: Out-of-Sample Generalization Test ---")
    # Evaluate on the UNSEEN test files
    test_dfs = process_files(test_files)
    
    for i, df in enumerate(test_dfs):
        print(f"\nTesting on unseen stock: {test_files[i]}")
        env = StockTradingEnv(df)
        
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, info = env.step(action)
            
        print(f"Final Profit on {test_files[i]}: ${info['net_worth'] - env.initial_balance:.2f}")

if __name__ == "__main__":
    main()
