import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

# Ensure src is in path
sys.path.append(os.getcwd())

from src.env.trading_env import StockTradingEnv
from src.models.rl_agent import load_rl_agent
from train_pipeline import load_and_process_data, train_lstm_model

st.set_page_config(page_title="AI Trading Agent Dashboard", layout="wide")

st.title("Autonomous AI Trading Agent 📈")
st.markdown("""
This dashboard visualizes the performance of the AI Agent which combines:
- **LSTM**: For price forecasting
- **Sentiment Analysis**: For market mood
- **PPO (RL)**: For decision making
""")

@st.cache_data
def get_data():
    data_path = 'data/sp500_5y.csv'
    if not os.path.exists(data_path):
        st.error("Data file not found. Run download_data.py first.")
        return None
    return load_and_process_data(data_path)

@st.cache_resource
def get_model(_env):
    model_path = "models/ppo_research_agent"
    if not os.path.exists(model_path + ".zip"):
        st.warning(f"Model not found at {model_path}. Please train the model first.")
        return None
    return load_rl_agent(model_path, _env)

def run_simulation(df, initial_balance):
    # Check if LSTM_Prediction is in df, if not add it.
    if 'LSTM_Prediction' not in df.columns:
         st.info("Generating LSTM Forecasts (this might take a moment)...")
         # We assume train_lstm_model returns df with predictions
         # and we suppress output for streamlit
         df, _ = train_lstm_model(df, epochs=5)
         
    env = StockTradingEnv(df, initial_balance=initial_balance)
    
    # We need a dummy env to load the model (PPO needs env to know observation space)
    # But we can also pass env=None if policy is loaded, but better to pass env.
    model = get_model(env)
    
    if model is None:
        return None
    
    obs, _ = env.reset()
    done = False
    
    history = []
    
    # Progress bar
    progress_bar = st.progress(0)
    total_steps = len(df)
    
    step_count = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        history.append(info)
        step_count += 1
        if step_count % 100 == 0:
            progress_bar.progress(min(step_count / total_steps, 1.0))
            
    progress_bar.progress(1.0)
        
    return pd.DataFrame(history)

df = get_data()

st.sidebar.header("Settings")
initial_balance = st.sidebar.number_input("Initial Balance ($)", 1000, 100000, 10000)

if df is not None:
    st.write(f"Data loaded: {len(df)} records. Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    if st.button("Run Backtest"):
        with st.spinner("Simulating Trading..."):
            results = run_simulation(df.copy(), initial_balance)
            
            if results is not None:
                st.success("Simulation Complete!")
                
                # Metrics
                final_nw = results.iloc[-1]['net_worth']
                profit = final_nw - initial_balance
                return_pct = (profit / initial_balance) * 100
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Final Net Worth", f"${final_nw:,.2f}")
                col2.metric("Total Profit", f"${profit:,.2f}", f"{return_pct:.2f}%")
                
                trades = results[results['action'] != 0]
                col3.metric("Total Trades", len(trades))
                
                # Plot Net Worth
                st.subheader("Net Worth Over Time")
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=results['net_worth'], mode='lines', name='Net Worth', line=dict(color='blue')))
                st.plotly_chart(fig, use_container_width=True)
                
                # Plot Price with Buy/Sell
                st.subheader("Trading Actions")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(y=results['price'], mode='lines', name='Price', line=dict(color='gray', width=1)))
                
                # Buys
                buys = results[results['action'] == 1]
                fig2.add_trace(go.Scatter(x=buys.index, y=buys['price'], mode='markers', name='Buy', 
                                        marker=dict(color='green', size=10, symbol='triangle-up')))
                
                # Sells
                sells = results[results['action'] == 2]
                fig2.add_trace(go.Scatter(x=sells.index, y=sells['price'], mode='markers', name='Sell', 
                                        marker=dict(color='red', size=10, symbol='triangle-down')))
                
                st.plotly_chart(fig2, use_container_width=True)
                
                with st.expander("View Transaction Log"):
                    st.dataframe(results)
