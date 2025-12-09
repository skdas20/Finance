# Autonomous AI Trading Agent 📈

A comprehensive AI-powered stock trading system that combines Deep Learning (LSTM), Reinforcement Learning (PPO), and Sentiment Analysis.

## 🚀 Features

- **Deep Learning Forecast**: LSTM model trains on historical price data to predict future trends.
- **Reinforcement Learning**: PPO agent (Stable Baselines 3) learns optimal trading strategies (Buy/Sell/Hold) based on market state and account status.
- **Sentiment Analysis**: Incorporates synthetic sentiment scores (placeholder for real-time news analysis).
- **Technical Indicators**: Uses SMA, RSI, MACD, Bollinger Bands via `ta` library.
- **Interactive Dashboard**: Streamlit-based dashboard to visualize backtest performance and trades.

## 🛠️ Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Data**:
    ```bash
    python download_data.py
    ```
    Downloads S&P 500 and other stock data to `data/`.

## 🏃‍♂️ Usage

### 1. Train the Models
Run the full training pipeline (LSTM + RL):
```bash
python train_pipeline.py
```
This will:
- Load and preprocess data.
- Train the LSTM model.
- Train the PPO agent.
- Save the trained agent to `models/ppo_trading_agent`.

### 2. Run the Dashboard
Visualize the results and run backtests:
```bash
streamlit run dashboard.py
```
Open the provided URL in your browser.

## 📂 Project Structure

- `src/`: Source code modules.
    - `env/`: Custom Gymnasium trading environment.
    - `models/`: LSTM and RL agent definitions.
    - `feature_engineering.py`: Technical indicators and sentiment logic.
- `data/`: Storage for CSV data.
- `models/`: Saved model artifacts.
- `train_pipeline.py`: Main script for training.
- `dashboard.py`: Streamlit application.
- `download_data.py`: Data fetching script.

## 📝 Notes
- The current sentiment score is synthetic. To use real data, update `src/feature_engineering.py` to fetch from an API (e.g., Twitter/X, NewsAPI).
- The LSTM model predicts the *next day's* price, which is fed as a feature to the RL agent.
