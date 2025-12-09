import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, max_steps=None):
        super(StockTradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.current_step = 0
        self.max_steps = len(df) if max_steps is None else max_steps
        
        # Actions of the format Buy x%, Sell x%, Hold, etc.
        # For simplicity: 0: Hold, 1: Buy (All-in), 2: Sell (All-out)
        # Or Discrete(3)
        self.action_space = spaces.Discrete(3)
        
        # State:
        # [Balance, Net Worth, Max Net Worth, Shares Held, Cost Basis, 
        #  Close, SMA_20, SMA_50, RSI, MACD, MACD_Signal, BB_High, BB_Low, Sentiment, ...]
        # We need to normalize these features for RL
        
        # Count numeric columns for observation space
        self.feature_cols = [c for c in df.columns if c not in ['Date', 'Open', 'High', 'Low', 'Volume', 'Adj Close']]
        # We keep Close, and indicators
        # Let's explicitly define features to be safe
        self.feature_cols = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_signal', 'BB_high', 'BB_low', 'Sentiment', 'Returns']
        
        # Add account state features
        self.obs_shape = len(self.feature_cols) + 5 # Balance, NetWorth, Shares, CostBasis, MaxNetWorth
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.current_step = 0
        
        # History for rendering
        self.history = []
        
        return self._next_observation(), {}

    def _next_observation(self):
        # Get current step data
        frame = self.df.iloc[self.current_step]
        
        # Extract market features
        market_features = frame[self.feature_cols].values.astype(np.float32)
        
        # Account features
        account_features = np.array([
            self.balance,
            self.net_worth,
            self.shares_held,
            self.cost_basis,
            self.max_net_worth
        ], dtype=np.float32)
        
        # Normalize/Scale if needed?
        # For now, raw values (StandardScaler should be used in preprocessing pipeline or wrapper)
        # Ideally, we divide balance by initial_balance, price by initial_price, etc.
        # But let's stick to raw for this prototype and rely on SB3's VecNormalize if we use it.
        
        return np.concatenate((account_features, market_features))

    def step(self, action):
        self.current_step += 1
        
        # Get current price
        # Using iloc to get the scalar value, handling MultiIndex if necessary in preprocessing
        current_price = self.df.iloc[self.current_step - 1]['Close']
        if isinstance(current_price, pd.Series):
             current_price = current_price.iloc[0]
        
        # Execute action
        # 0: Hold
        # 1: Buy (All-in)
        # 2: Sell (All-out)
        
        trade_info = ""
        
        if action == 1: # Buy
            if self.balance > current_price:
                shares_to_buy = self.balance // current_price
                cost = shares_to_buy * current_price
                
                # Transaction Cost (0.1% per trade - Realistic for research)
                fee = cost * 0.001 
                
                total_cost = (self.shares_held * self.cost_basis) + cost
                self.shares_held += shares_to_buy
                self.balance -= (cost + fee)
                self.cost_basis = total_cost / self.shares_held if self.shares_held > 0 else 0
                trade_info = f"Bought {shares_to_buy} @ {current_price:.2f}"
                
        elif action == 2: # Sell
            if self.shares_held > 0:
                revenue = self.shares_held * current_price
                
                # Transaction Cost
                fee = revenue * 0.001
                
                self.balance += (revenue - fee)
                trade_info = f"Sold {self.shares_held} @ {current_price:.2f}"
                self.shares_held = 0
                self.cost_basis = 0
        
        # Update Net Worth
        self.net_worth = self.balance + (self.shares_held * current_price)
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        # --- RESEARCH NOVELTY: Risk-Adjusted Reward Shaping ---
        # Instead of simple PnL, we reward stability and punish drawdown.
        
        # 1. Calculate Daily Return
        prev_net_worth = self.history[-1]['net_worth'] if self.history else self.initial_balance
        step_return = (self.net_worth - prev_net_worth) / prev_net_worth
        
        # 2. Risk Penalty (Volatility punishment)
        # If we just hold cash, return is 0. If we trade wildly, volatility is high.
        # We want high return with low volatility.
        reward = step_return
        
        # 3. Drawdown Penalty
        # If current net worth is significantly below max net worth, punish hard.
        drawdown_pct = (self.max_net_worth - self.net_worth) / self.max_net_worth
        if drawdown_pct > 0.1: # 10% drawdown
            reward -= (drawdown_pct * 0.1) # Add penalty
            
        # Scale reward for PPO stability (PPO likes rewards around -1 to 1)
        reward = reward * 100 
        
        done = self.net_worth <= 0 or self.current_step >= len(self.df) - 1
        
        obs = self._next_observation()
        
        info = {
            'net_worth': self.net_worth,
            'action': action,
            'price': current_price,
            'trade': trade_info,
            'drawdown': drawdown_pct
        }
        self.history.append(info)
        
        return obs, reward, done, False, info

    def render(self, mode='human'):
        if self.current_step > 0:
            print(f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Action: {self.history[-1]['action']}")
