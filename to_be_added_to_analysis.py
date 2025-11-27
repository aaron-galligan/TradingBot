
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta

# ==========================================
# Alpha Vantage Intraday Analysis
# ==========================================

# TODO: Replace with your actual Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = "YOUR_API_KEY_HERE"


"""
To use this do the following: 
# Assuming df_by_ticker, X_test, and y_pred are already defined in your notebook
results_df = analyze_intraday_strategies(
    df_metadata=df_by_ticker.loc[X_test.index], 
    predictions=y_pred,
    api_key="YOUR_ACTUAL_KEY" # Optional if set globally
)

"""





def get_intraday_data_av(ticker, target_date, interval='15min', api_key=ALPHA_VANTAGE_API_KEY):
    """
    Fetches intraday data for a specific ticker and date from Alpha Vantage.
    
    Args:
        ticker (str): Stock symbol.
        target_date (datetime or str): The date to fetch data for.
        interval (str): Candle interval ('1min', '5min', '15min', '30min', '60min').
        api_key (str): Alpha Vantage API key.
        
    Returns:
        pd.DataFrame: DataFrame with intraday data for the target date, or None if not found/error.
    """
    # Ensure target_date is a date object
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date).date()
    elif isinstance(target_date, pd.Timestamp):
        target_date = target_date.date()
        
    # Alpha Vantage API URL
    # outputsize='full' returns the last 1-2 months of intraday data.
    # For older data, TIME_SERIES_INTRADAY_EXTENDED is needed (requires premium or complex slicing).
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval={interval}&outputsize=full&apikey={api_key}&datatype=csv"
    
    try:
        # Fetch data
        df = pd.read_csv(url)
        
        # Check for API error messages in the response
        if 'timestamp' not in df.columns:
            # Common errors: Invalid API call, rate limit, etc.
            if 'Error Message' in df.columns or 'Information' in df.columns:
                # Uncomment to debug
                # print(f"API Message for {ticker}: {df.iloc[0,0]}")
                pass
            return None
            
        # Process DataFrame
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Filter for the specific date
        # We look for data on the target_date
        day_data = df[df.index.date == target_date]
        
        if day_data.empty:
            # print(f"No data found for {ticker} on {target_date}. Data might be too old for standard endpoint.")
            return None
            
        return day_data
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def run_stop_loss_simulation(day_data, stop_loss_pct=0.03, trailing_stop_pct=0.03):
    """
    Simulates two strategies on intraday data:
    1. Fixed Stop Loss
    2. Fixed Stop Loss + Trailing Stop
    
    Args:
        day_data (pd.DataFrame): Intraday data (must contain 'open', 'high', 'low', 'close').
        stop_loss_pct (float): Percentage for fixed stop loss (e.g., 0.03 for 3%).
        trailing_stop_pct (float): Percentage for trailing stop (e.g., 0.03 for 3%).
        
    Returns:
        dict: Returns for both strategies.
    """
    if day_data is None or day_data.empty:
        return None

    # Entry assumptions: Enter at the Open of the first candle available for the day
    entry_price = day_data.iloc[0]['open']
    
    # --- Strategy 1: Fixed Stop Loss ---
    fixed_stop_price = entry_price * (1 - stop_loss_pct)
    exit_price_fixed = day_data.iloc[-1]['close'] # Default exit at EOD close
    
    for idx, row in day_data.iterrows():
        if row['low'] <= fixed_stop_price:
            exit_price_fixed = fixed_stop_price # Stopped out
            break
            
    return_fixed = (exit_price_fixed - entry_price) / entry_price
    
    # --- Strategy 2: Fixed Stop Loss + Trailing Stop ---
    # Logic: We have a fixed stop (safety net) AND a trailing stop.
    # Usually, trailing stop activates immediately or after some profit.
    # Here we assume standard trailing stop: High Watermark * (1 - pct)
    # But we also respect the initial fixed stop if it's tighter? 
    # Usually trailing stop starts at (Entry * (1-pct)), which is same as fixed stop.
    # So we just track the trailing stop level.
    
    high_watermark = entry_price
    trailing_stop_level = entry_price * (1 - trailing_stop_pct)
    exit_price_trailing = day_data.iloc[-1]['close'] # Default exit at EOD close
    
    for idx, row in day_data.iterrows():
        # Check if stopped out first (using Low)
        if row['low'] <= trailing_stop_level:
            exit_price_trailing = trailing_stop_level
            break
            
        # Update High Watermark and Trailing Stop (using High)
        if row['high'] > high_watermark:
            high_watermark = row['high']
            new_stop = high_watermark * (1 - trailing_stop_pct)
            # Trailing stop only moves up
            if new_stop > trailing_stop_level:
                trailing_stop_level = new_stop
                
    return_trailing = (exit_price_trailing - entry_price) / entry_price
    
    return {
        'return_fixed_sl': return_fixed,
        'return_trailing_sl': return_trailing,
        'entry_price': entry_price
    }

def analyze_intraday_strategies(df_metadata, predictions, api_key=ALPHA_VANTAGE_API_KEY):
    """
    Main function to run the analysis on predicted trades.
    
    Args:
        df_metadata (pd.DataFrame): DataFrame containing 'ticker' and 'eff_trans_date'.
                                    Should correspond to the test set.
        predictions (array-like): Predicted labels (1 for Buy, 0 for Ignore).
        api_key (str): Alpha Vantage API key.
        
    Returns:
        pd.DataFrame: Results of the simulation.
    """
    # Combine metadata with predictions
    analysis_df = df_metadata.copy()
    analysis_df['prediction'] = predictions
    
    # Filter for Buy signals
    buy_signals = analysis_df[analysis_df['prediction'] == 1]
    
    print(f"Analyzing {len(buy_signals)} buy signals with intraday data...")
    
    results = []
    
    # Counter for rate limiting
    request_count = 0
    start_time = time.time()
    
    for idx, row in buy_signals.iterrows():
        ticker = row['ticker']
        date = row['eff_trans_date']
        
        # Rate Limiting (Free Tier: 5 calls/min)
        request_count += 1
        if request_count % 5 == 0:
            elapsed = time.time() - start_time
            if elapsed < 60:
                sleep_time = 60 - elapsed + 2 # Add buffer
                print(f"Rate limit pause... sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                start_time = time.time()
        
        # Fetch Data
        # Note: We assume we want to trade ON the effective transaction date
        # Adjust 'date' if you intend to trade the NEXT day
        day_data = get_intraday_data_av(ticker, date, interval='15min', api_key=api_key)
        
        if day_data is not None:
            sim_res = run_stop_loss_simulation(day_data)
            if sim_res:
                res_entry = {
                    'ticker': ticker,
                    'date': date,
                    'return_fixed_sl': sim_res['return_fixed_sl'],
                    'return_trailing_sl': sim_res['return_trailing_sl']
                }
                results.append(res_entry)
                print(f"Processed {ticker}: Fixed={sim_res['return_fixed_sl']:.2%}, Trailing={sim_res['return_trailing_sl']:.2%}")
        else:
            # print(f"Skipping {ticker} (No data)")
            pass
            
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        print("\n--- Strategy Comparison ---")
        print(f"Average Return (Fixed 3% SL): {results_df['return_fixed_sl'].mean():.4%}")
        print(f"Average Return (Trailing 3% SL): {results_df['return_trailing_sl'].mean():.4%}")
        
        # Win Rate comparison (Positive returns)
        win_rate_fixed = (results_df['return_fixed_sl'] > 0).mean()
        win_rate_trailing = (results_df['return_trailing_sl'] > 0).mean()
        print(f"Win Rate (Fixed): {win_rate_fixed:.2%}")
        print(f"Win Rate (Trailing): {win_rate_trailing:.2%}")
        
    return results_df

# Example Usage:
# results = analyze_intraday_strategies(df_by_ticker.loc[X_test.index], y_pred)
















# ==========================================
# Probability-Weighted Trading Strategy
# ==========================================


def run_probability_weighted_backtest(df_by_ticker, X_test, y_pred, y_pred_proba,
                                      starting_balance=10000,
                                      daily_total_risk=0.33,
                                      max_concurrent_positions=20,
                                      stop_loss=-0.03,
                                      take_profit=10,
                                      spread_pct=0.001,
                                      slippage_pct=0.001,
                                      start_date="2024-01-01 00:00:00",
                                      check_stopout=True):
    """
    Runs a backtest with probability-weighted capital allocation.
    
    Args:
        df_by_ticker: Main dataframe with trade data
        X_test: Test features
        y_pred: Binary predictions (0 or 1)
        y_pred_proba: Predicted probabilities
        starting_balance: Initial capital
        daily_total_risk: Fraction of account deployed across all trades per day
        max_concurrent_positions: Maximum number of positions per day
        stop_loss: Stop loss threshold (e.g., -0.03 for -3%)
        take_profit: Take profit cap
        spread_pct: Round-trip spread assumption
        slippage_pct: Execution slippage
        start_date: Filter trades after this date
        check_stopout: Whether to check for intraday stop-outs
        
    Returns:
        tuple: (trades DataFrame, summary DataFrame)
    """
    
    # Prepare data
    X_test_temp = X_test.copy()
    X_test_temp['probability'] = y_pred_proba
    
    predicted_trues = X_test_temp[y_pred == 1]
    pred_trades = df_by_ticker.loc[predicted_trues.index]
    pred_trades = pred_trades[pred_trades["transaction_date"] > start_date]
    pred_trades = pred_trades.drop_duplicates(subset=['eff_trans_date', 'ticker'], keep='first')
    
    # Build trades dataframe
    trades = pd.DataFrame(index=pred_trades.index)
    trades['probability'] = trades.index.map(X_test_temp['probability'])
    trades['eff_trans_date'] = pred_trades['eff_trans_date']
    
    # Calculate buy/sell prices with costs
    trades['buy_date'] = pred_trades['future_prices'].apply(lambda df: df.index[0])
    trades['buy'] = pred_trades['future_prices'].apply(
        lambda df: df['Open'].iloc[0] * (1 + spread_pct/2 + slippage_pct)
    )
    trades['sell_date'] = pred_trades['future_prices'].apply(lambda df: df.index[0])
    trades['sell'] = pred_trades['future_prices'].apply(
        lambda df: df['Close'].iloc[0] * (1 - spread_pct/2 - slippage_pct)
    )
    
    trades = trades.dropna(subset=['buy', 'sell'])
    
    # Compute raw returns
    trades['returns_prop'] = trades['sell'] / trades['buy'] - 1
    trades['returns_capped'] = trades['returns_prop'].clip(lower=stop_loss, upper=take_profit)
    
    # Check for stop-outs
    if check_stopout:
        trades['low'] = pred_trades['future_prices'].apply(lambda df: df['Low'].iloc[0])
        stopout_mask = trades['low'] < trades['buy'] * (1 + stop_loss)
        trades.loc[stopout_mask, 'returns_capped'] = stop_loss
    
    trades = trades.sort_values('eff_trans_date')
    
    # === PROBABILITY-WEIGHTED CAPITAL ALLOCATION ===
    balance = starting_balance
    balances = []
    capital_allocated_list = []
    
    for date, group in trades.groupby('eff_trans_date'):
        # Enforce concurrency cap
        group = group.iloc[:max_concurrent_positions]
        
        # Total capital available for this day
        capital_for_day = balance * daily_total_risk
        
        # Calculate weights based on probabilities
        prob_sum = group['probability'].sum()
        
        # Allocate capital proportionally to probability
        for idx, row in group.iterrows():
            # Weight = this trade's probability / sum of all probabilities
            weight = row['probability'] / prob_sum
            
            # Capital for this specific trade
            per_trade_capital = capital_for_day * weight
            capital_allocated_list.append(per_trade_capital)
            
            # Calculate profit/loss in dollars for this trade
            trade_pnl = row['returns_capped'] * per_trade_capital
            balance += trade_pnl
            balances.append(balance)
    
    # Attach results to trades
    trades = trades.iloc[:len(balances)]  # Ensure length match
    trades['balance'] = balances
    trades['capital_allocated'] = capital_allocated_list
    
    # Calculate metrics
    trades['win'] = trades['returns_capped'] > 0
    trades['loss'] = ~trades['win']
    trades['profit_pct'] = trades['returns_capped'] * 100
    trades['profit_dollars'] = trades['returns_capped'] * trades['capital_allocated']
    
    # Summary by year
    summary = trades.groupby(trades['eff_trans_date'].dt.year).agg(
        start_balance=('balance', 'first'),
        end_balance=('balance', 'last'),
        avg_return=('returns_capped', 'mean'),
        avg_capital_per_trade=('capital_allocated', 'mean'),
        total_capital_deployed=('capital_allocated', 'sum'),
        wins=('win', 'sum'),
        losses=('loss', 'sum'),
        num_trades=('returns_capped', 'size')
    )
    summary['annual_return_%'] = (summary['end_balance']/summary['start_balance'] - 1) * 100
    summary['win_loss_ratio'] = summary['wins'] / summary['losses']
    
    return trades, summary


# Example usage in your notebook:
"""
trades, summary = run_probability_weighted_backtest(
    df_by_ticker=df_by_ticker,
    X_test=X_test,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba,
    starting_balance=10000,
    daily_total_risk=0.33,
    max_concurrent_positions=20
)

# Display results
display(summary)

# Plot balance over time
trades.plot(x='eff_trans_date', y='balance', logy=True, figsize=(12, 6))
plt.title('Portfolio Balance Over Time (Probability-Weighted)')
plt.ylabel('Balance ($, log scale)')
plt.xlabel('Date')
plt.grid(True, alpha=0.3)
plt.show()

# Additional analysis: Compare capital allocation
trades.plot(x='eff_trans_date', y='capital_allocated', figsize=(12, 6), alpha=0.6)
plt.title('Capital Allocated Per Trade (Probability-Weighted)')
plt.ylabel('Capital ($)')
plt.xlabel('Date')
plt.grid(True, alpha=0.3)
plt.show()
"""
