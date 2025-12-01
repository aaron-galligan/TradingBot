"""
analysis_funcs.py contains functions that are used in the analysis.ipynb file.
This file was created just to clean up analysis.ipynb.


"""






import pandas as pd
from datetime import timedelta, time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import itertools
#import features as ft
#import yfinance as yf




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def model_eval(clf, X_test, y_test, plot=False):
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Examine range of probabilities
    print("Min prob:", y_pred_proba.min(), "Max prob:", y_pred_proba.max())

    from sklearn.metrics import precision_recall_curve
    prec, rec, th = precision_recall_curve(y_test, y_pred_proba)
    prec, rec, th = prec[:-1], rec[:-1], th  # drop last precision/recall to match thresholds

    # Filter for thresholds that still yield some recall
    valid = rec > 0
    prec = prec[valid]
    rec = rec[valid]
    th = th[valid]

    # Choose threshold where precision >= recall (balanced conservatism)
    import numpy as np
    idx = np.argmax(prec >= rec)
    chosen_threshold = th[idx] if len(th) > 0 else 0.5

    y_pred = (y_pred_proba > chosen_threshold).astype(int)

    from sklearn.metrics import precision_score, recall_score, accuracy_score

    _precision_score = precision_score(y_test, y_pred)
    _recall_score = recall_score(y_test, y_pred)
    _accuracy_score = accuracy_score(y_test, y_pred)

    print(f"Threshold: {chosen_threshold:.3f}")
    print("Precision:", _precision_score, ' How many of the predicted up days were correct.')
    print("Recall:", _recall_score, ' How many up days were predicted as a proportion.') 
    print("Accuracy:", _accuracy_score)


    if plot:
        confusion_mtx = confusion_matrix(y_test, y_pred) 
        print(confusion_mtx)
        plot_confusion_matrix(confusion_mtx, classes = range(2)) 

    return chosen_threshold, y_pred_proba, y_pred, _precision_score, _recall_score, _accuracy_score











"""
    Runs a (for a single model) backtest with probability-weighted capital allocation.
    
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









"""
Runs a backtest with EQUAL capital allocation per trade (not probability-weighted).

On each day, the daily risk capital is divided equally among all trades that day,
up to max_concurrent_positions.

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
def run_equal_weighted_backtest(df_by_ticker, X_test, y_pred, y_pred_proba,
                                starting_balance=10000,
                                daily_total_risk=0.33,
                                max_concurrent_positions=20,
                                stop_loss=-0.03,
                                take_profit=10,
                                spread_pct=0.001,
                                slippage_pct=0.001,
                                start_date="2024-01-01 00:00:00",
                                check_stopout=True):

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
    
    # === EQUAL CAPITAL ALLOCATION ===
    balance = starting_balance
    balances = []
    capital_allocated_list = []
    
    for date, group in trades.groupby('eff_trans_date'):
        # Enforce concurrency cap
        group = group.iloc[:max_concurrent_positions]
        num_trades = len(group)
        
        # Total capital available for this day
        capital_for_day = balance * daily_total_risk
        
        # Divide equally among all trades
        per_trade_capital = capital_for_day / max(1, num_trades)
        
        # Compute per-trade profit in dollars
        daily_profit = (group['returns_capped'] * per_trade_capital).sum()
        balance += daily_profit
        
        # Record balance and capital for each trade
        for _ in range(num_trades):
            balances.append(balance)
            capital_allocated_list.append(per_trade_capital)
    
    # Attach results to trades
    # Ensure length match by only keeping trades that were processed
    trades = trades.iloc[:len(balances)]
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







def backtest_list_of_models(results,
                        df_by_ticker,
                        X,
                        allocation_method='equal',  # 'equal' or 'probability'
                        spread_pct=0.001,     # round-trip spread assumption (0.1%)
                        slippage_pct=0.001,   # execution slippage (0.1%)
                        stop_loss=-0.03,
                        take_profit=10,
                        check_stopout=True,
                        starting_balance=10000,
                        daily_total_risk=0.33,
                        max_concurrent_positions=5,
                        start_date="2024-01-01 00:00:00"):
    """
    Runs backtests for a list of models using either equal or probability-weighted allocation.

    Args:
        results: Dictionary of model results by year, each containing:
                 'model': clf,
                 'threshold': chosen_threshold,
                 'precision': p,
                 'recall': r,
                 'accuracy': a,
                 'y_test_true': y_test,
                 'y_pred': y_pred,
                 'y_pred_proba': y_pred_proba
        df_by_ticker: Main dataframe with trade data
        X: Feature dataframe
        allocation_method: 'equal' for equal-weighted or 'probability' for probability-weighted
        spread_pct: Round-trip spread assumption
        slippage_pct: Execution slippage
        stop_loss: Stop loss threshold (e.g., -0.03 for -3%)
        take_profit: Take profit cap
        check_stopout: Whether to check for intraday stop-outs
        starting_balance: Initial capital
        daily_total_risk: Fraction of account deployed across all trades per day
        max_concurrent_positions: Maximum number of positions per day
        start_date: Filter trades after this date

    Returns:
        tuple: (all_summaries dict, all_trades dict) - one entry per year/model
    """

    # Select the backtest function based on allocation method
    if allocation_method == 'probability':
        backtest_func = run_probability_weighted_backtest
    elif allocation_method == 'equal':
        backtest_func = run_equal_weighted_backtest
    else:
        raise ValueError(f"allocation_method must be 'equal' or 'probability', got '{allocation_method}'")

    all_summaries = {}
    all_trades = {}

    # Track cumulative balance across years
    cumulative_balance = starting_balance

    for yr, data in results.items():
        y_pred_proba = data['y_pred_proba']
        y_pred = data['y_pred']
        y_test = data['y_test_true']
        X_test = X.loc[y_test.index]
        start_date = pd.Timestamp(str(yr) + '-01-01')
        # Run the selected backtest function
        trades, summary = backtest_func(
            df_by_ticker=df_by_ticker,
            X_test=X_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            starting_balance=cumulative_balance,
            daily_total_risk=daily_total_risk,
            max_concurrent_positions=max_concurrent_positions,
            stop_loss=stop_loss,
            take_profit=take_profit,
            spread_pct=spread_pct,
            slippage_pct=slippage_pct,
            start_date=start_date,
            check_stopout=check_stopout
        )

        # Update cumulative balance for next year
        if len(trades) > 0:
            cumulative_balance = trades['balance'].iloc[-1]

        # Display summary for this year
        # Assuming 'display' is a function available in the environment (e.g., IPython/Jupyter)
        # If not, it should be replaced with print(summary.to_string()) or similar
        display(summary)

        all_summaries[yr] = summary
        all_trades[yr] = trades

    return all_summaries, all_trades


def run_vectorbt_backtest(df_by_ticker, X_test, y_pred, y_pred_proba=None,
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
    Runs a backtest using VectorBT for efficient vectorized backtesting.
    
    Buys at the open of the day after eff_trans_date and sells at the close of the same day.
    
    Args:
        df_by_ticker: Main dataframe with trade data
        X_test: Test features
        y_pred: Binary predictions (0 or 1)
        y_pred_proba: Predicted probabilities (optional, for compatibility)
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
        tuple: (trades DataFrame, summary DataFrame, portfolio object)
    """
    import vectorbt as vbt
    
    # Prepare data
    X_test_temp = X_test.copy()
    if y_pred_proba is not None:
        X_test_temp['probability'] = y_pred_proba
    
    predicted_trues = X_test_temp[y_pred == 1]
    pred_trades = df_by_ticker.loc[predicted_trues.index]
    pred_trades = pred_trades[pred_trades["transaction_date"] > start_date]
    pred_trades = pred_trades.drop_duplicates(subset=['eff_trans_date', 'ticker'], keep='first')
    
    if len(pred_trades) == 0:
        # Return empty results if no trades
        empty_trades = pd.DataFrame()
        empty_summary = pd.DataFrame()
        return empty_trades, empty_summary, None
    
    # Build trades dataframe with future prices
    trades = pd.DataFrame(index=pred_trades.index)
    trades['ticker'] = pred_trades['ticker']
    trades['eff_trans_date'] = pred_trades['eff_trans_date']
    
    # Extract buy/sell prices from future_prices
    trades['buy_date'] = pred_trades['future_prices'].apply(lambda df: df.index[0] if len(df) > 0 else None)
    trades['buy_open'] = pred_trades['future_prices'].apply(lambda df: df['Open'].iloc[0] if len(df) > 0 else None)
    trades['sell_close'] = pred_trades['future_prices'].apply(lambda df: df['Close'].iloc[0] if len(df) > 0 else None)
    trades['low'] = pred_trades['future_prices'].apply(lambda df: df['Low'].iloc[0] if len(df) > 0 else None)
    
    # Drop rows with missing data
    trades = trades.dropna(subset=['buy_open', 'sell_close'])
    
    if len(trades) == 0:
        empty_trades = pd.DataFrame()
        empty_summary = pd.DataFrame()
        return empty_trades, empty_summary, None
    
    # Apply transaction costs
    trades['buy'] = trades['buy_open'] * (1 + spread_pct/2 + slippage_pct)
    trades['sell'] = trades['sell_close'] * (1 - spread_pct/2 - slippage_pct)
    
    # Calculate returns
    trades['returns_prop'] = trades['sell'] / trades['buy'] - 1
    trades['returns_capped'] = trades['returns_prop'].clip(lower=stop_loss, upper=take_profit)
    
    # Check for stop-outs
    if check_stopout:
        stopout_mask = trades['low'] < trades['buy'] * (1 + stop_loss)
        trades.loc[stopout_mask, 'returns_capped'] = stop_loss
    
    # Sort by date
    trades = trades.sort_values('eff_trans_date')
    
    # === PORTFOLIO SIMULATION ===
    # Group trades by date to handle position sizing
    balance = starting_balance
    balances = []
    capital_allocated_list = []
    
    for date, group in trades.groupby('eff_trans_date'):
        # Enforce concurrency cap
        group = group.iloc[:max_concurrent_positions]
        num_trades = len(group)
        
        # Total capital available for this day
        capital_for_day = balance * daily_total_risk
        
        # Divide equally among all trades
        per_trade_capital = capital_for_day / max(1, num_trades)
        
        # Compute per-trade profit in dollars
        daily_profit = (group['returns_capped'] * per_trade_capital).sum()
        balance += daily_profit
        
        # Record balance and capital for each trade
        for _ in range(num_trades):
            balances.append(balance)
            capital_allocated_list.append(per_trade_capital)
    
    # Attach results to trades
    trades = trades.iloc[:len(balances)]
    trades['balance'] = balances
    trades['capital_allocated'] = capital_allocated_list
    
    # Calculate metrics
    trades['win'] = trades['returns_capped'] > 0
    trades['loss'] = ~trades['win']
    trades['profit_pct'] = trades['returns_capped'] * 100
    trades['profit_dollars'] = trades['returns_capped'] * trades['capital_allocated']
    
    # Create a simple portfolio using VectorBT for visualization/analysis
    # Build price series and signals
    all_dates = pd.date_range(start=trades['buy_date'].min(), end=trades['buy_date'].max(), freq='D')
    
    # Create entries and exits signals
    entries = pd.Series(False, index=all_dates)
    exits = pd.Series(False, index=all_dates)
    
    for idx, row in trades.iterrows():
        if row['buy_date'] in entries.index:
            entries.loc[row['buy_date']] = True
            exits.loc[row['buy_date']] = True  # Exit same day
    
    # Build a simple price series (using average buy/sell prices)
    price_series = pd.Series(index=all_dates, dtype=float)
    for date in all_dates:
        day_trades = trades[trades['buy_date'] == date]
        if len(day_trades) > 0:
            price_series.loc[date] = day_trades['buy'].mean()
        else:
            price_series.loc[date] = price_series.ffill().iloc[-1] if len(price_series.dropna()) > 0 else 100
    
    price_series = price_series.ffill().bfill()
    
    # Create VectorBT portfolio
    try:
        pf = vbt.Portfolio.from_signals(
            price_series,
            entries,
            exits,
            init_cash=starting_balance,
            fees=spread_pct + slippage_pct,
            freq='D'
        )
    except Exception as e:
        print(f"Warning: Could not create VectorBT portfolio: {e}")
        pf = None
    
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
    
    return trades, summary, pf
