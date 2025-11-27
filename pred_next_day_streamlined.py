

"""




gpt











"""










import pandas as pd
import xgboost as xgb
from datetime import date, timedelta
import features as ft
import os
import sys
import argparse

# ==========================================
# USER CONFIGURATION
# ==========================================
# Path to the XGBoost model file. 
# You can change this here or pass it as a command line argument.
DEFAULT_MODEL_PATH = "saved_vars/XGBoost/23_10_2025_none_rand_spli_accuracy_0.85_threshold_0.728.json"

# Threshold for prediction probability.
# Trades with probability > threshold will be selected.
DEFAULT_THRESHOLD = 0.728
# ==========================================


def load_model(model_path):
    """Loads the XGBoost model from the specified path."""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    
    print(f"Loading model from {model_path}...")
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)
    return clf

def get_latest_trades():
    """Pulls the latest trades starting from yesterday."""
    yesterday = date.today() - timedelta(days=1)
    print(f"Pulling data starting from {yesterday}...")
    
    # ft.pull_data handles scraping, cleaning, and feature engineering
    todays_trades = ft.pull_data(startdate=str(yesterday), filtered=False)
    
    # Filter to ensure we only have trades from yesterday onwards
    # (scrape might return a bit more depending on implementation)
    todays_trades = todays_trades[todays_trades['eff_trans_date'] >= pd.to_datetime(yesterday)]
    
    return todays_trades

def prepare_data_for_prediction(df, model):
    """Prepares the DataFrame for prediction by selecting relevant features."""
    
    # Columns to drop as they are not features
    # Note: 'last_price' was dropped in the original notebook, ensuring consistency.
    cols_to_drop = [
        'transaction_date', 'trade_date', 'eff_trans_date', 
        'ticker', 'company_name', 'owner_name', 'prev_prices', 'future_prices', 
        'Title', 'transaction_type', 'Owned', 'trade_date_epoch', 'eff_trans_date_epoch',
        'last_price'
    ]
    
    # Drop non-feature columns if they exist
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df_features = df.drop(existing_cols_to_drop, axis=1)
    
    # Reorder/Select columns to match the model's expected input
    # This handles any potential column ordering mismatches
    if hasattr(model, 'feature_names_in_'):
        try:
            df_features = df_features[model.feature_names_in_]
        except KeyError as e:
            print(f"Error: Missing features in data: {e}")
            print("Available columns:", df_features.columns.tolist())
            print("Expected columns:", model.feature_names_in_.tolist())
            sys.exit(1)
            
    return df_features

def main():
    parser = argparse.ArgumentParser(description="Predict next day stock movements based on insider trades.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to the XGBoost model JSON file.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Probability threshold for positive prediction.")
    args = parser.parse_args()

    model_path = args.model
    threshold = args.threshold

    # 1. Load Model
    clf = load_model(model_path)

    # 2. Get Data
    todays_trades = get_latest_trades()
    
    if todays_trades.empty:
        print("No trades found for the specified period.")
        return

    print(f"Found {len(todays_trades)} trades.")

    # 3. Prepare Data
    X = prepare_data_for_prediction(todays_trades, clf)

    # 4. Predict
    print("Running predictions...")
    probs = clf.predict_proba(X)[:, 1]
    preds = (probs > threshold).astype(int)

    # 5. Display Results
    todays_trades['probability'] = probs
    todays_trades['prediction'] = preds
    
    selected_trades = todays_trades[todays_trades['prediction'] == 1]
    
    print("\n" + "="*50)
    print(f"PREDICTIONS (Threshold: {threshold})")
    print("="*50)
    
    if selected_trades.empty:
        print("No trades passed the threshold.")
    else:
        # Select columns to display
        display_cols = ['eff_trans_date', 'ticker', 'company_name', 'owner_name', 'Title', 'probability']
        print(selected_trades[display_cols].to_string(index=False))
        
        print("\nFull details for selected trades:")
        for _, row in selected_trades.iterrows():
            print("-" * 30)
            print(f"Ticker: {row['ticker']}")
            print(f"Company: {row['company_name']}")
            print(f"Owner: {row['owner_name']}")
            print(f"Title: {row['Title']}")
            print(f"Date: {row['eff_trans_date']}")
            print(f"Probability: {row['probability']:.4f}")

if __name__ == "__main__":
    main()
