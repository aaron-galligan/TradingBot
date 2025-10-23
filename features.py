
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import time, timedelta, datetime
from pandas.tseries.offsets import BMonthEnd
from ta.momentum import RSIIndicator
from ta.trend import MACD
import openinsider_scraper as OIScraper




def clean_scraped_data(df):

    for i in ['last_price', 'Qty', 'shares_held', 'Owned', 'Value']:
        df[i] = (
            df[i]
            .astype(str)
            .str.replace('New', '1000000000', regex=False)
            .str.replace('[\$,\+\-\%>]', '', regex=True)
            .astype(float)
        )

    df['transaction_date'] = pd.to_datetime(df['transaction_date'], format="%Y-%m-%d %H:%M:%S")
    df['trade_date'] = pd.to_datetime(df['trade_date'])

    MARKET_OPEN = time(9, 30)
    def adjust_transaction_date(dt):
        dt = pd.to_datetime(dt)
        if dt.time() < MARKET_OPEN:
            return (dt - pd.Timedelta(days=1)).date()
        return dt.date()

    df['eff_trans_date'] = df['transaction_date'].apply(adjust_transaction_date)


    df = df.sort_values(by='transaction_date')

    return df


def select_price_time_window(trade, downloaded_prices):
    '''
    Download_ticker_prices() gets prices from df.['effective_transaction_date'].min to max for all tickers.
    Now we take just the 130 days around the eff_trans_price to be inserted into df later.
    trade: a single row of df, one openinsider trade.
    downloaded_prices: Price data for all the tickers in df

    returns row: a list to be joined to df.
    '''
    prev_price = downloaded_prices.loc[trade['eff_trans_date'] - timedelta(days=100):trade['eff_trans_date'], (slice(None), trade['ticker'])]
    prev_price.columns = prev_price.columns.droplevel(1)

    future_price = downloaded_prices.loc[trade['eff_trans_date']+ timedelta(days=1) : trade['eff_trans_date'] + timedelta(days=31), (slice(None), trade['ticker'])]
    future_price.columns = future_price.columns.droplevel(1)

    row = {
        'ticker': trade['ticker'],
        'prev_prices': prev_price,
        'future_prices' : future_price
    }
    return row

def download_ticker_prices(df):
    tickers = df['ticker'].unique()

    # Bulk download full date range covering all needed periods
    # any missing failed downloads with still have a row but poplated with NaN's
    start=df['trade_date'].min() - timedelta(days=120)
    end= df['transaction_date'].max() + timedelta(days=60)

    downloaded_prices = yf.download(
        tickers=list(tickers),
        start=start,
        end=end
    )[['Open', 'High', 'Low', 'Close', 'Volume']]

    df_ticker_data = df.apply(lambda trade: select_price_time_window(trade, downloaded_prices), axis=1, result_type='expand')

    df['prev_prices'] = df_ticker_data['prev_prices']
    df['future_prices'] = df_ticker_data['future_prices']

    return df


def add_features(df):



    def create_features(trade):
        """
        ticker_data: DataFrame with OHLCV for one stock over 3 months
        Returns: Single row of engineered features:
            '1mo_return', '3mo_return', '30d_volatility', 'rsi_14', 'macd',
            'volume_zscore', 'price_vs_sma50'.
        """
        #print(trade['eff_trans_date'])

        ticker_data = trade['prev_prices']
        features = {}

        features['trade_date_epoch'] = pd.to_datetime(trade['trade_date']).timestamp() # seconds since epoch
        features['eff_trans_date_epoch'] = pd.to_datetime(trade['eff_trans_date']).timestamp() 

        # Price returns
        features['1mo_return'] = ticker_data['Close'].iloc[-1] / ticker_data['Close'].iloc[-22] - 1
        features['3mo_return'] = ticker_data['Close'].iloc[-1] / ticker_data['Close'].iloc[-63] - 1
        
        # Volatility, use second line to silence the warning.
        features['30d_volatility'] = ticker_data['Close'].pct_change(fill_method=None).std() * np.sqrt(252)
        
        #ATR (Average True Range, 14-day)
        high_low = ticker_data['High'] - ticker_data['Low']
        high_close = np.abs(ticker_data['High'] - ticker_data['Close'].shift())
        low_close = np.abs(ticker_data['Low'] - ticker_data['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['ATR_14'] = tr.rolling(14).mean().iloc[-1]
        
        # Momentum indicators
        # RSI 14
        rsi = RSIIndicator(close=ticker_data['Close'], window=14).rsi()  # series
        features['rsi_14'] = rsi.iloc[-1]
        # MACD line
        macd = MACD(close=ticker_data['Close'])
        features['macd'] = macd.macd().iloc[-1]  # MACD line
        
        # Volume indicators
        features['volume_zscore'] = (
            (ticker_data['Volume'].iloc[-1] - ticker_data['Volume'].mean()) 
            / ticker_data['Volume'].std()
        )
        vol_20 = ticker_data['Volume'].rolling(20).mean().iloc[-1]
        features['Vol_Ratio_20'] = ticker_data['Volume'].iloc[-1] / vol_20 if vol_20 and not np.isnan(vol_20) else np.nan

        # Trend relationships
        features['price_vs_sma50'] = ticker_data['Close'].iloc[-1] / ticker_data['Close'].rolling(50).mean().iloc[-1]

        # Day of the week the trade was made on.
        features['day_sin'] = np.sin(2 * np.pi * trade['eff_trans_date'].weekday() / 7)
        features['day_cos'] = np.cos(2 * np.pi * trade['eff_trans_date'].weekday() / 7)

        # Number of days between making the trade and filing with the SEC
        features['filing_lad_days'] = (trade['transaction_date']-trade['trade_date']).days

        '''
        #Get the shares outstanding so to then calculate market cap with.
        def get_shares_outstanding(ticker):
            ticker_yf = yf.Ticker(ticker)
            info = ticker_yf.info
            return info.get("sharesOutstanding", np.nan)

        if 'ticker' in trade:
            features['shares_outstanding'] = get_shares_outstanding(trade['ticker'])
        else:
            features['shares_outstanding'] = np.nan


        #Market Cap
        if 'shares_outstanding' in trade and pd.notnull(trade['shares_outstanding']):
            features['MarketCap'] = ticker_data['Close'].iloc[-1] * trade['shares_outstanding']
        elif 'MarketCap' in ticker_data.columns:
            features['MarketCap'] = ticker_data['MarketCap'].iloc[-1]
        else:
            features['MarketCap'] = np.nan
        '''
        return pd.Series(features)

    print('Starting create features')
    features = df.apply(lambda trade: create_features(trade), axis=1)
    df[features.columns] = features
    print('Finish create features')




    titles = df['Title'].str.join('')
    df['title_rank'] = np.select(
        [titles.str.contains('CEO'), titles.str.contains('C'), titles.str.contains('Dir')],
        [4, 3, 2],
        default=1
    )



    df['eff_trans_date'] = pd.to_datetime(df['eff_trans_date']).dt.tz_localize(None)



    #Spy 1 day return
    if 'SPY_1d_return' in df.columns:
        df = df.drop('SPY_1d_return', axis = 1)
    spy_history = yf.Ticker('SPY').history(start=df['eff_trans_date'].min(), end=datetime.today())[['Open', 'High', 'Low', 'Close', 'Volume']]
    spy_returns = (spy_history['Close'] / spy_history['Open'] - 1).rename('SPY_1d_return')
    spy_returns.index = spy_returns.index.tz_localize(None)
    df = df.merge(spy_returns, left_on='eff_trans_date', right_index=True, how='left')




    # Bool on whether the transaction was made within trading hours.
    df['is_during_market_hours'] = (
        df['transaction_date'].dt.time.between(time(9,30), time(16))
    )




    month_end = df['trade_date'] + BMonthEnd(0)
    df['month_end_flag'] = ((month_end - df['trade_date']).dt.days < 3).astype(int)
    #df = df.dropna()

    #`'Owned_norm'`: Created inorder to deal with the 'New' tag. Owned is normalised from 0 to 1. 
    # 0 => they already owned lots, their posision has not meaningfully changed. 
    # ~0.5 => A 100% increase in shares owned. 
    # 1 => they are new to owning this stock.
    df['Owned_norm'] = 1 - np.exp(-df['Owned']*np.log(2)/100)





    #Counting how many trades have been made on the same stock.
    def count_same_day_trades(df):
        """
        Adds a column 'same_day_trade_count' representing
        how many trades occurred for the same company on the same eff_trans_date.
        """
        counts = (
            df.groupby(['ticker', 'eff_trans_date'])
            .size()
            .rename('same_day_trade_count')
        )
        return df.merge(counts, on=['ticker', 'eff_trans_date'], how='left')
    
    print('Starting count_same_day_trades')

    df = count_same_day_trades(df)
    print('Finished count_same_day_trades')

    return df


def pull_data(startdate, filtered = True):

    scraper = OIScraper.OpenInsiderScraper()

    print('Scraping data')
    df = scraper.scrape(startdate)
    print('Cleaning data')
    df = clean_scraped_data(df)
    print('Downloading ticker prices')
    df = download_ticker_prices(df)
    print('Adding features, this may take a while.')
    df = add_features(df)


    if filtered:
        print('Removeing trades with out of range or missing values.')
        df = df.dropna()
        df = df[(df[['last_price', 'Qty', 'shares_held', 'Owned', 'Value', 'Owned_norm', 'title_rank', 'filing_lad_days', 'same_day_trade_count']] >= 0).all(axis=1)]

    return df


def update_latest_data(df, filtered = True):
    date_newest_trade = df['transaction_date'].max() #datetine of most recent trade.
    startdate = str(date_newest_trade.date())

    df_new = pull_data(startdate, filtered)

    df_new = df_new[df_new['transaction_date'] > date_newest_trade] #scrape() rounds down to the first of the month, this ensures only new trades are merged.
    merged_df = pd.concat([df, df_new], ignore_index=True)

    return merged_df