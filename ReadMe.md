TradingBot
=========

Does it work? Yes.

Should you use it? Hell no.

As of 25/10/2025 the theoretical returns of this model mean you *double* your money each year, sounds great. But, this does not account for trading fees and the market spread when buying and selling stocks. The code is also currently spaghetti so you'll do well to actually put it to use.



Short overview
--------------
This project analyses US insider (OpenInsider) filings and builds machine-learning models to predict short-term price movements after filings (next-day, one-week, one-month). It downloads/loads historical price data, engineers time-series and event features (RSI, MACD, volatility, returns, volume z-scores, etc.), creates targets from future price windows, and trains classifiers (XGBoost / RandomForest) while taking care to avoid look-ahead bias.

The goal is to be able to receive a notification of an insider trade (an insider trading meaning a trade by an executive, director, 10% stakeholder, etc. not necessarily an illegal insider trade), analyse the trade, and determine if the stock will grow more or less than 2%. 


Issues
------
Current issue, Getting screwed on the spread:
It was found that there is a general trend of the smaller the stock (trading value) the larger the potential price movement, but similarly the smaller the stock the larger the spread is between the buy and sell price.

Quick explanation on Spread: When you go to buy a stock that is listed at $10, you don't actually end up buy at that price. You'll only be able to buy at a more expensive price e.g. $10.05 and sell at a cheaper price e.g. $9.95. This means if you were to buy and then instantly sell 1 stock, you would loose 10 cents. Meaning the spread for this stock is (0.1/10)*100 = 1%. This is a relatively high spread, for large cap stocks the spread is closer to 0.01% to 0.05%.

Back to the issue, for all these small cap stocks with big movements they can have spreads from 0.5 up to 5% if there is very little liquidity. I have filtered out any stocks that trade below $2 per share, but you may still end up with a spread above 1%.

My average trade return is only 2%, so loosing half of that profit is not ideal.

Now even if the average return was 1% this is still quite good over 400 trades each year, giving an unrealistic return of approx 50% which I'll be honest I just don't trust. I trust that it's predicting the trades correctly, but my estimation of returns sure as hell doesn't reflect true market conditions where for lack of a better phrase, weird shit happens that ruins the returns.




Fun note: If small cap and penny stocks were to be included, an estimated return of 1000% was found. These small cap stock introduce another issue of market liquidity, people just aren't sell the stocks for you to buy them. My system involves buying at market open, selling at market close, some of these markets don't even have a single order filled at the start of the day.
 



Repository contents
--------------------------
- analysis.ipynb — Main analysis notebook. Loads cleaned insider-trade CSV and cached price data, constructs per-trade price windows (`prev_prices`, `future_prices`), engineers features, creates target labels (next-day / 1-week / 1-month thresholds), trains models (XGBoost/RandomForest), evaluates results, and contains code to simulate simple trading using the model predictions.

- features.py — (Module) Utility functions to clean scraped OpenInsider data, download ticker prices in bulk, slice historical windows, and compute feature columns. Used by the update pipeline and notebooks to keep feature code reusable.

- openinsider_scraper.py — (Module) Scraper to collect OpenInsider filings into a CSV / DataFrame.

- Update_dataset.ipynb — Notebook that demonstrates the pipeline to import new scraped data, clean it (via `features`), download prices, compute features for new rows, and update the saved features dataset.

- pred_next_day.ipynb — Notebook intended for using a trained model to make next-day predictions for newly scraped/received trades (lightweight inference workflow).

- saved_vars/ — Directory storing serialized intermediate artifacts and trained models (joblib/pickle). Examples: `All_features_and_prices_df.joblib`, feature snapshots, model JSONs.

- 2015_2025_data/insider_trades.csv — Raw historical OpenInsider CSV used to create training and evaluation data.

- requirements.txt — Python dependencies used for the environment. Use this to reproduce the environment (pip install -r requirements.txt).

- ReadMe.md — (older readme) A previous quick readme. This README is the authoritative short description.

Notes and caveats
-----------------
- The project is time-series sensitive. The notebooks include fixes to avoid look-ahead/data leakage (time-based train/test splits and slicing price windows relative to filing timestamps), but please review the `analysis.ipynb` `get_ticker_data_from_cache` / feature creation code to ensure that no future prices are used when constructing features.

- Use the `Update_dataset.ipynb` pipeline to add new scraped months of data and to regenerate features.

Atribution
----------

The scraper is from sd3v with minor changes made and can be found at https://github.com/sd3v/openinsiderData
