import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error

def run_darvas_replication():
    print("Starting replication of Darvas & Schepp (2024) model...")

    # 1. Data Fetching
    ticker = "EURUSD=X"
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, start="2015-01-01", interval="1d")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [col.lower() for col in df.columns]
    df.index.name = 'time'

    # 2. Preprocessing
    df = df[~df.index.duplicated(keep='first')]
    df['s'] = np.log(df['close'])

    # 3. Define Proxy for Long-Maturity Forward Rate (f_t^n)
    W = 750
    df['f_proxy'] = df['s'].rolling(window=W).mean()
    df['x'] = df['f_proxy'] - df['s']

    # 4. Forecast Horizon (h = 22 days ~ 1 month)
    H = 22
    df['target'] = df['s'].shift(-H) - df['s']

    df = df.dropna(subset=['x']) # Keep s and x, target might be NaN at the end

    # 5. Recursive Out-of-Sample Forecasting (WITHOUT LOOK-AHEAD BIAS)
    split_idx = int(len(df) * 0.8)

    print(f"Total samples: {len(df)}")
    print(f"Testing period start: {df.index[split_idx].date()}")

    predictions = []
    actuals = []
    dates = []

    print("Running recursive OOS forecasting (correctly avoiding future data)...")
    # Loop through the OOS test period
    # At each time t (represented by index i), we want to forecast target at i (which is s_{i+H} - s_i)
    # The training set MUST only include realized targets.
    # A target at index j is realized at time j+H.
    # So for forecasting at time i, we can only use training data up to index j where j + H <= i.
    for i in range(split_idx, len(df) - H):
        # Correctly avoiding look-ahead bias:
        # The last usable training observation j must satisfy j + H <= i
        last_train_idx = i - H
        if last_train_idx < 1: continue # Not enough data yet

        current_train = df.iloc[:last_train_idx+1].dropna(subset=['target'])

        if len(current_train) < 100: continue # Need minimum training size

        # Estimate ECM: Δs_{t+h} = α + β(f_t - s_t) + ε
        model = smf.ols(formula='target ~ x', data=current_train).fit()

        # Predict change using information available AT time i
        x_t = df.iloc[i]['x']
        pred_change = model.params['Intercept'] + model.params['x'] * x_t

        predictions.append(pred_change)
        actuals.append(df.iloc[i]['target'])
        dates.append(df.index[i])

    results = pd.DataFrame({
        'Actual': actuals,
        'Model_Pred': predictions,
        'RW_Pred': 0
    }, index=dates)

    # 6. Performance Metrics
    rmse_model = np.sqrt(mean_squared_error(results['Actual'], results['Model_Pred']))
    rmse_rw = np.sqrt(mean_squared_error(results['Actual'], results['RW_Pred']))
    ratio = rmse_model / rmse_rw

    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"Forecast Horizon: {H} days")
    print(f"RMSE Model:      {rmse_model:.6f}")
    print(f"RMSE Random Walk: {rmse_rw:.6f}")
    print(f"RMSE Ratio:      {ratio:.4f}")

    # 7. Visualization
    plt.figure(figsize=(14, 7))
    plt.plot(results['Actual'].cumsum(), label='Actual Cumulative returns', color='blue', alpha=0.6)
    plt.plot(results['Model_Pred'].cumsum(), label='Model Predicted Cumulative returns', color='red', linestyle='--')
    plt.title(f'Corrected Cumulative Forecast Performance (h={H}, W={W})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('darvas_model_performance_fixed.png')
    print("\nPerformance plot saved as 'darvas_model_performance_fixed.png'")

if __name__ == "__main__":
    run_darvas_replication()
