from final_package_203.class_basket import StockPortfolio, StockDataFetcher, Equally_weighted
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pandas as pd
from final_package_203.stats import Stats

name_bt="Backtest"
N = 3  # Number of assets
VT = 0.2  # Volatility target (example)
start_date = '2023-11-15'
end_date = '2025-01-16'

portfolio = StockPortfolio(N, start_date, end_date, VT)
BT, tickers = portfolio.run_optimization()
BT.index=pd.to_datetime(BT.index)


df_equally_weighted, _ = Equally_weighted(N, start_date, end_date,tickers=None).compute_equally_weighted_basket()
df_equally_weighted.index=pd.to_datetime(df_equally_weighted.index.strftime('%Y-%m-%d'))

df=pd.concat([BT,df_equally_weighted], axis=1)

# Ensure the index is a DatetimeIndex
df.index = pd.to_datetime(df.index)

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(df.index, df.iloc[:, 0], label=df.columns[0], linestyle='-', marker='o')  # First column (BT)
plt.plot(df.index, df.iloc[:, 1], label=df.columns[1], linestyle='-', marker='s')  # Second column (Equally Weighted)

# Formatting
plt.xlabel("Date")
plt.ylabel("Index Level")
plt.title("Comparison of Backtest (BT) and Equally Weighted Basket")
plt.legend()
plt.grid(True)

# Improve x-axis readability
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45)

# Define the path to save the Plot
plot_dir = os.path.join(os.path.dirname(__file__), "../comparisons_plots")  # Navigate to backtests/
plot_path = os.path.join(plot_dir, f"Performance_comparator_{tickers}.png")

# Ensure the backtests directory exists
os.makedirs(plot_dir, exist_ok=True)

# Save Index Level plot as PNG
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

print(f"✅ Plot saved successfully in: {plot_path}")

print("Optimized Basket Stats:", Stats(BT.dropna()).summary())
df_equally_weighted_temp=df_equally_weighted.copy()
df_equally_weighted_temp.columns=['Index Level']
print("Equally Weighted Basket Stats:", Stats(df_equally_weighted_temp).summary())
