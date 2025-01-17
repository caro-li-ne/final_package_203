from final_package_203.class_basket import StockPortfolio, StockDataFetcher
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

name_bt="Backtest"
N = 3  # Number of assets
VT = 0.2  # Volatility target (example)
start_date = '2023-11-15'
end_date = '2025-01-16'

# COV_mat = np.zeros((N, N, data.shape[0]), dtype=float)  # covariance matrix
portfolio = StockPortfolio(N, start_date, end_date, VT)
BT, tickers = portfolio.run_optimization()

# Define the path to save the CSV
output_dir = os.path.join(os.path.dirname(__file__), "../backtests")  # Navigate to backtests/
output_path = os.path.join(output_dir, f"{name_bt}{tickers}.csv")

# Ensure the backtests directory exists
os.makedirs(output_dir, exist_ok=True)

# Save the DataFrame as CSV
BT.to_csv(output_path, index=True)

# Plot the DataFrame
plt.figure(figsize=(10, 5))
plt.plot(BT.index, BT["Index Level"], label="Index Level", color="blue")

# Improve x-axis date formatting
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=21))  # Show every 21th day
plt.xticks(rotation=45)  # Rotate labels for better readability

# Labels and title
plt.xlabel("Date")
plt.ylabel("Index Level")
plt.title("Index Level Over Time")
plt.legend()
plt.grid(True)

# Define the path to save the Plot
plot_dir = os.path.join(os.path.dirname(__file__), "../backtests_plots")  # Navigate to backtests/
plot_path = os.path.join(plot_dir, f"BT_{tickers}.png")

# Ensure the backtests directory exists
os.makedirs(plot_dir, exist_ok=True)

# Save Index Level plot as PNG
plt.savefig(plot_path, dpi=300, bbox_inches='tight')


print(f"✅ DataFrame saved successfully in: {output_path}")
print(f"✅ Plot saved successfully in: {plot_path}")
