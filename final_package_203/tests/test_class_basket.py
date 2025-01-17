from final_package_203.class_basket import StockPortfolio
import matplotlib.pyplot as plt
import os

name_bt="Backtest"
N = 4  # Number of assets
VT = 0.2  # Volatility target (example)
start_date = '2023-11-15'
end_date = '2025-01-16'

# COV_mat = np.zeros((N, N, data.shape[0]), dtype=float)  # covariance matrix
portfolio = StockPortfolio(N, start_date, end_date, VT)
BT = portfolio.run_optimization()

# Define the path to save the CSV
output_dir = os.path.join(os.path.dirname(__file__), "../backtests")  # Navigate to backtests/
output_path = os.path.join(output_dir, f"{name_bt}.csv")

# Ensure the backtests directory exists
os.makedirs(output_dir, exist_ok=True)

# Save the DataFrame as CSV
BT.to_csv(output_path, index=True)

print(f"✅ DataFrame saved successfully in: {output_path}")
