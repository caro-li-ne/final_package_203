from final_package_203.stats import Stats
import pandas as pd
import os

start_date = '2010-01-01'
end_date = '2025-01-01'

# Define the path to save the CSV
backtest_dir = os.path.join(os.path.dirname(__file__), "../backtests")  # Navigate to backtests/

# Test on DataFrame created using class_basket module
df_example = pd.read_csv(f"{backtest_dir}/Backtest.csv",index_col=0)

stats = Stats(df_example)
print("Portfolio Stats:", stats.summary())