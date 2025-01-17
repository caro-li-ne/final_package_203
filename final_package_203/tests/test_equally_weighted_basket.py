from final_package_203.class_basket import Equally_weighted, StockDataFetcher
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

N = 3  # Number of assets
start_date = '2023-11-15'
end_date = '2025-01-16'
df_equally_weighted, tickers = Equally_weighted(N, start_date, end_date).compute_equally_weighted_basket()
print("Equally Weighted Basket:\n", df_equally_weighted.head())
