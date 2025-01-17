from final_package_203.class_basket import StockPortfolio

N = 4  # Number of assets
VT = 0.2  # Volatility target (example)
start_date = '2023-11-15'
end_date = '2025-01-16'

# COV_mat = np.zeros((N, N, data.shape[0]), dtype=float)  # covariance matrix
portfolio = StockPortfolio(N, start_date, end_date, VT)
BT = portfolio.run_optimization()

print(BT)
