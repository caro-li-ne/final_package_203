# Optimized Basket with Volatility Target and Equally Weighted Basket Comparison

## 📌 Project Overview
This project implements an **optimized investment basket** with a **volatility target**, comparing its performance against an **equally weighted (EW) basket**. It retrieves historical stock data, constructs an optimized portfolio, and visualizes performance over time.

## 📈 Features
✅ **Optimized Basket with Volatility Target**: Uses `scipy.optimize` to compute asset weights while targeting a specific volatility level.  
✅ **Equally Weighted Basket**: A simple benchmark where each asset receives equal weight.  
✅ **Stock Data Fetching**: Dynamically retrieves historical data for a selection of stocks.  
✅ **Performance Comparison**: Visualizes the performance of both portfolios over time.

## 📊 How It Works
1️⃣ **Fetch stock data**: Selects a set of valid tickers and retrieves their historical prices.  
2️⃣ **Optimize portfolio weights**: Uses an optimization algorithm to adjust weights while maintaining the volatility target.  
3️⃣ **Compute equally weighted basket**: Constructs a naive benchmark where all assets have equal weights.  
4️⃣ **Plot results**: Displays the performance of both baskets for easy comparison.

## 🛠 Installation
Make sure you have Python installed and then install the required dependencies:
```bash
pip install numpy pandas matplotlib scipy pybackteschain
```

## 🚀 Usage
Run the script to fetch stock data, optimize weights, and plot the performance comparison:
```bash
python test_performance_comparator.py
```

## 📂 File Structure
```bash
final_package_203/
│── src/
│   ├── performance_comparator.ipynb 
│   ├── class_basket.py            # Portfolio classes
│   ├── stats.py                   # Performance statistics
│── tests/                         # Unit tests
│── README.md                      # Project documentation
```

## 📌 Example Output
The script generates a plot comparing the **optimized portfolio vs. the equally weighted basket** stored in **comparisons_plots**.

## 🛠 Future Improvements
- ✅ Add more optimization strategies (e.g., risk parity, mean-variance)
- ✅ Include real-time stock data retrieval
- ✅ Implement backtesting with risk metrics

## 📬 Contact
For any questions or improvements, feel free to contribute or reach out!

🌍 GitHub: [caro-li-ne]([https://github.com/your-repo](https://github.com/caro-li-ne)


