import numpy as np
import pandas as pd
import random
import string
import matplotlib.pyplot as plt
from datetime import datetime
from dataclasses import dataclass
from scipy.optimize import minimize, Bounds
from pybacktestchain.data_module import get_stocks_data, UNIVERSE_SEC, get_stock_data

@dataclass
class StockDataFetcher:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def generate_random_ticker(self, n):
        return [random.choice(UNIVERSE_SEC) for _ in range(n)]

    def get_N_valid_tickers(self, N):
        valid_tickers = set()
        while len(valid_tickers) < N:
            # Generate new random tickers
            new_tickers = self.generate_random_ticker(N - len(valid_tickers))
            # Try to fetch valid tickers
            try:
                df = get_stocks_data(new_tickers, self.start_date, self.end_date)
                if not df.empty:
                    found_tickers = set(df['ticker'].unique()) # Extract tickers that returned data
                    valid_tickers.update(found_tickers) # Add only valid ones
                    print(f"Current valid tickers: {valid_tickers}")
            except Exception as e:
                #pass  # Suppressing errors
                print(f"Error fetching stock data: {e}")
        return list(valid_tickers)

    def get_data(self):
        tickers = self.get_N_valid_tickers(N)
        data = get_stocks_data(tickers, start_date, end_date).groupby(['Date', 'ticker'])['Close'].sum().unstack(
            'ticker').dropna(axis=0).ffill()
        data.index = pd.to_datetime(data.index)
        return data


class ObjectiveFunction:
    """Handles the objective function for optimization."""

    @staticmethod
    def fun_opt(W, R):
        return -np.dot(W, R)  # Maximizing dot product (minimizing negative)

class ConstraintsHandler:
    """Handles portfolio constraints."""

    def __init__(self, a, b, N, VT):
        self.a = a
        self.b = b
        self.N = N
        self.VT = VT  # Volatility Target

    def contrainte(self, W, cov):
        return self.VT - np.sqrt(np.dot(np.dot(W.T, cov), W))

    def get_constraints(self, cov=None):
        """Generates constraints for the optimizer."""
        cons = [
            {'type': 'ineq', 'fun': lambda W: 1 - np.sum([W[i] for i in range(self.a)])},
            {'type': 'ineq', 'fun': lambda W: 1 - np.sum([W[i] for i in range(self.a, self.b)])},
            {'type': 'ineq', 'fun': lambda W: 1 - np.sum([W[i] for i in range(self.b, self.N)])},
            {'type': 'eq', 'fun': lambda W: np.sum(W) - 1},  # Sum of weights must be 1
        ]
        if cov is not None:
            cons.append({'type': 'ineq', 'fun': self.contrainte, 'args': (cov,)})  # Covariance constraint
        return cons
@dataclass
class PortfolioOptimizer:
    """Optimizes the portfolio weights using scipy.optimize.minimize."""

    def __init__(self, N, min_w=0.0, max_w=0.35):
        self.N = N
        self.min_w = min_w
        self.max_w = max_w
        self.x0 = np.array([1 / N for _ in range(N)])  # Initial weights

    def optimize_weights(self, t, return_mat, constraints):
        """Runs optimization for time step t."""
        bounds = Bounds([self.min_w] * self.N, [self.max_w] * self.N)
        res = minimize(ObjectiveFunction.fun_opt, self.x0, bounds=bounds, args=(return_mat.to_numpy()[t, :]), constraints=constraints)
        return res.x
@dataclass
class StockPortfolio:
    """Handles fetching stock data and running the optimizer."""

    def __init__(self, N, start_date, end_date, VT, a=1, b=2, min_w=0.0, max_w=0.5, COV_mat=None):

        self.constraints_handler = ConstraintsHandler(a, b, N, VT)
        self.optimizer = PortfolioOptimizer(N, min_w, max_w)
        self.data = StockDataFetcher(start_date, end_date).get_data()

        self.DCF = pd.Series(((pd.DataFrame(self.data.index) - pd.DataFrame(self.data.index).shift(1)).iloc[:, 0].dt.days).values)

        self.COV_mat = COV_mat
        self.Weights = np.zeros((N, self.data.shape[0]))  # Weight matrix
        self.t_start = 252 // 4
        self.IL_vect = np.zeros((self.data.shape[0]))
        self.IL_vect[:self.t_start + 22] = 100
        self.Exp_vect = np.zeros((self.data.shape[0]))
        self.Exp_vect[:self.t_start + 22] = 1.2
        self.Basket_vect = np.zeros((self.data.shape[0]))
        self.Basket_vect[:self.t_start + 1] = 100
        self.HV_vect = np.zeros((self.data.shape[0]))
        self.t_rebal = np.zeros((self.data.shape[0]))


    def IL(self,t):
        rate = get_stock_data("SOFR", '2023-11-15', "2025-01-16")
        rate.index = pd.to_datetime(rate.Date)
        rate = rate[["Close"]].reindex(self.data.index.to_list()).ffill()
        return self.IL_vect[t - 1] * (1 + self.Exp_vect[t - 1] * (
                    self.Basket_vect[t] / self.Basket_vect[t - 1] - 1 - rate.iloc[t - 1]["Close"] / 10000 * self.DCF[t]))

    def BL(self,t):
        return self.Basket_vect[t - 1] * np.sum(
            [self.Weights[i, t] * (self.data.iloc[t, i] / self.data.shift(1).iloc[t, i]) for i in range(N)])

    def HV(self,t):
        return np.sqrt(252 * np.mean([np.log(self.Basket_vect[h] / self.Basket_vect[h - 1]) ** 2 for h in range(t - 20, t)]))

    def expo(self,t):
        return min(VT / self.HV(t - 2), 1.2)

    def run_optimization(self):
        """Optimizes weights for all time steps."""
        return_mat = np.log(self.data / self.data.shift(1)).multiply(np.sqrt(252 / self.DCF).values, axis=0)

        for t in range(self.t_start,return_mat.shape[0]):
            constraints = self.constraints_handler.get_constraints()#self.COV_mat[:, :, t])
            self.Weights[:, t] = self.optimizer.optimize_weights(t,return_mat, constraints)
            print(self.Weights[:, t].sum())

        for t in range(self.t_start, return_mat.shape[0]):
            self.Basket_vect[t] = self.BL(t)

        for t in range(self.t_start + 20, self.t_start + 23):
            self.HV_vect[t] = self.HV(t)

        for t in range(self.t_start + 22, return_mat.shape[0]):
            self.Exp_vect[t] = self.expo(t)
            self.IL_vect[t] = self.IL(t)
        results = self.IL_vect * 100 / self.IL_vect[self.t_start]
        return results



# Example Usage
if __name__ == "__main__":
    N = 4  # Number of assets
    VT = 0.2  # Volatility target (example)
    start_date = '2023-11-15'
    end_date = '2025-01-16'

   # COV_mat = np.zeros((N, N, data.shape[0]), dtype=float)  # covariance matrix
    portfolio = StockPortfolio(N, start_date, end_date, VT)
    BT = portfolio.run_optimization()

    print(BT)




