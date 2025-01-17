import numpy as np
import pandas as pd
class Stats:
    def __init__(self, df):
        self.df = df
        self.returns = df['Index Level'].pct_change().dropna()

    def annualized_return(self):
        total_return = self.df['Index Level'].iloc[-1] / self.df['Index Level'].iloc[0] - 1
        self.df.index=pd.to_datetime(self.df.index)
        num_years = (self.df.index[-1] - self.df.index[0]).days / 252
        return (1 + total_return) ** (1 / num_years) - 1

    def annualized_volatility(self):
        return self.returns.std() * np.sqrt(252)

    def sharpe_ratio(self, risk_free_rate=0.02):
        excess_return = self.annualized_return() - risk_free_rate
        return excess_return / self.annualized_volatility()

    def max_drawdown(self):
        cumulative_returns = self.df['Index Level'] / self.df['Index Level'].iloc[0]
        rolling_max = cumulative_returns.cummax()
        drawdown = cumulative_returns / rolling_max - 1
        return drawdown.min()

    def summary(self):
        stats_dict = {
            "Annualized Return": self.annualized_return(),
            "Annualized Volatility": self.annualized_volatility(),
            "Sharpe Ratio": self.sharpe_ratio(),
            "Max Drawdown": self.max_drawdown()
        }
        return pd.DataFrame.from_dict(stats_dict, orient='index', columns=["Value"])



# Example Usage
if __name__ == "__main__":
    start_date = '2010-01-01'
    end_date = '2025-01-01'

    # Example DataFrame for Stats
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    index_levels = np.cumprod(1 + np.random.normal(0, 0.01, len(dates))) * 1000
    df_example = pd.DataFrame({'Index Level': index_levels}, index=dates)

    stats = Stats(df_example)
    print("Portfolio Stats:", stats.summary())
