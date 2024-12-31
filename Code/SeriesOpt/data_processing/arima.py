import numpy as np
import pandas as pd
from ..config import Config

# ARIMA parameters from the configuration
p = Config.get_param('p')  # AR order
d = Config.get_param('d')  # Differencing order
q = Config.get_param('q')  # MA order

class ARIMA_model:
    def __init__(self, p, d, q, ar_params=None, ma_params=None, history=None):
        """
        p: Order of the AR component    
        d: Order of differencing
        q: Order of the MA component
        ar_params: Coefficients for AR component
        ma_params: Coefficients for MA component
        history: Initial time series history (list or array)
        """
        self.model_name = "ARIMA"
        self.p = p
        self.d = d
        self.q = q
        self.ar_params = np.array(ar_params) if ar_params else np.zeros(p)
        self.ma_params = np.array(ma_params) if ma_params else np.zeros(q)
        self.history = list(history) if history else []
        self.residuals = []

    def difference(self, series, order):
        """
        Apply differencing to the series.
        """
        diff = series.copy()
        for _ in range(order):
            diff = np.diff(diff, n=1)
        return diff

    def inverse_difference(self, diff_series, original_series, order):
        """
        Restore the original series from the differenced series.
        """
        restored = diff_series.copy()
        for _ in range(order):
            restored = np.concatenate(([original_series[-1]], np.cumsum(restored)))
        return restored

    def fit(self, train):
        """
        Fit the ARIMA model to the training data.
        train: Training time series
        """
        self.history.extend(train)
        diff_train = self.difference(self.history, self.d)
        n = len(diff_train)
        fitted = np.zeros(n)
        
        for t in range(max(self.p, self.q), n):
            # Autoregressive component
            ar_term = np.dot(self.ar_params, diff_train[t - self.p:t][::-1]) # [::-1] reverses the array so that the most recent value comes first

            # Moving Average component
            ma_term = np.dot(self.ma_params, self.residuals[-self.q:][::-1]) if len(self.residuals) >= self.q else 0

            fitted[t] = ar_term + ma_term
            self.residuals.append(diff_train[t] - fitted[t])

        self.fitted = self.inverse_difference(fitted, self.history, self.d)

    def forecast(self, h):
        """
        Generate forecasts for h steps ahead.
        """
        forecast = []
        hybrid_history = self.difference(self.history, self.d)
        residuals = self.residuals[-self.q:] if len(self.residuals) >= self.q else [0] * self.q

        for _ in range(h):
            # Autoregressive component
            ar_term = np.dot(self.ar_params, hybrid_history[-self.p:][::-1]) if len(hybrid_history) >= self.p else 0

            # Moving Average component
            ma_term = np.dot(self.ma_params, residuals[::-1]) if residuals else 0

            y_hat = ar_term + ma_term
            forecast.append(y_hat)

            # Update history and residuals
            hybrid_history = np.append(hybrid_history, y_hat)
            residuals = np.append(residuals, 0)

        # Reverse differencing to get the final forecast
        return self.inverse_difference(forecast, self.history, self.d)

    def update(self, new_data):
        """
        Update the model with new data points.
        new_data: New data to extend the time series.
        """
        self.history.extend(new_data)
        self.fit(self.history)

    def generate_series(self, n_periods, randomness_model):
        """
        Generate synthetic ARIMA-style time series.
        n_periods: Number of periods to generate.
        randomness_model: Random noise generator (e.g., normal distribution).
        """
        series = self.history.copy()
        residuals = np.zeros(self.q)

        for _ in range(n_periods):
            ar_term = np.dot(self.ar_params, series[-self.p:][::-1]) if len(series) >= self.p else 0
            ma_term = np.dot(self.ma_params, residuals[::-1]) if len(residuals) >= self.q else 0
            noise = randomness_model.sample()

            value = ar_term + ma_term + noise
            series.append(value)
            residuals = np.append(residuals, noise)

        return pd.Series(series)

if __name__ == "__main__":
    train_data = [100, 110, 120, 130, 125, 128]
    arima_instance = ARIMA_model(p=2, d=1, q=2, ar_params=[0.7, -0.2], ma_params=[0.5, 0.3])
    arima_instance.fit(train_data)
    forecast = arima_instance.forecast(5)
    print("Forecast:", forecast)
