import numpy as np
import pandas as pd

class HW_model:
    def __init__(self, m, l0=None, d0=None, s0=None, season_index0=None):
        """
        train: training data, time series
        m: season length
        alpha, beta, gamma: update parameters
        l0, d0, s0: current level, trend, and seasonality values
        season_index0: current season index, pointing to index of the upcoming unknown value
        l0 + d0 + s0[season_index0] is the forecast for the next period
        """
        self.model_name = 'HW'
        self.m = m
        self.fitted = None
        self.cur_l = l0
        self.cur_d = d0
        self.cur_s = s0

        self.cur_season_index = 0 if season_index0==None else season_index0

    def fit(self, train, alpha, beta, gamma):
        """
        train: training data, time series
        h: forecast horizon
        m: season length
        alpha, beta, gamma: update parameters
        """
        self.train = train
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        n = len(self.train)
        fitted = np.zeros(n)
        y = np.array(self.train)

        # Initialize level, trend, and seasonality
        l = y[0] # level
        d = sum(self.train[self.m:2*self.m])/self.m - sum(self.train[:self.m])/self.m # trend
        s = [self.train[i] - sum(self.train[:self.m])/self.m for i in range(self.m)] # seasonal

        # store level, trend, and seasonality
        hist_l = np.zeros(n)
        hist_d = np.zeros(n)
        hist_s = np.zeros(n)

        # Iterative training
        for t in range(1, n):
            hist_l[t] = l
            hist_d[t] = d
            hist_s[t] = s[t % self.m]

            fitted[t] = l + d + s[t % self.m]

            prel = l
            l = self.alpha * (y[t] - s[t % self.m]) + (1 - self.alpha) * (prel + d)
            d = self.beta * (l - prel) + (1 - self.beta) * d
            s[t % self.m] = self.gamma * (y[t] - l) + (1 - self.gamma) * s[t % self.m]

        self.fitted = fitted
        self.cur_l = l
        self.cur_d = d
        self.cur_s = s
        self.cur_season_index = t % self.m

        self.hist_l = hist_l
        self.hist_d = hist_d
        self.hist_s = hist_s

    def forecast(self, h):
        """
        h: forecast horizon
        """
        # Forecast for the next h steps
        forecast = np.zeros(h)
        for j in range(h):
            forecast[j] = self.cur_l + j * self.cur_d + self.cur_s[(self.cur_season_index + j) % self.m]

        return forecast
    
    def update(self, new_data):
        """
        Update the model with new data
        new_data: new data, time series
        """
        l = self.cur_l
        d = self.cur_d
        s = self.cur_s
        season_index = self.cur_season_index
        # Iterative training
        for t in range(0, len(new_data)):
            y = np.array(new_data)
            self.fitted = np.append(self.fitted, l + d + s[season_index])
            prel = l
            l = self.alpha * (y[t] - s[season_index]) + (1 - self.alpha) * (prel + d)
            d = self.beta * (l - prel) + (1 - self.beta) * d
            s[season_index] = self.gamma * (y[t] - l) + (1 - self.gamma) * s[season_index]
            season_index = (season_index + 1) % self.m

        self.cur_l = l
        self.cur_d = d
        self.cur_s = s
        self.cur_season_index = season_index


    def generate_series(self, n_periods, sigma, random_level=False, random_trend=False, random_seasonality=False):
        """
        Generates a synthetic Holt-Winters style time series with random level, trend, and seasonality (optional).
        
        Parameters:
        - n_periods: Total number of periods in the time series.
        - self.cur_l: Initial level of the series.
        - self.cur_d: Trend slope for each period.
        - self.cur_s: List of seasonal effects.
        - sigma: Standard deviation of the noise.
        - random_level: Apply randomness to the level.
        - random_trend: Apply randomness to the trend.
        - random_seasonality: Apply randomness to the seasonality.
        
        Returns:
        - synthetic_series: Generated time series as a numpy array with seasonality index, seasonality value, level, and trend.
        """
        # Initialize the series
        synthetic_series = []

        for t in range(n_periods):
            # Update level and trend based on specified randomness
            if random_level:
                self.cur_l = self.cur_l * np.random.uniform(0.9, 1.1)
            if random_trend:
                self.cur_d = self.cur_d * np.random.uniform(0.9, 1.1)
            
            # Optionally randomize each seasonality value
            seasonality_index = (t + self.cur_season_index + 1) % self.m
            if random_seasonality:
                self.cur_s[seasonality_index] *= np.random.uniform(0.9, 1.1)
            
            # Calculate the value at time t
            self.cur_l = self.cur_l + self.cur_d
            value = self.cur_l + self.cur_s[seasonality_index] + np.random.normal(0, sigma)
            
            # Append the result with seasonality index, seasonality value, level, and trend
            synthetic_series.append((seasonality_index, self.cur_l, self.cur_d, self.cur_s[seasonality_index], value))

        # Convert to pandas DataFrame for easier manipulation
        synthetic_series = np.array(synthetic_series, dtype=[('seasonality_index', 'i4'), 
                                                        ('seasonality_value', 'f4'), 
                                                        ('level', 'f4'), 
                                                        ('trend', 'f4'), 
                                                        ('value', 'f4')])
        synthetic_series = pd.DataFrame(synthetic_series, columns=['seasonality_index', 'level', 'trend', 'seasonality_value', 'value'])

        return synthetic_series


