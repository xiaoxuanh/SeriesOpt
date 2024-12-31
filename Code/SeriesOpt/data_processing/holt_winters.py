import numpy as np
import pandas as pd
from ..config import Config

alpha = Config.get_param('alpha')
beta = Config.get_param('beta')
gamma = Config.get_param('gamma')

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
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.cur_season_index = 0 if season_index0==None else season_index0

    def fit(self, train):
        """
        train: training data, time series
        h: forecast horizon
        m: season length
        alpha, beta, gamma: update parameters
        """
        self.train = train
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
        hist_s = np.zeros((n, self.m))

        # Iterative training
        for t in range(1, n):
            hist_l[t] = l
            hist_d[t] = d
            hist_s[t, :] = s

            fitted[t] = l + d + s[t % self.m]

            prel = l
            prel = l
            pred = d
            l = self.alpha * (y[t] - s[t % self.m]) + (1 - self.alpha) * (prel + pred)
            d = self.beta * (l - prel) + (1 - self.beta) * pred
            s[t % self.m] = self.gamma * (y[t] - l - pred) + (1 - self.gamma) * s[t % self.m]

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
            pred = d
            l = self.alpha * (y[t] - s[season_index]) + (1 - self.alpha) * (prel + pred)
            d = self.beta * (l - prel) + (1 - self.beta) * pred
            s[season_index] = self.gamma * (y[t] - prel - pred) + (1 - self.gamma) * s[season_index]
            season_index = (season_index + 1) % self.m

        self.cur_l = l
        self.cur_d = d
        self.cur_s = s
        self.cur_season_index = season_index


    def generate_series(self, n_periods, randomness_model):
        """
        Generates a synthetic Holt-Winters style time series with random level, trend, and seasonality (optional).
        
        Parameters:
        - n_periods: Total number of periods in the time series.
        - self.cur_l: Initial level of the series.
        - self.cur_d: Trend slope for each period.
        - self.cur_s: List of seasonal effects.
        
        Returns:
        - synthetic_series: Generated time series as a numpy array with seasonality index, seasonality value, level, and trend.
        """
        # Initialize the series
        synthetic_series = []

        for t in range(n_periods):
            # Calculate the value at time t
            epsilon = randomness_model.sample()
            value = self.cur_l + self.cur_d + self.cur_s[self.cur_season_index] + epsilon
            synthetic_series.append((self.cur_season_index, self.cur_l, self.cur_d, 
                                 self.cur_s[self.cur_season_index], 
                                 value))
            
            # update level
            self.cur_l = self.cur_l + self.cur_d + self.alpha * epsilon
            self.cur_s[self.cur_season_index] = self.cur_s[self.cur_season_index] + (self.cur_d+epsilon)*self.gamma
            self.cur_d = self.cur_d + self.alpha*self.beta * epsilon
            self.cur_season_index = (self.cur_season_index + 1) % self.m

        # Convert to pandas DataFrame for easier manipulation
        synthetic_series = np.array(synthetic_series, dtype=[('seasonality_index', 'i4'), 
                                                        ('level', 'f4'), 
                                                        ('trend', 'f4'),
                                                        ('seasonality_value', 'f4'),  
                                                        ('value', 'f4')])
        synthetic_series = pd.DataFrame(synthetic_series, columns=['seasonality_index', 'level', 'trend', 'seasonality_value', 'value'])

        return synthetic_series

    def dp_func_transition(self, state, cur_season_index, epsilon): # TODO: consider merge with the update function
        """
        Transition function for the DP model.
        Holt-Winters price transition function
        :param state: Current states
        :param epsilon: Random noise
        :param period_index: Current period index; 0 to N-1
        :return: Next states
        """
        l, t, *s = state
        s = np.array(s, dtype=float)
        t_new = t + alpha * beta * epsilon
        l_new = l + t + alpha * epsilon
        s_new = s.copy()
        s_new[cur_season_index] = s[cur_season_index] + gamma*epsilon
        
        return (l_new, t_new, *s_new)

    

if __name__ == "__main__":
    import os
    wd = os.getcwd()
    # x0 = [30, 0, -30, 10, -40, -20]
    # ts_instance = HW_model(4, x0[0],x0[1],x0[2:],0) # reset the state
    # series = ts_instance.generate_series(8, 10)
    # print(series)
    price_pjm = pd.read_csv(os.path.dirname(wd)+'\\data\\PJM.csv')

    # keep the price column only
    price_pjm['Date'] = pd.to_datetime(price_pjm['Date'])
    price_pjm = price_pjm[['Date',' Zonal COMED price']].set_index('Date')[' Zonal COMED price'].asfreq('H')

    # split into train and test
    price_train = price_pjm[price_pjm.index.year!=2018]
    price_test = price_pjm[price_pjm.index.year==2018]

    hw_model = HW_model(24)
    hw_model.fit(price_train)