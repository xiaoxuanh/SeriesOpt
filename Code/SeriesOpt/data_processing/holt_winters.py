import numpy as np
import pandas as pd
from ..config import Config
from . import tune_params
from . import load_data
from collections import defaultdict

class HW_model:
    def __init__(self, m, l0=None, d0=None, s0=None, season_index0=None, hyperparams=None):
        """
        train: training data, time series
        m: season length
        alpha, beta, gamma: update parameters
        l0, d0, s0: current level, trend, and seasonality values
        season_index0: current season index, pointing to index of the upcoming unknown value
        l0 + d0 + s0[season_index0] is the forecast for the next period
        hyperparams: dictionary of hyperparameters, alpha, beta, gamma
        """
        self.model_name = 'HW'
        self.m = m
        self.fitted = None
        self.cur_l = l0
        self.cur_d = d0
        self.cur_s = s0

        if hyperparams==None:
            self.alpha = Config.get_param('alpha')
            self.beta = Config.get_param('beta')
            self.gamma = Config.get_param('gamma')
        else:
            self.alpha = hyperparams['alpha']
            self.beta = hyperparams['beta']
            self.gamma = hyperparams['gamma']

        self.cur_season_index = 0 if season_index0==None else season_index0

    def reset_hyperparams(self, hyperparams):
        """
        Reset the hyperparameters
        """
        self.alpha = hyperparams['alpha']
        self.beta = hyperparams['beta']
        self.gamma = hyperparams['gamma']

    def fit(self, train, hyperparams=None):
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

        # Update parameters if specified
        if hyperparams!=None:
            self.alpha = hyperparams['alpha']
            self.beta = hyperparams['beta']
            self.gamma = hyperparams['gamma']

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
        self.cur_season_index = (t+1) % self.m 
        # move season index to the next unknown value; so that l+d+s[season_index] is the forecast for the next period
        self.residuals = y[50*self.m:] - fitted[50*self.m:]

        self.hist_l = hist_l
        self.hist_d = hist_d
        self.hist_s = hist_s

    def forecast(self, h, state=None, starting_index=None):
        """
        h: forecast horizon
        """
        # Forecast for the next h steps
        forecast = np.zeros(h)
        if state!=None and starting_index!=None:
            l, d, *s = state
            for j in range(h):
                forecast[j] = l + d + s[(starting_index + j) % self.m]
        else:
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
        s = self.cur_s.copy()
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


    def generate_series(self, n_periods, randomness_models, return_states=False):
        """
        Generates a synthetic Holt-Winters style time series with random level, trend, and seasonality (optional).
        
        Parameters:
        - n_periods: Total number of periods in the time series.
        - self.cur_l: Initial level of the series.
        - self.cur_d: Trend slope for each period.
        - self.cur_s: List of seasonal effects.
        - randomness_models: List of randomness models for each season.
        
        Returns:
        - synthetic_series: Generated time series as a numpy array with seasonality index, seasonality value, level, and trend.
        """
        # Initialize the series
        synthetic_series = []
        cur_l, cur_d, cur_season_index = self.cur_l, self.cur_d, self.cur_season_index
        cur_s = self.cur_s.copy() # make a copy of the current seasonality values
        if return_states:
            states = []
        for t in range(n_periods):
            # Calculate the value at time t
            epsilon = randomness_models[t%self.m].sample()[0] if len(randomness_models)>1 else randomness_models[0].sample()[0]
            value = cur_l + cur_d + cur_s[cur_season_index] + epsilon
            synthetic_series.append(value)
            # update level
            cur_l = cur_l + cur_d + self.alpha * epsilon
            cur_s[cur_season_index] = cur_s[cur_season_index] + (cur_d+epsilon)*self.gamma
            cur_d = cur_d + self.alpha*self.beta * epsilon
            cur_season_index = (cur_season_index + 1) % self.m
            if return_states:
                states.append((cur_l, cur_d, *cur_s))

        # Convert to numpy array for easier manipulation
        synthetic_series = np.array(synthetic_series)
        if return_states:
            states = np.array(states)
            return synthetic_series, states
        else:
            return synthetic_series

    def generate_series_ARerror(self, n_periods, randomness_model):
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
        old_epsilon = 0
        cur_l, cur_d, cur_season_index = self.cur_l, self.cur_d, self.cur_season_index
        cur_s = self.cur_s.copy() # make a copy of the current seasonality values
        for t in range(n_periods):
            # Calculate the value at time t
            epsilon = old_epsilon*0.6 + randomness_model.sample()[0]
            value = cur_l + cur_d + cur_s[cur_season_index] + epsilon
            synthetic_series.append((cur_season_index, cur_l, cur_d, 
                                 cur_s[cur_season_index], 
                                 value))
            
            # update level
            cur_l = cur_l + cur_d + self.alpha * epsilon
            cur_s[cur_season_index] = cur_s[cur_season_index] + (cur_d+epsilon)*self.gamma
            cur_d = cur_d + self.alpha*self.beta * epsilon
            cur_season_index = (cur_season_index + 1) % self.m

            old_epsilon = epsilon

        # Convert to pandas DataFrame for easier manipulation
        synthetic_series = np.array(synthetic_series, dtype=[('seasonality_index', 'i4'), 
                                                        ('level', 'f4'), 
                                                        ('trend', 'f4'),
                                                        ('seasonality_value', 'f4'),  
                                                        ('value', 'f4')])
        synthetic_series = pd.DataFrame(synthetic_series, columns=['seasonality_index', 'level', 'trend', 'seasonality_value', 'value'])

        return synthetic_series


    def dp_func_transition(self, state, epsilon, cur_season_index):
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
        t_new = t + self.alpha * self.beta * epsilon
        l_new = l + t + self.alpha * epsilon
        s_new = s.copy()
        s_new[cur_season_index] = s[cur_season_index] + self.gamma*epsilon
        
        return (l_new, t_new, *s_new)
    
    def dp_generate_state_range(self, init_state, randomness_models, opt_horizon):
        """
        Generate the memo table for the DP model.
        :param init_state: Initial state
        :param randomness_model: Randomness model
        :param opt_horizon: Number of periods to optimize
        :return: Memo table
        """
        if hasattr(randomness_models[0], 'sigma'):
            return self.generate_state_range_continuous(init_state, randomness_models, opt_horizon)
        else:
            return self.generate_state_range_discrete(init_state, randomness_models, opt_horizon)
        
    def generate_state_range_continuous(self, init_state, randomness_models, opt_horizon):
        """
        Generate the memo table for the DP model with continuous randomness.
        :param init_state: Initial state
        :param randomness_model: Randomness model
        :param opt_horizon: Number of periods to optimize
        :return: Memo table
        """
        state_range = [list() for _ in range(opt_horizon)]
        l0, t0, *s0 = init_state

        for k in range(opt_horizon):
            # if randomness_models is a list, use the corresponding randomness model for the period;
            # otherwise, use the same randomness model for all periods
            randomness_model = randomness_models[k % self.m] if isinstance(randomness_models, list) else randomness_models

            period_range = []
            l_variance_term = np.sqrt(self.beta**2 * (k*(k-1)*(2*k-1)/6) 
                                  + self.beta * k*(k-1) + k) * self.alpha * randomness_model.sigma
            lmin = l0 + k * t0 - 1.96 * l_variance_term
            lmax = l0 + k * t0 + 1.96 * l_variance_term
            period_range.append((lmin, lmax))

            t_variance_term = np.sqrt(k) * self.alpha * self.beta * randomness_model.sigma
            tmin = t0 - 1.96 * t_variance_term
            tmax = t0 + 1.96 * t_variance_term
            period_range.append((tmin, tmax))

            # specify relevant seasons. Towards the end of the optimization horizon, some seasons no longer matter.
            cur_season_index = k % self.m # assumes the season index starts from 0
            relevant_seasons = [(cur_season_index + j) % self.m for j in range(opt_horizon - k)]
            # make all seasons relevant
            # relevant_seasons = range(self.m)
            for i in range(self.m):
                if i in relevant_seasons:
                    if i < k % self.m:
                        multiplier = max(0, np.floor((k-1)/self.m)+1)
                    else:
                        multiplier = max(0, np.floor(k/self.m))
                    s_variance_term = np.sqrt(multiplier) * self.gamma * randomness_model.sigma
                    smin = s0[i] - 1.96 * s_variance_term
                    smax = s0[i] + 1.96 * s_variance_term
                else: # use mean if the season is not relevant
                    smin = s0[i]
                    smax = s0[i]
                
                period_range.append((smin, smax))
        
            state_range[k] = period_range
        
        return state_range
    
    def generate_state_range_discrete(self, initial_state, randomness_model, opt_horizon):
        """
        generate the memo table for the DP model with discrete randomness
        """
        memo = defaultdict(dict)
        initial_state = tuple(initial_state)
        memo[0][initial_state] = []

        for k in range(opt_horizon - 1):
            # Determine relevant seasons for period k
            # Relevant seasons are those that will be used in future periods
            cur_season_index = k % self.m 
            future_season_indices = [(cur_season_index + i) % self.m for i in range(opt_horizon - k)]
            relevant_seasons = set(future_season_indices)
            mask = np.isin(np.arange(self.m), list(relevant_seasons), invert=True) # Mask for irrelevant seasons

            # Iterate over all states at time k
            for xk in memo[k]:
                xk_list = list(xk)
                # compute the mean next state when epsilon = 0
                next_x_mean = list(self.dp_func_transition(xk_list, 0, cur_season_index))
                # For each possible value of epsilon, compute the next state
                for epsilon in randomness_model.values:
                    next_x = list(self.dp_func_transition(xk_list, epsilon, cur_season_index))
                    # override the season parameters with next_x_mean if the season is not relevant
                    next_x_mean[2:2+self.m] = np.array(next_x_mean[2:2+self.m], dtype=float)
                    next_x[2:2+self.m] = np.array(next_x[2:2+self.m], dtype=float)
                    # Now apply the np.where logic
                    next_x[2:2+self.m] = np.where(mask, next_x_mean[2:2+self.m], next_x[2:2+self.m])
                    next_x = tuple(np.round(next_x, decimals=2)) 
                    if next_x not in memo[k + 1]:
                        memo[k + 1][next_x] = []
            
            print(k, len(memo[k]))
        return memo


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

    # make prices more coarse-grained from 24 hours to 4 segments in each day
    price_train = [price_train.iloc[i*24:(i+1)*24] for i in range(len(price_train)//24)]
    price_train = load_data.find_opt_season_group(price_train, 4)

    # Initialize the Holt-Winters model
    ts_instance = HW_model(4)
    # Define the hyperparameter space
    param_grid = {
    'alpha': np.linspace(0.01, 0.05, 5),
    'beta': np.linspace(0.01, 0.05, 5),
    'gamma': np.linspace(0.3, 1, 9)
}
    # Perform Bayesian optimization to tune hyperparameters
    tune_params.grid_search_tune_params(ts_instance, price_train, param_grid)
    # Fit the model
    ts_instance.fit(price_train)
    # print the fitted parameters
    print(ts_instance.alpha, ts_instance.beta, ts_instance.gamma)
    # print final rMAE
    print(tune_params.disjoint_time_series_cross_val(ts_instance, price_train, {'alpha': ts_instance.alpha, 'beta': ts_instance.beta, 'gamma': ts_instance.gamma}))