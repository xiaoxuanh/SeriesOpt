import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from collections import defaultdict
import warnings
from ..config import Config
from . import load_data
from .randomness_models import NormalRandomness

class SARIMA_model:
    def __init__(self, p=1, d=0, q=1, P=1, D=0, Q=1, m=4):
        """
        Initialize SARIMA model with specified parameters.
        
        Parameters:
        - p: Order of the AR term
        - d: Order of integration (differencing)
        - q: Order of the MA term
        - P: Order of seasonal AR term
        - D: Order of seasonal integration
        - Q: Order of seasonal MA term
        - m: Number of periods in a season (seasonal length)
        """
        self.model_name = 'SARIMA'
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.m = m  # Season length
        self.fitted = None
        self.model = None
        self.results = None
        self.train = None
            
        # Current state information (will be populated after fitting)
        self.current_state = None
    
    def fit(self, train):
        """
        Fit the SARIMA model to the training data
        
        Parameters:
        - train: Training data, time series
        """
        self.train = np.array(train)
        
        # Filter warnings during model fitting
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            # Create and fit the SARIMAX model
            self.model = SARIMAX(
                self.train,
                order=(self.p, self.d, self.q),
                seasonal_order=(self.P, self.D, self.Q, self.m),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            # Fit the model
            self.results = self.model.fit(disp=False)

        # Generate in-sample predictions
        self.fitted = self.results.fittedvalues
        
        # Calculate residuals for the entire series or after stabilization period
        self.residuals = self.train[50*self.m:] - self.fitted[50*self.m:]
        
        # Extract the final state for forecasting
        # For SARIMA(1,0,1)(1,0,1)_m, we need a state vector with 2m+2 elements
        # Initialize price lags and error lags with zeros
        price_lags = np.zeros(self.m+1)
        error_lags = np.zeros(self.m+1)
        
        # Get the latest prices (up to m+1 lags)
        n = len(self.train)
        for i in range(min(self.m+1, n)):
            price_lags[i] = self.train[n-i-1]
        
        # Get the latest residuals/errors (up to m+1 lags)
        residuals = self.train - self.fitted
        for i in range(min(self.m+1, n)):
            error_lags[i] = residuals[n-i-1]
        
        # Combine price and error lags to form the state vector
        # Format: [p_{t-1}, p_{t-2}, ..., p_{t-m-1}, ε_{t-1}, ε_{t-2}, ..., ε_{t-m-1}]
        self.current_state = tuple(np.concatenate([price_lags, error_lags]))
        
        # Extract model parameters for state transition
        # For SARIMA(1,0,1)(1,0,1)_m
        if hasattr(self.results, 'polynomial_ar'):
            self.phi1 = -self.results.polynomial_ar[1] if len(self.results.polynomial_ar) > 1 else 0
            self.Phi1 = -self.results.polynomial_seasonal_ar[self.m] if len(self.results.polynomial_seasonal_ar) > self.m else 0
            self.theta1 = self.results.polynomial_ma[1] if len(self.results.polynomial_ma) > 1 else 0
            self.Theta1 = self.results.polynomial_seasonal_ma[self.m] if len(self.results.polynomial_seasonal_ma) > self.m else 0
        else:
            # Fallback method to extract parameters if polynomial attributes aren't available
            params = self.results.params
            # This part would need customization based on how parameters are ordered in your model
            # A typical ordering might be: [ar.L1, ma.L1, ar.S.L1, ma.S.L1, sigma2]
            self.phi1 = params[0] if self.p > 0 else 0
            self.theta1 = params[1] if self.q > 0 else 0
            self.Phi1 = params[2] if self.P > 0 else 0
            self.Theta1 = params[3] if self.Q > 0 else 0

        ### Construct transition matrix and input vector for state update; assume SARIMA(1,0,1)(1,0,1)_m
        # State dimension
        state_dim = 2*self.m + 2
        # Create Z vector (coefficients for forecast equation)
        Z = np.zeros(state_dim)
        Z[0] = self.phi1                # coefficient for p_{k-1}
        Z[self.m-1] = self.Phi1                # coefficient for p_{k-m}
        Z[self.m] = -self.phi1 * self.Phi1      # coefficient for p_{k-m-1}
        Z[self.m+1] = self.theta1            # coefficient for ε_{k-1}
        Z[2*self.m] = self.Theta1          # coefficient for ε_{k-m}
        Z[2*self.m+1] = self.theta1 * self.Theta1  # coefficient for ε_{k-m-1}

        # Create T matrix (transition matrix)
        T = np.zeros((state_dim, state_dim))

        # First row is Z
        T[0, :] = Z

        # Tp block: shift matrix for price lags
        for i in range(1, self.m+1):
            T[i, i-1] = 1
        
        # Tε block: shift matrix for error lags
        for i in range(self.m+2, state_dim):
            T[i, i-1] = 1
        
        # Create R vector (input vector for error terms)
        R = np.zeros(state_dim) 
        R[0] = 1    # Add error to predicted price
        R[self.m+1] = 1  # Add new error to error history

        self.T = T
        self.R = R
        self.Z = Z

    
    def forecast(self, h, state=None, ts_args=None):
        """
        Forecast future values
        
        Parameters:
        - h: Forecast horizon
        - state: Optional state vector to start forecasting from
        
        Returns:
        - forecast: Array of forecasted values
        """
        if self.results is None:
            raise ValueError("Model must be fitted before forecasting")
        
        # If state is provided, use the simulate method with zero errors
        # This effectively produces forecasts from the custom state
        if state is None:
            state = self.current_state
        
        # Convert state to numpy array if it's not already
        state = np.array(state)
        # Initialize forecasts array
        forecasts = np.zeros(h)
        # Current state for iteration
        current_state = state.copy()
        # pdb.set_trace()
        for i in range(h):
            # Generate forecast for the next period
            forecasts[i] = self.Z @ current_state
            # Update state for the next iteration
            current_state = self.dp_func_transition(current_state, 0)
        
        return forecasts
    
    def update(self, new_data):
        """
        Update the model with new data; hold parameters fixed, just update the state
        
        Parameters:
        - new_data: New observations to update the model with
        """
        if self.results is None:
            raise ValueError("Model must be fitted before updating")
        
        # Convert to numpy array if not already
        new_data = np.array(new_data)
        
        for val in new_data:
            # new forecast for the next period
            forecast = self.forecast(1)
            # update the fitted values with the new observation
            self.fitted = np.append(self.fitted, forecast[0])
            # Update the residuals
            self.residuals = np.append(self.residuals, val - self.fitted[-1])
            # Update the state vector with the new observation
            self.current_state = self.dp_func_transition(self.current_state, self.residuals[-1])
    
    def generate_series(self, n_periods, randomness_models):
        """
        Generate a synthetic SARIMA time series
        
        Parameters:
        - n_periods: Number of periods to generate
        - randomness_models: Model for generating random innovations
        
        Returns:
        - synthetic_series: Generated time series as a pandas DataFrame
        """
        if self.results is None:
            raise ValueError("Model must be fitted before generating series")
        
        # Generate innovations from the randomness model
        current_state = self.current_state
        simulated = np.zeros(n_periods)
        for i in range(n_periods):
            error = randomness_models[i % self.m].sample()[0] if len(randomness_models)>1 else randomness_models[0].sample()[0]
            # Generate forecast for the next period
            simulated[i] = self.Z @ current_state + error
            # Update the state vector with the generated error
            current_state = self.dp_func_transition(current_state, error)
        
        return simulated
    
    def dp_func_transition(self, state, epsilon, cur_season_index=None):
        """
        Construct the transition matrix T and input vector R for a SARIMA(1,0,1)×(1,0,1)_m model.
        Then update the state vector based on the transition function.
        
        Returns:
        - next_state: Next state vector
        """
        if self.results is None:
            raise ValueError("Model must be fitted before using transition function")
        
        return self.T @ state + self.R * epsilon

    
    def dp_generate_state_range(self, init_state, randomness_models, opt_horizon):
        """
        Generate the state range for dynamic programming optimization
        
        Parameters:
        - init_state: Initial state
        - randomness_models: Models for generating random innovations
        - opt_horizon: Optimization horizon
        
        Returns:
        - state_range: Range of possible states
        """
        # if hasattr(randomness_models[0], 'sigma'):
        return self.generate_state_range_continuous(init_state, randomness_models, opt_horizon)
        
    
    def generate_state_range_continuous(self, init_state, randomness_models, opt_horizon):
        """
        Generate state range for continuous randomness models based on SARIMA structure
        
        Parameters:
        - init_state: Initial state vector structured according to SARIMA components
        - randomness_models: Continuous randomness models (single or list)
        - opt_horizon: Optimization horizon
        
        Returns:
        - state_range: List of state ranges for each period in optimization horizon
        """
        # Convert initial state to numpy array if it's not already
        x0 = np.array(init_state)
        # State dimension
        state_dim = len(x0)

        # Initialize state range list
        state_range = [list() for _ in range(opt_horizon)]
         # Compute expected state and variance for each period
        for k in range(opt_horizon):
            randomness_model = randomness_models[k % self.m] if isinstance(randomness_models, list) else randomness_models

            ########### get mask of relevant indices for this period and future periods
            relevant_mask = np.zeros(state_dim, dtype=bool)
            for future_k in range(k, opt_horizon):
                # indices that will be used
                p1_idx = future_k - k # relative position of p_{k-1}
                pm_idx = future_k - k + self.m - 1 # relative position of p_{k-m}
                p1e1_idx = future_k - k + self.m # relative position of p_{k-m-1}

                # if these indices are within the state vector, set the mask to True
                if p1_idx < self.m + 1:
                    relevant_mask[p1_idx] = True
                if pm_idx < self.m + 1:
                    relevant_mask[pm_idx] = True
                if p1e1_idx < self.m + 1:
                    relevant_mask[p1e1_idx] = True
                
                # indices for error terms
                e1_idx = future_k - k + self.m + 1 # relative position of e_{k-1}
                em_idx = future_k - k + 2*self.m # relative position of e_{k-m}
                e1e1_idx = future_k - k + 2*self.m + 1 # relative position of e_{k-m-1}

                # if these indices are within the state vector, set the mask to True
                if e1_idx < 2*self.m + 2:
                    relevant_mask[e1_idx] = True
                if em_idx < 2*self.m + 2:
                    relevant_mask[em_idx] = True
                if e1e1_idx < 2*self.m + 2:
                    relevant_mask[e1e1_idx] = True

            # Deterministic part: Expected state at period k
            expected_state = np.linalg.matrix_power(self.T,k) @ x0  # Matrix power using ** operator
            # Stochastic part: Compute the full covariance matrix using the formula
            cov_matrix = np.zeros((state_dim, state_dim))
            # pdb.set_trace()
            # Accumulate variance contributions from each error term
            for j in range(k):
                power = k - 1 - j
                T_power = np.linalg.matrix_power(self.T, power)
                T_power_R = T_power @ self.R
                cov_matrix += np.outer(T_power_R, T_power_R)  # Matrix product and transpose
            
            # Scale by error variance
            cov_matrix *= randomness_model.sigma ** 2
            # Extract the diagonal as variance vector
            component_variance = np.diag(cov_matrix)
            
            # Calculate confidence interval width
            ci_width = 1.96 * np.sqrt(component_variance)

            period_range = []
            for i in range(state_dim):
                if relevant_mask[i]:
                    period_range.append((expected_state[i] - ci_width[i], expected_state[i] + ci_width[i]))
                else:
                    period_range.append((expected_state[i], expected_state[i]))
            
            state_range[k] = period_range
        
        return state_range

if __name__ == '__main__':
    # Example usage
    model = SARIMA_model(p=1, d=0, q=1, P=1, D=0, Q=1, m=4)
    import os
    import pdb
    wd = os.getcwd()
    price_pjm = pd.read_csv(os.path.dirname(wd)+'\\data\\PJM.csv')

    # keep the price column only
    price_pjm['Date'] = pd.to_datetime(price_pjm['Date'])
    price_pjm = price_pjm[['Date',' Zonal COMED price']].set_index('Date')[' Zonal COMED price'].asfreq('h')

    # split into train and test
    price_train = price_pjm[price_pjm.index.year!=2018]
    price_test = price_pjm[price_pjm.index.year==2018]

    # make prices more coarse-grained from 24 hours to 4 segments in each day
    price_train = [price_train.iloc[i*24:(i+1)*24] for i in range(len(price_train)//24)]
    price_test = [price_test.iloc[i*24:(i+1)*24] for i in range(len(price_test)//24)]
    segments = load_data.find_opt_season_group(price_train, 4)

    price_train = load_data.aggregate_prices(price_train, segments)
    price_test = load_data.aggregate_prices(price_test, segments)

    # Initialize the Holt-Winters model
    ts_instance = SARIMA_model(m=4)

    # Fit the model to the training data
    ts_instance.fit(price_train)
    # Generate forecasts
    # forecast = ts_instance.forecast(24)
    # print(forecast)
    # forecast2 = ts_instance.forecast(24, state=ts_instance.current_state)
    # print(forecast2)
    # # generate synthetic series
    # randomness_model = NormalRandomness(sigma=10)
    # synthetic_series = ts_instance.generate_series(24, randomness_model)
    # print(synthetic_series)
    # pdb.set_trace()
    # generate state range
    state_range = ts_instance.dp_generate_state_range(ts_instance.current_state, randomness_model, 8)
    print(state_range)
    pdb.set_trace()
    # Update the model with new data
    ts_instance.update([price_test[0]])
    print(ts_instance.current_state)
