import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
from ..config import Config
from . import load_data
import pdb
# from .randomness_models import NormalRandomness

class AR1_model:
    def __init__(self):
        """
        Initialize AR(1) model.
        AR(1) is a special case of ARIMA where p=1, d=0, q=0
        """
        self.model_name = 'AR1'
        self.p = 1  # AR order
        self.d = 0  # Integration order
        self.q = 0  # MA order
        self.fitted = None
        self.model = None
        self.results = None
        self.train = None
        
        # Current state information (will be populated after fitting)
        self.current_state = None
        # AR(1) coefficient
        self.phi = None
        # Constant term
        self.constant = None
        # Error variance
        self.sigma2 = None

    
    def fit(self, train):
        """
        Fit the AR(1) model to the training data
        
        Parameters:
        - train: Training data, time series
        """
        self.train = np.array(train)
        
        # Filter warnings during model fitting
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            # Create and fit the ARIMA model with AR(1) specification
            self.model = ARIMA(
                self.train,
                order=(self.p, self.d, self.q)
            )
            
            # Fit the model
            self.results = self.model.fit()

        # Generate in-sample predictions
        self.fitted = self.results.fittedvalues
        
        # Calculate residuals
        self.residuals = self.train[200:] - self.fitted[200:]
        
        # Extract model parameters
        # In AR(1): x_t = c + φ*x_{t-1} + ε_t
        self.constant = self.results.params[0]
        self.phi = self.results.params[1]
        # Error variance
        self.sigma2 = self.results.params[2]
        
        # For AR(1), the state is simply the previous value
        self.current_state = self.train[-1]
        

    def forecast(self, h, state=None, starting_index=None):
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
        
        # If state is provided, use iterative forecasting
        if state is not None:
            # Initialize forecasts array
            forecasts = np.zeros(h)
            # For stationary AR(1), the process has mean = constant/(1-phi)
            if abs(self.phi) < 1:  # Stationary case
                mean = self.constant / (1 - self.phi)
                # Iterate through forecast horizon
                for i in range(h):
                    # The correct forecasting equation for AR(1) with mean:
                    # x_t = μ + φ(x_{t-1} - μ) = μ(1-φ) + φ*x_{t-1}
                    forecasts[i] = mean + self.phi * (state - mean)
                    # Update state for next iteration
                    state = forecasts[i]
            else:  # Non-stationary case
                # Iterate through forecast horizon
                for i in range(h):
                    forecasts[i] = self.constant + self.phi * state
                    state = forecasts[i]
        else:
            # Use the model's built-in forecast method
            forecast_results = self.results.forecast(steps=h)
            forecasts = np.array(forecast_results)
        
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
        # For AR(1), updating the state is straightforward - we just need the most recent observation
        if len(new_data) > 0:
            self.current_state = [new_data[-1]]
    
    def generate_series(self, n_periods, randomness_model=None):
        """
        Generate a synthetic AR(1) time series
        
        Parameters:
        - n_periods: Number of periods to generate
        - randomness_model: Model for generating random innovations
        
        Returns:
        - synthetic_series: Generated time series
        """
        if self.results is None:
            raise ValueError("Model must be fitted before generating series")
        
        # Generate innovations from the randomness model
        errors = np.random.normal(0, np.sqrt(self.sigma2), n_periods)
        
        # Initialize series with the current state
        simulated = np.zeros(n_periods)
        current_value = self.current_state
        
        for i in range(n_periods):
            # Generate next value: x_t = c + φ*x_{t-1} + ε_t
            next_value = self.constant + self.phi * current_value + errors[i]
            simulated[i] = next_value
            current_value = next_value
        
        return simulated
    
    def dp_func_transition(self, state, epsilon, cur_season_index=None):
        """
        State transition function for AR(1) model.
        
        Parameters:
        - state: Current state
        - epsilon: Random innovation
        
        Returns:
        - next_state: Next state vector
        """
        if self.results is None:
            raise ValueError("Model must be fitted before using transition function")
        
        # For AR(1), next state is c + φ*current_state + ε
        next_state = [self.constant + self.phi * state[0] + epsilon]
        return next_state

    def dp_generate_state_range(self, init_state, randomness_models, opt_horizon):
        """
        Generate the state range for dynamic programming optimization
        
        Parameters:
        - init_state: Initial state
        - opt_horizon: Optimization horizon
        
        Returns:
        - state_range: Range of possible states
        """

        return self.generate_state_range_continuous(init_state, randomness_models, opt_horizon)

    
    def generate_state_range_continuous(self, init_state, randomness_models, opt_horizon):
        """
        Generate state range for continuous randomness model based on AR(1) structure
        
        Parameters:
        - init_state: Initial state value
        - opt_horizon: Optimization horizon
        
        Returns:
        - state_range: List of state ranges for each period in optimization horizon
        """
        # Convert initial state to numpy array if it's not already
        x0 = init_state  # Extract the scalar value
        
        # Initialize state range list
        state_range = [list() for _ in range(opt_horizon)]
        
        # For AR(1): x_t = c + φ*x_{t-1} + ε_t
        # Expected state after k periods: E[x_t+k] = c*(1-φ^k)/(1-φ) + φ^k*x_t (if |φ| < 1)
        # Variance after k periods: Var[x_t+k] = σ²*(1-φ^(2k))/(1-φ²) (if |φ| < 1)
        
        # Handle special case when phi is close to 1
        if abs(self.phi - 1.0) < 1e-10:
            for k in range(opt_horizon):
                # If φ=1, then AR(1) becomes a random walk with drift
                # E[x_t+k] = x_t + k*c
                expected_value = x0 + k * self.constant
                # Var[x_t+k] = k*σ²
                variance = k * self.sigma**2
                
                # Calculate confidence interval
                ci_width = 1.96 * np.sqrt(variance)
                state_range[k] = [(expected_value - ci_width, expected_value + ci_width)]
        else:
            for k in range(opt_horizon):
                # Expected value calculation
                if abs(self.phi) < 1:  # Stationary case
                    expected_value = self.constant * (1 - self.phi**k) / (1 - self.phi) + self.phi**k * x0
                else:  # Non-stationary case
                    expected_value = self.phi**k * x0
                    if self.constant != 0:
                        # Include constant term's cumulative effect
                        expected_value += self.constant * (1 - self.phi**k) / (1 - self.phi)
                
                # Variance calculation
                if abs(self.phi) < 1:  # Stationary case
                    variance = self.sigma2**2 * (1 - self.phi**(2*k)) / (1 - self.phi**2)
                else:  # Non-stationary case
                    # Sum of squared phi powers: 1 + φ² + φ⁴ + ... + φ²(k-1)
                    variance = self.sigma2**2 * sum(self.phi**(2*j) for j in range(k))
                
                # Calculate confidence interval
                ci_width = 1.96 * np.sqrt(variance)
                state_range[k] = [(expected_value - ci_width, expected_value + ci_width)]
        
        return state_range

if __name__ == '__main__':
    # Example usage
    model = AR1_model()
    import os
    
    # Sample data path
    wd = os.getcwd()
    price_data = pd.read_csv(os.path.dirname(wd)+'\\data\\PJM.csv')

    # Keep the price column only
    price_data['Date'] = pd.to_datetime(price_data['Date'])
    price_data = price_data[['Date', ' Zonal COMED price']].set_index('Date')[' Zonal COMED price'].asfreq('h')

    # Split into train and test
    price_train = price_data[price_data.index.year != 2018]
    price_test = price_data[price_data.index.year == 2018]

    # make prices more coarse-grained from 24 hours to 4 segments in each day
    price_train = [price_train.iloc[i*24:(i+1)*24] for i in range(len(price_train)//24)]
    price_test = [price_test.iloc[i*24:(i+1)*24] for i in range(len(price_test)//24)]
    segments = load_data.find_opt_season_group(price_train, 4)

    price_train = load_data.aggregate_prices(price_train, segments)
    price_test = load_data.aggregate_prices(price_test, segments)

    # Fit the model to the training data
    model = AR1_model()
    model.fit(price_train)
    
    # Generate forecasts
    forecast = model.forecast(24)
    print("Standard forecast:")
    print(forecast)
    
    forecast2 = model.forecast(24, state=model.current_state)
    print("\nForecast from current state:")
    print(forecast2)
    
    # Generate synthetic series
    synthetic_series = model.generate_series(24)
    print("\nSynthetic series:")
    print(synthetic_series)
    
    # Generate state range
    state_range = model.dp_generate_state_range(model.current_state, 4)
    print("\nState range:")
    print(state_range)
    
    # Update the model with new data
    model.update([price_test[0]])
    print("\nUpdated state:")
    print(model.current_state)
    