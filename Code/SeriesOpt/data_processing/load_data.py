import numpy as np
import pandas as pd
from .holt_winters import HW_model
from ..utils import get_data_path

def load_synthetic_data(time_series_class, **kwargs) -> pd.DataFrame:
    """
    Load synthetic data for the chosen time series model.

    return: synthetic time series data with columns ['seasonality_index', 'value']
    """
    # Separate the kwargs for initialization and generate_series
    init_kwargs = {k: v for k, v in kwargs.items() if k not in time_series_class.generate_series.__code__.co_varnames}
    gen_kwargs = {k: v for k, v in kwargs.items() if k in time_series_class.generate_series.__code__.co_varnames}
    
    # Initialize the time series model
    model = time_series_class(**init_kwargs)
    
    # Generate the synthetic series
    synthetic_series = model.generate_series(**gen_kwargs)

    # return seasonal index and value
    synthetic_series = synthetic_series[['seasonality_index','value']]
    
    return synthetic_series

def save_synthetic_data(synthetic_series: pd.DataFrame, data_file_name: str):
    """
    Save synthetic data to the given path.
    """
    # Save the synthetic data to the given path
    synthetic_series.to_csv(get_data_path(data_file_name), index=False)

def load_actual_data(data_file_name: str) -> pd.DataFrame:
    """
    Load actual data from the given path.

    return: actual time series data with columns ['seasonality_index', 'value']
    """
    # Load the actual data from the given path
    actual_series = pd.read_csv(get_data_path(data_file_name))
    
    # return seasonal index and value
    actual_series = actual_series[['seasonality_index','value']]
    
    return actual_series

def find_opt_season_group(prices, num_segments):
    """
    Find the optimal seasonality group for the given hourly data.
    num_segments: number of seasons
    prices: 24 hours of data

    Returns:
    dp[num_segments][len(prices)]: Minimum total error.
    segments: List of tuples representing the start and end indices of each segment.
    """
    def calculate_error(start, end):
        """
        Calculate the squared error for a segment from start to end (inclusive).
        """
        segment = prices[start:end+1]
        mean = np.mean(segment)
        error = np.sum((segment - mean) ** 2)
        return error
    
    n = len(prices)
    dp = np.full((num_segments + 1, n + 1), np.inf)  # dp[s][h] -> min error for s segments, h hours
    split = np.zeros((num_segments + 1, n + 1), dtype=int)  # Tracks split points

    # Base case: 1 segment, error is calculated directly
    for h in range(1, n + 1):
        dp[1][h] = calculate_error(0, h - 1)

    # Fill DP table for s segments
    for s in range(2, num_segments + 1):
        for h in range(1, n + 1):
            for k in range(1, h):
                error = calculate_error(k, h - 1)
                if dp[s][h] > dp[s-1][k] + error:
                    dp[s][h] = dp[s-1][k] + error
                    split[s][h] = k

    # Backtrack to find the segments
    segments = []
    current_hour = n
    for s in range(num_segments, 0, -1):
        start = split[s][current_hour]
        segments.append((start, current_hour - 1))
        current_hour = start

    segments.reverse()

    # Compute average price and length for each segment
    for i, (start, end) in enumerate(segments):
        segment = prices[start:end+1]
        mean = np.mean(segment)
        segments[i] = (start, end, mean, len(segment))

    return segments

    

if __name__ == "__main__":

    ### Test the synthetic data generation function ###
    # Example usage
    n_periods = 1000
    initial_level = 50
    trend = 0
    seasonality = [10, -5, 0, 5]  # Example of 4-period seasonality
    sigma = 0  # Noise standard deviation

    randomness = False

    # Generate the synthetic series using load_synthetic_data
    synthetic_series = load_synthetic_data(HW_model, n_periods=n_periods, m=len(seasonality), l0=initial_level,
                                           d0=trend, s0=seasonality, sigma=sigma,
                                           random_level=randomness, random_trend=randomness, random_seasonality=randomness)
    save_synthetic_data(synthetic_series, 'synthetic_data_sigma0.csv')

    # Plot the series values
    import matplotlib.pyplot as plt
    plt.plot(synthetic_series['value'], label="Synthetic Series")
    plt.title("Synthetic Series with Randomized Components (Level, Trend, Seasonality)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
