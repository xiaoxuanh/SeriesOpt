import numpy as np
from sklearn.model_selection import TimeSeriesSplit
# from bayes_opt import BayesianOptimization
from ..config import Config
from itertools import product
from statsmodels.stats.diagnostic import acorr_ljungbox

opt_horizon = Config.get_param('opt_horizon')

# Define the evaluation metrics
def MAPE(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    actual = np.where(actual == 0, 0.01, actual)  # Avoid division by zero
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def rMAE(actual, predicted, naive_predictions):
    actual, predicted = np.array(actual), np.array(predicted)
    naive_predictions = np.array(naive_predictions)
    return np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual - naive_predictions))

def MASE(actual, predicted, naive_predictions):
    actual, predicted = np.array(actual), np.array(predicted)
    naive_predictions = np.array(naive_predictions)
    return np.mean(np.abs(actual - predicted)) / np.mean(np.abs(actual - naive_predictions))

def MASE_iderror(actual, predicted, naive_predictions):
    actual, predicted = np.array(actual), np.array(predicted)
    naive_predictions = np.array(naive_predictions)
    mase = np.mean(np.abs(actual - predicted)) / np.mean(np.abs(actual - naive_predictions))
    # calculate the lb stat of errors
    error = actual - predicted
    lb_stat = np.min(acorr_ljungbox(error, lags=4)['lb_pvalue'].values)
    # return a high value if the lb_stat is significant
    if lb_stat < 0.05:
        return np.inf
    else:
        return mase

def time_series_cross_val(ts_instance, price_train, params):
    """
    Perform time series cross-validation to evaluate model performance.
    """
    tscv = TimeSeriesSplit(n_splits=10)
    rmae_scores = []

    for train_index, test_index in tscv.split(price_train):
        train, test = price_train[train_index], price_train[test_index[:opt_horizon]]
        ts_instance.fit(train, hyperparams=params)
        predictions = ts_instance.forecast(len(test))
        rmae_scores.append(rMAE(test, predictions, price_train[-opt_horizon:]))

    return -np.mean(rmae_scores)  # Negative because Bayesian optimization maximizes

def disjoint_time_series_cross_val(ts_instance, price_train, params):
    """
    Perform time series cross-validation with disjoint horizons to evaluate model performance.

    Args:
        ts_instance: Time series model instance.
        params: Dictionary of hyperparameters.
        test_horizon: Forecast horizon for each split.
        n_splits: Number of disjoint test splits.

    Returns:
        Negative rMAE as Bayesian optimization maximizes the objective.
    """
    n = len(price_train)
    n_splits = 10 # Number of disjoint splits
    split_size = n // (n_splits + 1)  # Size of each split, leaving room for test sets
    mase_scores = []

    for i in range(n_splits):
        # Determine training and test indices
        train_end = (i + 1) * split_size
        test_start = train_end
        test_end = test_start + opt_horizon

        if test_end > n:  # Ensure we don't exceed the dataset size
            break

        train = price_train[:train_end]
        test = price_train[test_start:test_end]

        # Fit and forecast
        ts_instance.fit(train, hyperparams=params)
        predictions = ts_instance.forecast(opt_horizon)
        mase_scores.append(MASE_iderror(test, predictions, price_train[train_end - opt_horizon:train_end]))

    return -np.mean(mase_scores)  # Negative because Bayesian optimization maximizes

# not working well
# def Bayesian_optimize_tune_params(ts_instance, price_train, param_dict):
#     """
#     Perform Bayesian optimization to tune hyperparameters for a time series model.

#     Args:
#         ts_instance: Time series model instance with `fit` and `forecast` methods.
#         param_dict: Dictionary specifying hyperparameter ranges for Bayesian optimization.
#                     Example: {'alpha': (0.01, 1), 'beta': (0.005, 1), 'gamma': (0.01, 0.5)}

#     Returns:
#         dict: Best hyperparameters found.
#     """
#     def target_function(**params):
#         """
#         Objective function for Bayesian optimization.
#         Dynamically passes parameters to `fit` method.
#         """
#         score = disjoint_time_series_cross_val(ts_instance, price_train, params)
#         if np.isnan(score) or score < -1e6:
#             return -1e6  # Large penalty for invalid scores
#         else:
#             return score

#     # Bayesian optimization
#     optimizer = BayesianOptimization(
#         f=target_function,
#         pbounds=param_dict,
#         random_state=10
#     )
#     optimizer.maximize(init_points=50, n_iter=1000)
#     output_params = optimizer.max['params']
#     ts_instance.reset_hyperparams(output_params)


def grid_search_tune_params(ts_instance, price_train, param_grid):
    """
    Perform grid search to tune hyperparameters for a time series model.

    Args:
        ts_instance: Time series model instance.
        param_grid: Dictionary specifying hyperparameter ranges for grid search.

    Returns:
        dict: Best hyperparameters found.
    """
    best_params = None
    best_score = -np.inf

    # Generate all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    for combination in product(*values):
        params = dict(zip(keys, combination))
        score = time_series_cross_val(ts_instance, price_train, params)

        if score > best_score:
            best_score = score
            best_params = params

        print(f"Params: {params}, Score: {score}")

    print(f"Best Params: {best_params}, Best Score: {best_score}")
    ts_instance.reset_hyperparams(best_params)
    return best_params