import pandas as pd
from ..config import Config
from ..utils import get_results_path
from .lp_optimizer import lp_optimize
# from .dp_optimizer import dp_optimize


def mpc_opt(ts_instance, test_data, optimizer, reopt_freq, save_results=False, save_file=None):
    """
    Periodically run optimization models against the dataset. The optimization is rerun at every step.
    Optimization horizon takes a receding horizon approach.

    params:
    ts_instance: TimeSeries model instance, e.g. holt winters model, already fitted with training data
    test_data: testing data
    optimizer: optimization model
    save_path: path to save the results

    returns:
    None
    """

    b_results = []
    control_results = []
    profit_results = []
    b = Config.get_param('b0')
    eta = Config.get_param('eta')

    if optimizer == 'lp':
        for i in range(0, len(test_data)):
            # update the model with the past data
            if i > 0:
                ts_instance.update(test_data[i-1])
            # get the price forecast for the next few steps; step based on receding horizon
            # when i = 0, the forecast is for the next reopt_freq steps; 
            # when i = 1, the forecast is for the next reopt_freq-1 steps, and so on
            receding_horizon = reopt_freq - i % reopt_freq
            p_forecast = ts_instance.forecast(receding_horizon)
            # run the optimizer
            controls = lp_optimize(b, p_forecast, receding_horizon)
            # apply the controls to the test data
            for j in range(receding_horizon):
                if i+j >= len(test_data):
                    break
                action = controls['value'][j]
                control_results.append(action)

                b += action
                b_results.append(b)

                # calculate the profit
                profit = max(action * eta, action / eta) * -test_data[i+j]
                profit_results.append(profit)

    # combine results into a dataframe, with test price data and index
    all_results = pd.DataFrame({'price': test_data, 'control': control_results, 
                                'profit': profit_results, 'b': b_results})

    if save_results:
        # save the results
        all_results.to_csv(get_results_path(save_file), index=False)

    return all_results
