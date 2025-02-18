import pandas as pd
from ..config import Config
from ..utils import get_results_path
from .lp_optimizer import lp_optimize
# from .dp_optimizer import dp_optimize


def mpc_opt(ts_instance, test_data, optimizer, reopt_freq, 
            Mc_set=None, Md_set=None,
            save_results=False, save_file=None):
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
    if Mc_set is None:
        Mc_set = [Config.get_param('Mc')] * reopt_freq
    if Md_set is None:
        Md_set = [Config.get_param('Md')] * reopt_freq

    if optimizer == 'lp':
        for i in range(0, len(test_data)):
            # update the model with the past data
            if i > 0:
                ts_instance.update([test_data[i-1]])
            # get the price forecast for the next few steps; step based on receding horizon
            # when i = 0, the forecast is for the next reopt_freq steps; 
            # when i = 1, the forecast is for the next reopt_freq-1 steps, and so on
            receding_horizon = reopt_freq - i % reopt_freq
            p_forecast = ts_instance.forecast(receding_horizon)
            Mc_subset = Mc_set[-receding_horizon:]
            Md_subset = Md_set[-receding_horizon:]
            # run the optimizer
            controls = lp_optimize(b, p_forecast, receding_horizon, Mc_subset, Md_subset)
            # apply the first control to the test data
            action = controls['value'][0]
            control_results.append(action)
            b += action
            b_results.append(b)
            # calculate the profit
            profit = max(action * eta, action / eta) * -test_data[i]
            profit_results.append(profit)

    # combine results into a dataframe, with test price data and index
    all_results = pd.DataFrame({'price': test_data, 'control': control_results, 
                                'profit': profit_results, 'b': b_results})

    if save_results:
        # save the results
        all_results.to_csv(get_results_path(save_file), index=False)

    return all_results


if __name__=="__main__":
    from SeriesOpt.utils import *
    import csv
    import json
    from SeriesOpt.data_processing.holt_winters import HW_model
    from SeriesOpt.data_processing.randomness_models import NormalRandomness

    ############ Normal randomness; real prices ################
    randomness_model = NormalRandomness(Config.get_param('sigma'))
    season = Config.get_param('m')
    wd = os.getcwd()

    ############ PJM data ################
    price_pjm = pd.read_csv(os.path.dirname(wd)+'\\Data\\PJM.csv')
    # keep the price column only
    price_pjm['Date'] = pd.to_datetime(price_pjm['Date'])
    price_pjm = price_pjm[['Date',' Zonal COMED price']].set_index('Date')[' Zonal COMED price'].asfreq('H')
    # split into train and test
    price_train = price_pjm[price_pjm.index.year!=2018]
    price_test = price_pjm[price_pjm.index.year==2018]

    hw_model = HW_model(season)
    hw_model.fit(price_train, hyperparams={'alpha': 0.1, 'beta': 0.1, 'gamma': 0.275})
    test_data = price_test[:season*2]

    # run the MPC optimizer
    results = mpc_opt(hw_model, test_data, 'lp', 12)
    print(results)
