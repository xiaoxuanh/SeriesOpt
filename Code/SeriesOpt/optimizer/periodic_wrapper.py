import pandas as pd
from ..config import Config
from ..utils import get_results_path
from .lp_optimizer import lp_optimize
from .dp_optimizer_cont_b import *
import pickle


def periodic_opt(ts_instance, test_data, optimizer, opt_horizon, reopt_freq, save_results=False, save_file=None):
    """
    Periodically run optimization models against the dataset. Each optimization is done 
    over a horizon of opt_horizon steps. The optimization is run every reopt_freq steps.

    params:
    ts_instance: TimeSeries model instance, e.g. holt winters model, already fitted with training data
    test_data: testing data
    optimizer: optimization model
    H: horizon for each optimization step
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
        for i in range(0, len(test_data), reopt_freq):
            # update the model with the past data
            ts_instance.update(test_data[i-reopt_freq:i])
            # get the price forecast for the next H steps
            p_forecast = ts_instance.forecast(opt_horizon)
            # run the optimizer
            controls = lp_optimize(b, p_forecast, opt_horizon)
            # apply the controls to the test data
            for j in range(reopt_freq):
                if i+j >= len(test_data):
                    break
                action = controls['value'][j]
                control_results.append(action)

                b += action
                b_results.append(b)

                # calculate the profit
                profit = max(action * eta, action / eta) * -test_data[i+j]
                profit_results.append(profit)

    if optimizer == 'dp':
        for i in range(0, len(test_data), reopt_freq):
            # update the model with the past data
            ts_instance.update(test_data[i-reopt_freq:i])
            real_prices = test_data[i:i+reopt_freq]
            # count the number of horizon, assuming the policies have been solved; can change this to 
            # actually solve the policy at each step
            horizon = i//reopt_freq
            with open(get_results_path(f'dp_cont_policies_horizon_{horizon}.pkl'), 'rb') as f:
                dp_policy = pickle.load(f)
            # apply the DP policy to the test data
            dp_profit_horizon, dp_controls_horizon, dp_storage_horizon = apply_dp(real_prices, ts_instance, dp_policy, b)
            # update the storage
            b = dp_storage_horizon[-1]
            # append the results
            control_results.extend(dp_controls_horizon)
            profit_results.extend(dp_profit_horizon)
            b_results.extend(dp_storage_horizon)

    # combine results into a dataframe, with test price data and index
    all_results = pd.DataFrame({'price': test_data, 'control': control_results, 
                                'profit': profit_results, 'b': b_results})

    if save_results:
        # save the results
        all_results.to_csv(get_results_path(save_file), index=False)

    return all_results


if __name__ == '__main__':
    from ..config import Config
    Config.set_params({'Me': 8, 'Mc':2, 'Md':2, 'eta':0.9})
    # this set_params won't carry through if directly run this file. Because lp_optimize is initialized with the default values.
    # which is imported at the top of this file before the set_params is called.

    from ..data_processing.load_data import load_actual_data
    from ..data_processing.holt_winters import HW_model
    # from .lp_optimizer import lp_optimize
    alpha = Config.get_param('alpha')
    beta = Config.get_param('beta')
    gamma = Config.get_param('gamma')
    # Mc = Config.get_param('Mc')
    # Md = Config.get_param('Md')
    # Me = Config.get_param('Me')
    # eta = Config.get_param('eta')
    # print(Mc, Md, Me, eta)

    # generate some data
    data = load_actual_data('synthetic_data_sigma0.csv')
    # split the data
    train_data = data['value'][:200]
    test_data = data['value'][200:400]
    # reset the index
    test_data.reset_index(drop=True, inplace=True)
    # train the model
    ts_instance = HW_model(4)
    ts_instance.fit(train_data, alpha, beta, gamma)
    
    optimizer = 'lp'
    opt_horizon = 12
    reopt_freq = 12

    periodic_opt(ts_instance, test_data, optimizer, opt_horizon, reopt_freq, save_results=True, save_file='lp_results.csv')