import pandas as pd
from ..config import Config
from ..utils import get_results_path
from .lp_optimizer import lp_optimize
# from .dp_optimizer import dp_optimize


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