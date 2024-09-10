from ..config import Config
from ..data_processing.holt_winters import HW_model
from ..data_processing.load_data import load_actual_data

def Run_Case(data, ts_model, optimizer, wrapper, save_results=False, save_file=None):
    """
    Run the optimization case.

    params:
    optimizer: optimization model
    wrapper: optimization wrapper

    returns:
    None
    """
    # split data into train and test by 80/20
    train_data = data[:int(0.8*len(data))]
    test_data = data[int(0.8*len(data)):]
    # reset test data index
    test_data.reset_index(drop=True, inplace=True)

    # train
    if ts_model == 'HW':

        ts_instance = HW_model(Config.get_param('m'))
        ts_instance.fit(train_data, Config.get_param('alpha'), Config.get_param('beta'), 
                        Config.get_param('gamma'))

    # test with chosen optimizer and wrapper
    if wrapper == 'periodic':
        from .periodic_wrapper import periodic_opt
        results = periodic_opt(ts_instance, test_data, optimizer, Config.get_param('opt_horizon'), 
                        Config.get_param('reopt_freq'), save_results=save_results, save_file=save_file)

    elif wrapper == 'mpc':
        from .mpc_wrapper import mpc_opt
        
    else:
        print('Wrapper not recognized.')

    return results

if __name__ == '__main__':
    Config.set_params({'Me': 8, 'Mc':2, 'Md':2, 'eta':0.9,
                       'opt_horizon': 8, 'reopt_freq': 8})
    data = load_actual_data('synthetic_data_sigma0.csv')
    results = Run_Case(data['value'], 'HW', 'lp', 'periodic', save_results=True, save_file='lp_results.csv')
    # print(results)
