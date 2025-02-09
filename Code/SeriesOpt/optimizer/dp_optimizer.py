import pandas as pd
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from itertools import product
from collections import defaultdict
from SeriesOpt.config import Config
from SeriesOpt.data_processing.randomness_models import *
from SeriesOpt.data_processing.holt_winters import HW_model
from SeriesOpt.data_processing import load_data
from multiprocessing import Pool

Me = Config.get_param('Me')
Mc = Config.get_param('Mc')
Md = Config.get_param('Md')
eta = Config.get_param('eta')
alpha = Config.get_param('alpha')
beta = Config.get_param('beta')
gamma = Config.get_param('gamma')
sigma = Config.get_param('sigma')
m = Config.get_param('m')
max_num_b_states = Config.get_param('max_num_b_states')
max_b_step_size = Config.get_param('max_b_step_size')
max_num_x_states = Config.get_param('max_num_x_steps')
min_x_step_size = Config.get_param('min_x_step_size')
opt_horizon = Config.get_param('opt_horizon')

def _single_step_opt(b, w_i, y_i, p) -> tuple:
    """
    Solve Single-step problem with closed-form solution

    params:
    b: current battery state
    w_i, y_i: slopes and params of the J function from period k+1
    p: price forecast for period k

    returns:
    v_star: optimal value
    u_star: optimal control
    b_coef: integration of the corresponding coefficient of b of the J_k function over epsilon
    intercept: integration of the corresponding intercept of the J_k function over epsilon
    """
    
    # Compute the upper and lower bounds for u, different coefficients of b depending where are the 
    # upper and lower bounds
    upper_u = min(Mc, Me - b)
    lower_u = max(-Md, -b)

    def _solve_sub_problem(wi, yi, lower_u, upper_u, coef, b, is_negative = False):
        """
        coef: p/eta or p*eta
        """
        wi = np.array(wi)
        yi = np.array(yi)

        # Compute wi - coef once
        wi_minus_coef = wi - coef
        if is_negative:
            # Adjust masks for discharging
            min_pos_mask = wi_minus_coef > 0
            min_neg_mask = wi_minus_coef <= 0
        else:
            # Masks for charging
            min_pos_mask = wi_minus_coef >= 0
            min_neg_mask = wi_minus_coef < 0

        # Handle positive part
        if upper_u == Me - b and not is_negative:
            min_pos = np.column_stack([
                wi_minus_coef[min_pos_mask] * upper_u + b * wi[min_pos_mask] + yi[min_pos_mask],
                np.full(np.sum(min_pos_mask), upper_u),  # Fill with upper_u
                np.full(np.sum(min_pos_mask), coef),
                yi[min_pos_mask] + wi_minus_coef[min_pos_mask] * Me
            ])
        else:
            min_pos = np.column_stack([
                wi_minus_coef[min_pos_mask] * upper_u + b * wi[min_pos_mask] + yi[min_pos_mask],
                np.full(np.sum(min_pos_mask), upper_u),
                wi[min_pos_mask],
                yi[min_pos_mask] + wi_minus_coef[min_pos_mask] * upper_u
            ])

        # Handle negative part
        if lower_u == -b and is_negative:
            min_neg = np.column_stack([
                wi_minus_coef[min_neg_mask] * lower_u + b * wi[min_neg_mask] + yi[min_neg_mask],
                np.full(np.sum(min_neg_mask), lower_u),  # Fill with lower_u
                np.full(np.sum(min_neg_mask), coef),
                yi[min_neg_mask]
            ])
        else:
            min_neg = np.column_stack([
                wi_minus_coef[min_neg_mask] * lower_u + b * wi[min_neg_mask] + yi[min_neg_mask],
                np.full(np.sum(min_neg_mask), lower_u),
                wi[min_neg_mask],
                yi[min_neg_mask] + wi_minus_coef[min_neg_mask] * lower_u
            ])

        # Handle cross terms (wi - coef > 0 and wi - coef < 0)
        cross_mask_i = wi_minus_coef > 0
        cross_mask_j = wi_minus_coef < 0

        wi_pos = wi[cross_mask_i]
        yi_pos = yi[cross_mask_i]
        wi_neg = wi[cross_mask_j]
        yi_neg = yi[cross_mask_j]
        wi_pos_minus_coef = wi_minus_coef[cross_mask_i]
        wi_neg_minus_coef = wi_minus_coef[cross_mask_j]

        if wi_pos.size > 0 and wi_neg.size > 0:
            wi_diff = wi_pos[:, None] - wi_neg  # (broadcasted subtraction)
            yi_diff = yi_neg - yi_pos[:, None]

            cross_terms = np.column_stack([
                (b * coef + (wi_pos_minus_coef[:, None] * yi_neg - wi_neg_minus_coef * yi_pos[:, None]) / wi_diff).reshape(-1),
                (-b + yi_diff / wi_diff).reshape(-1),
                np.full(wi_diff.size, coef),
                ((wi_pos_minus_coef[:, None] * yi_neg - wi_neg_minus_coef * yi_pos[:, None]) / wi_diff).reshape(-1)
            ]) 
        else:
            cross_terms = np.empty((0, 4))  # Handle cases where there are no cross terms

        # Combine all terms
        combined = np.vstack([min_pos, min_neg, cross_terms])
        
        ### Find the minimum v_star value and corresponding u_star, b_coef, and intercept.
        ### If there are multiple optimal solutions, choose the one with the largest b_coef.
        # Find the minimum v_star value
        min_v_star = np.min(combined[:, 0])
        # Get all rows where v_star equals min_v_star
        min_v_rows = combined[np.abs(combined[:, 0] - min_v_star)<1e-6]
        # From these, select the one with the largest or smallest b_coef (third element)
        # if b is zero, choose the smallest b_coef; otherwise, choose the largest b_coef
        if b == 0:
            selected_row = min_v_rows[np.argmin(min_v_rows[:, 2])]
        else:
            selected_row = min_v_rows[np.argmax(min_v_rows[:, 2])]
        # Unpack the selected row
        v_star, u_star, corresponding_b_coef, corresponding_intercept = selected_row

        # Ensure u_star is within bounds
        u_star = np.clip(u_star, lower_u, upper_u)
        
        return v_star, u_star, corresponding_b_coef, corresponding_intercept
    
    # Solving sub-problems
    v_star_pos, u_star_pos, b_coef_pos, intercept_pos = _solve_sub_problem(w_i, y_i, 0, upper_u, p/eta, b, is_negative=False)
    # print(v_star_pos, u_star_pos, b_coef_pos)
    v_star_neg, u_star_neg, b_coef_neg, intercept_neg = _solve_sub_problem(w_i, y_i, lower_u, 0, p*eta, b, is_negative=True)
    # print(v_star_neg, u_star_neg, b_coef_neg)
    
    # Optimal solution
    if v_star_pos < v_star_neg or (v_star_pos == v_star_neg and b == 0):
        v_star = v_star_neg
        u_star = u_star_neg
        b_coef = b_coef_neg
        intercept = intercept_neg

    else:
        v_star = v_star_pos
        u_star = u_star_pos
        b_coef = b_coef_pos
        intercept = intercept_pos
    
        # if the two values are equal, when b=Me, choose the positive side value because you can only approximate
        # Me from charging side, and when b=0, choose the negative side value; 
        # otherwise, choose the positive side value b/c the coefficient and intercept would be the same

    # return four decimal places for the optimal values
    return round(v_star, 4), round(u_star, 4), round(b_coef, 4), round(intercept, 4)

def __hw_price_transition(xk, cur_season_index, epsilon):
    """
    Holt-Winters price transition function
    :param cur_x: Current states
    :param epsilon: Random noise
    :param period_index: Current period index; 0 to N-1
    :return: Next states
    """
    l, t, *s = xk
    s = np.array(s, dtype=float)
    t_new = t + alpha * beta * epsilon
    l_new = l + t + alpha * epsilon
    s_new = s.copy()
    s_new[cur_season_index] = s[cur_season_index] + gamma*epsilon
    
    return (l_new, t_new, *s_new)

def __integrand(z, interval, component, next_x_dict, memo, k, cur_season_index, xk, randomness_model):
    """
    For a given z and component (0 for f and 1 for g), 
    return the integrand value
    """
    # calculate the corresponding x_k+1 given z and x_k
    if z not in next_x_dict:
        next_x_dict[z] = __hw_price_transition(xk, cur_season_index, z)
    next_x = next_x_dict[z]
    try:
        func_value = memo[k+1][next_x][interval][component]
    except KeyError:
        # find the nearest next_x in the memo
        keys = list(memo[k+1].keys())
        keys.remove('num_intervals')
        next_x = min([x for x in keys], key=lambda x: np.linalg.norm(np.array(x) - np.array(next_x)))
        func_value = memo[k+1][next_x][interval][component]
    return func_value * randomness_model.pdf(z)

def _compute_wi_yi(k, cur_season_index, xk, memo, randomness_model) -> tuple:
    """
    Integrate the values of f and g functions from period k+1 over epsilon for x_k

    returns:
    [(w1, y1), (w2, y2), ...] for the single step optimization for period k
    """
    if k == opt_horizon-1:
        return [0], [0]
    
    else:
        next_x_dict = {}
        num_intervals = memo[k+1]['num_intervals']

        if isinstance(randomness_model, NormalRandomness):
            results = []
            for i in range(num_intervals):
                f_integral, _ = quad(lambda z: __integrand(z, i, 0, next_x_dict, memo, k, cur_season_index, xk, randomness_model), -np.inf, np.inf)
                g_integral, _ = quad(lambda z: __integrand(z, i, 1, next_x_dict, memo, k, cur_season_index, xk, randomness_model), -np.inf, np.inf)
                results.append((f_integral, g_integral))
        elif isinstance(randomness_model, DiscreteRandomness):
            results = []
            for i in range(num_intervals):
                f_integral = sum([__integrand(z, i, 0, next_x_dict, memo, k, cur_season_index, xk, randomness_model) for z in randomness_model.values])
                g_integral = sum([__integrand(z, i, 1, next_x_dict, memo, k, cur_season_index, xk, randomness_model) for z in randomness_model.values])
                results.append((f_integral, g_integral))

        # separate the results into w and y
        w_i = [result[0] for result in results]
        y_i = [result[1] for result in results]
            
        return w_i, y_i

def _generate_memo_discrete(x0, season_index0, randomness_model, ts_model) -> dict: #TODO: move ts_model related to ts model class and use it in the dp_optimizer
    """
    Generate a memoization dictionary containing the exact possible states for each time step,
    given the initial state and a discrete randomness model.

    :param x0: Initial state vector (l0, t0, s0).
    :param season_index0: Initial season index.
    :param randomness_model: An instance of DiscreteRandomness.
    :return: memo, a dictionary storing possible states at each time step.
    """
    memo = defaultdict(dict)
    initial_state = tuple(x0)
    memo[0][initial_state] = []
    memo[0]['num_intervals'] = None  # Will be updated later if needed

    # We need to keep track of the season index at each time step
    season_indices = [(season_index0 + k) % m for k in range(opt_horizon)]

    for k in range(opt_horizon - 1):
        cur_season_index = season_indices[k]
        memo[k + 1]['num_intervals'] = None  # Initialize for the next time step

        # Determine relevant seasons for period k
        # Relevant seasons are those that will be used in future periods
        remaining_periods = opt_horizon - k - 1  # Periods remaining after current period
        future_season_indices = [(cur_season_index + i) % m for i in range(remaining_periods + 1)]
        relevant_seasons = set(future_season_indices)
        mask = np.isin(np.arange(m), list(relevant_seasons), invert=True) # Mask for irrelevant seasons

        # Iterate over all states at time k
        for xk in memo[k]:
            if xk == 'num_intervals':
                continue
            xk_list = list(xk)
            # compute the mean next state when epsilon = 0
            next_x_mean = list(__hw_price_transition(xk_list, cur_season_index, 0))
            # For each possible value of epsilon, compute the next state
            for epsilon in randomness_model.values:
                next_x = list(__hw_price_transition(xk_list, cur_season_index, epsilon))
                # override the season parameters with next_x_mean if the season is not relevant
                next_x_mean[2:2+m] = np.array(next_x_mean[2:2+m], dtype=float)
                next_x[2:2+m] = np.array(next_x[2:2+m], dtype=float)
                # Now apply the np.where logic
                next_x[2:2+m] = np.where(mask, next_x_mean[2:2+m], next_x[2:2+m])
                next_x = tuple(np.round(next_x, decimals=2)) 
                if next_x not in memo[k + 1]:
                    memo[k + 1][next_x] = []
        
        print(k, len(memo[k]))
    return memo

def __generate_memo_normal(x0, season_index0, randomness_model, ts_model) -> dict: #TODO: get rid of season_index0 in all dp implementations later
    """
    Generate a dictionary to store the f and g function values for each xk state in each period
    Tailored for normal randomness models

    Assume DP optimization starts from season 0

    :param x0: initial ts states
    """
    memo = defaultdict(dict)

    # Calculate the min and max values for each state based on the randomness model and ts model
    state_ranges = ts_model.dp_generate_state_range(x0, randomness_model, opt_horizon)
    for k in range(opt_horizon):
        period_memo = []
        for min_max_tuple in state_ranges[k]: # for each state, e.g., level, trend, season
            min_state, max_state = min_max_tuple
            num_states = min(max_num_x_states, int((max_state - min_state) / min_x_step_size) + 1)
            if num_states > 1:
                state_values = np.linspace(min_state, max_state, num_states)
            else: # if there is only one state, use the mean value
                state_values = [np.mean([min_state, max_state])]
            state_values = np.round(state_values).astype(int)

            period_memo.append(state_values)
        
        # Create all combinations of the datapoints for each k
        for values in product(*period_memo):
            memo[k][(values)] = []
        # have a value counting number of intervals for each k
        memo[k]['num_intervals'] = None
    
    return memo


def _generate_memo(x0, season_index0, randomness_model, ts_model) -> dict:
    """
    Generate a dictionary to store the f and g function values for each xk state in each period

    :param x0: initial ts states
    """
    if isinstance(randomness_model, NormalRandomness):
        return __generate_memo_normal(x0, season_index0, randomness_model, ts_model)
    elif isinstance(randomness_model, DiscreteRandomness):
        return _generate_memo_discrete(x0, season_index0, randomness_model, ts_model)
    else:
        raise ValueError("Randomness model not supported")

def _solve_xk(args):
    """
    Solve the optimization problem for a given xk state; worker function for parallel processing
    """
    k, xk, memo, b_states, cur_season_index, randomness_model = args
    l, t, *s = xk
    pk = l + t + s[cur_season_index]
    
    # Compute w_i and y_i
    w_i, y_i = _compute_wi_yi(k, cur_season_index, xk, memo, randomness_model)
    
    temp_results = []
    for b in b_states:
        # Solve single-step optimization
        _, u_star, b_coef, intercept = _single_step_opt(b, w_i, y_i, pk)
        # Store results
        temp_results.append((xk, b, (b_coef, intercept), u_star))
    return temp_results


def dp_optimize(x0, season_index0, randomness_model, ts_model) -> dict:
    """
    Dynamic programming optimization

    params:
    x0: initial states, tuple of (l0, t0, s0)
    season_index0: initial season index
    opt_horizon: optimization horizon
    ts_model: time series model

    returns:
    memo: indexed by (k, xk), stores the f and g function values for the input states [(f1, g1), (f2, g2), ...]
    policy: indexed by (k, xk, b), stores the optimal control action for the input states
    """
    # Initialize the memo dictionary
    memo = _generate_memo(x0, season_index0, randomness_model, ts_model)

    # Initialize the policy dictionary
    policy = {}

    # initialize b states
    num_b_states = min(max_num_b_states, int(Me / max_b_step_size) + 1)
    b_states = np.linspace(0, Me, num_b_states)
    b_states = np.round(b_states).astype(int)

    # Solve the optimization problem for each xk state
    for k in reversed(range(opt_horizon)):
        cur_season_index = (season_index0 + k) % m
        arg_list = []
        for xk in memo[k].keys():
            if xk == 'num_intervals':
                continue
            arg_list.append((k, xk, memo, b_states, cur_season_index, randomness_model))
        
        print(k, len(memo[k]), len(arg_list))
        with Pool() as pool:
            print(pool._processes)
            results = pool.map(_solve_xk, arg_list)

        # Store the results in the memo and policy dictionaries
        temp_memo = []
        for result in results:
            for temp_result in result:
                xk, b, b_coef_intercept, u_star = temp_result
                policy[(k, xk, b)] = u_star
                temp_memo.append([xk, b, b_coef_intercept])
        # for each b, get the list of pairs of b_coef and intercept
        temp_memo = pd.DataFrame(temp_memo, columns=['xk', 'b', 'b_coef_intercept'])
        param_list = temp_memo.groupby('b')['b_coef_intercept'].apply(set).reset_index()
        # then drop duplicates in list, keep the useful b states
        param_list.drop_duplicates(subset='b_coef_intercept', inplace=True, keep='first')
        useful_b_states = param_list['b'].values
        # for each xk, get corresponding b_coef and intercept from temp_memo for the useful b states
        for xk in memo[k].keys():
            if xk == 'num_intervals':
                memo[k]['num_intervals'] = len(useful_b_states)
            else:
                for b in useful_b_states:
                    b_coef, intercept = temp_memo[(temp_memo['xk'] == xk) & (temp_memo['b'] == b)]['b_coef_intercept'].values[0]
                    memo[k][xk].append((b_coef, intercept))

    return memo, policy


if __name__ == '__main__':
    import time
    from SeriesOpt.utils import *
    import csv
    import json
    from SeriesOpt.data_processing.holt_winters import HW_model
    from SeriesOpt.data_processing.randomness_models import DiscreteRandomness

    Config.set_params({'Me': 2, 'Mc':1, 'Md':1, 'eta':0.9,
                   'opt_horizon': 6})
    
    ############ Discrete randomness; randomly generated prices ################
    # randomness_model = DiscreteRandomness(np.arange(-15, 15), [1/30]*30)
    # level = np.random.randint(-10, 30)
    # trend = np.random.randint(-2,3)
    # season = [int(x) for x in np.random.randint(-5, 30, 6)]
    # x0 = [level, trend, *season]
    # ts_instance = HW_model(6, level, trend, season,0)
    
    ############ Normal randomness; real prices ################
    randomness_model = NormalRandomness(Config.get_param('sigma'))
    season = 4
    wd = os.getcwd()
    price_pjm = pd.read_csv(os.path.dirname(wd)+'\\Data\\PJM.csv')
    # keep the price column only
    price_pjm['Date'] = pd.to_datetime(price_pjm['Date'])
    price_pjm = price_pjm[['Date',' Zonal COMED price']].set_index('Date')[' Zonal COMED price'].asfreq('H')
    # split into train and test
    price_train = price_pjm[price_pjm.index.year!=2018]
    price_test = price_pjm[price_pjm.index.year==2018]

    price_train = [price_train.iloc[i*24:(i+1)*24] for i in range(len(price_train)//24)]
    price_test = [price_test.iloc[i*24:(i+1)*24] for i in range(len(price_test)//24)]
    # find the optimal segmentation for each 24-hour period based on the training data
    segments = load_data.find_opt_season_group(price_train, season)
    # aggregate the training and testing data into segments
    price_train = load_data.aggregate_prices(price_train, segments)
    price_test = load_data.aggregate_prices(price_test, segments)
    
    hw_model = HW_model(season)
    hw_model.fit(price_train, hyperparams={'alpha': 0.01, 'beta': 0.055, 'gamma': 1.0})
    x0 = [hw_model.cur_l, hw_model.cur_d, *hw_model.cur_s]

    b0 = 0

    # Run the DP optimizer to solve for optimal policy
    start = time.time()
    # Initialize the memo dictionary
    memo, policy = dp_optimize(x0,0,randomness_model, hw_model)
    end = time.time()
    print(f"DP optimizer took {end-start} seconds to run")
    # randomly generate a x0 and solve for the optimal policy; do this for 50 times
    # for i in range(30):
    #     level = np.random.randint(-10, 30)
    #     trend = np.random.randint(-2,3)
    #     season = [int(x) for x in np.random.randint(-5, 30, 4)]
    #     x0 = [level, trend, *season]
    #     print(f"Problem {i}: level = {level}, trend = {trend}, season = {season}")
    #     ts_instance = HW_model(4, level, trend, season,0)

    #     # Run the DP optimizer to solve for optimal policy
    #     start = time.time()
    #     # Initialize the memo dictionary
    #     memo, policy = dp_optimize(x0,0,randomness_model)
    #     end = time.time()
    #     print(f"Problem {i}: DP optimizer took {end-start} seconds to run")
        
    #     # Save the memo and policy to a file
    #     # Writing the policy dictionary into a CSV file
    #     with open(get_results_path(f'dp_experiment_{i}_H12.csv'), mode='w', newline='') as file:
    #         writer = csv.writer(file)
            
    #         # Write the header (adjust this based on your key structure)
    #         writer.writerow(['period', 'level', 'trend', 'season1', 'season2', 'season3', 'season4', 'storage', 'control'])
            
    #         # Write each key-value pair into the CSV
    #         for key, value in policy.items():
    #             key1, key2, key3 = key  # Unpacking the main tuple
    #             writer.writerow([key1, *key2, key3, value])

    #     # write the metadata for the csv file
    #     metadata = {'opt_horizon': opt_horizon,
    #                 'x0': x0,
    #                 'b0': b0,
    #                 'Me': Me,
    #                 'Mc': Mc,
    #                 'randomness_type': [int(x) for x in randomness_model.values],
    #                 'randomness_prob': [float(x) for x in randomness_model.probabilities]}

    #     with open(get_results_path(f'dp_experiment_{i}_H12_metadata.json'), mode='w') as json_file:
    #         json.dump(metadata, json_file, indent=4)