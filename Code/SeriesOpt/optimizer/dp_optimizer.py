import pandas as pd
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from itertools import product
from collections import defaultdict
from ..config import Config

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
max_x_step_size = Config.get_param('max_x_step_size')
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

    def _solve_sub_problem(wi, yi, lower_u, upper_u, coef, b):
        """
        coef: p/eta or p*eta
        """

        # compute potential v and corresponding coeffiyients of b and u values
        # in order, store optimal value, optimal control, b coef of v function, intercept of v function
        if upper_u == Me - b:
            min_pos = [[(wi_i - coef)*upper_u + b*wi_i + yi_i, upper_u, 
                        coef, yi_i+(wi_i-coef)*Me] for wi_i, yi_i in zip(wi, yi) if wi_i - coef > 0]
        else:
            min_pos = [[(wi_i - coef)*upper_u + b*wi_i + yi_i, upper_u, 
                        wi_i, yi_i+(wi_i-coef)*upper_u] for wi_i, yi_i in zip(wi, yi) if wi_i - coef > 0]
        
        if lower_u == -b:
            min_neg = [[(wi_j - coef)*lower_u + b*wi_j + yi_j, lower_u, coef, yi_j] for wi_j, yi_j in zip(wi, yi) if wi_j - coef < 0]
        else:
            min_neg = [[(wi_j - coef)*lower_u + b*wi_j + yi_j, lower_u, 
                        wi_j, yi_j+(wi_j-coef)*lower_u] for wi_j, yi_j in zip(wi, yi) if wi_j - coef < 0]
        
        cross_terms = [
        [b*coef + ((wi_i - coef) * yi_j - (wi_j - coef) * yi_i) / (wi_i - wi_j),
        -b + (yi_j - yi_i) / (wi_i - wi_j),
         coef, ((wi_i - coef) * yi_j - (wi_j - coef) * yi_i) / (wi_i - wi_j)]
        for wi_i, yi_i in zip(wi, yi) if wi_i - coef > 0
        for wi_j, yi_j in zip(wi, yi) if wi_j - coef <= 0
    ]

        comwined = min_pos + min_neg + cross_terms
        
        # Find the minimum first element and corresponding second element
        v_star, _, corresponding_b_coef, corresponding_intercept = min(comwined, key=lambda x: x[0])
        # Do the following because when there is w_i - coef = 0, the corresponding u value is not unique
        if cross_terms:
            u_star = max(lower_u, min(upper_u, min(cross_terms, key=lambda x: x[0])[1]))
        else:
            u_star = _
        
        return v_star, u_star, corresponding_b_coef, corresponding_intercept
    
    # Solving sub-problems
    v_star_pos, u_star_pos, b_coef_pos, intercept_pos = _solve_sub_problem(w_i, y_i, 0, upper_u, p/eta, b)
    # print(v_star_pos, u_star_pos, b_coef_pos)
    v_star_neg, u_star_neg, b_coef_neg, intercept_neg = _solve_sub_problem(w_i, y_i, lower_u, 0, p*eta, b)
    # print(v_star_neg, u_star_neg, b_coef_neg)
    
    # Optimal solution
    if v_star_pos > v_star_neg:
        v_star = v_star_pos
        u_star = u_star_pos
        b_coef = b_coef_pos
        intercept = intercept_pos
    elif v_star_pos < v_star_neg:
        v_star = v_star_neg
        u_star = u_star_neg
        b_coef = b_coef_neg
        intercept = intercept_neg
    else:
        # if the two values are equal, when b=Me, choose the positive side value because you can only approximate
        # Me from charging side, and when b=0, choose the negative side value; 
        # otherwise, choose the positive side value b/c the coefficient and intercept would be the same
        if b == 0:
            v_star = v_star_neg
            u_star = u_star_neg
            b_coef = b_coef_neg
            intercept = intercept_neg
        else:
            v_star = v_star_pos
            u_star = u_star_pos
            b_coef = b_coef_pos
            intercept = intercept_pos

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
    t_new = t + alpha * beta * epsilon
    l_new = l + t + alpha * epsilon
    s[cur_season_index] = s[cur_season_index] + gamma*(epsilon + t)
    
    return (l_new, t_new, *s)

def __integrand(z, interval, component, next_x_dict, memo, k, cur_season_index, xk):
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
    return func_value * norm.pdf(z, scale=sigma)

def _compute_wi_yi(k, cur_season_index, xk, memo) -> tuple:
    """
    Integrate the values of f and g functions from period k+1 over epsilon for x_k

    returns:
    [(w1, y1), (w2, y2), ...] for the single step optimization for period k
    """
    if k == opt_horizon-1:
        return [0], [0]
    
    else:
        results = []
        next_x_dict = {}
    
        for i in range(memo[k+1]['num_intervals']):
            f_integral, _ = quad(lambda z: __integrand(z, i, 0, next_x_dict, memo, k, cur_season_index, xk), -np.inf, np.inf)
            g_integral, _ = quad(lambda z: __integrand(z, i, 1, next_x_dict, memo, k, cur_season_index, xk), -np.inf, np.inf)
            results.append((f_integral, g_integral))
        
        # separate the results into w and y
        w_i = [result[0] for result in results]
        y_i = [result[1] for result in results]
            
        return w_i, y_i

def _generate_memo(x0, season_index0) -> dict:
    """
    Generate a dictionary to store the f and g function values for each xk state in each period

    :param x0: initial ts states
    """
    memo = defaultdict(dict)
    l0, t0, s0 = x0[0], x0[1], x0[2:]

    for k in range(opt_horizon):
        l_variance_term = np.sqrt(beta**2 * (k*(k-1)*(2*k-1)/6) + beta * k*(k-1) + k) * alpha * sigma
        lmin = l0 + k * t0 - 2 * l_variance_term
        lmax = l0 + k * t0 + 2 * l_variance_term
        num_l_states = min(max_num_x_states, int((lmax - lmin) / max_x_step_size) + 1)
        if num_l_states > 1: 
            l_values = np.linspace(lmin, lmax, num_l_states)
        else: # if there is only one state, use the mean value
            l_values = [l0 + k * t0]
        l_values = np.round(l_values).astype(int)

        t_variance_term = np.sqrt(k) * alpha * beta * sigma
        tmin = t0 - 2 * t_variance_term
        tmax = t0 + 2 * t_variance_term
        num_t_states = min(max_num_x_states, int((tmax - tmin) / max_x_step_size) + 1)
        if num_t_states > 1:
            t_values = np.linspace(tmin, tmax, num_t_states)
        else:
            t_values = [t0]
        t_values = np.round(t_values).astype(int)

        s_values = []
        for i in range(0, m):
            r = np.floor((k - 1 - ((i - season_index0 + m) % m)) / m) + 1 # number of updates of season index i as of period k
            variance_term_s = np.sqrt(r + alpha**2 * beta**2 * (r * ((i - season_index0 + m) % m) + (m / 2) * r * (r - 1))) * sigma
            smin = s0[i] + t0 * r - 2 * variance_term_s
            smax = s0[i] + t0 * r + 2 * variance_term_s
            num_s_states = min(max_num_x_states, int((smax - smin) / max_x_step_size) + 1)
            if num_s_states > 1:
                s_values.append(np.linspace(smin, smax, num_s_states))
            else:
                s_values.append([s0[i] + t0 * r])
        s_values = [np.round(s).astype(int) for s in s_values]

        # Create all combinations of the datapoints for each k
        for values in product(l_values, t_values, *s_values):
            memo[k][(values)] = []
        # have a value counting number of intervals for each k
        memo[k]['num_intervals'] = None
    
    return memo

def dp_optimize(x0, season_index0) -> dict:
    """
    Dynamic programming optimization

    params:
    x0: initial states, tuple of (l0, t0, s0)
    season_index0: initial season index
    opt_horizon: optimization horizon

    returns:
    memo: indexed by (k, xk), stores the f and g function values for the input states [(f1, g1), (f2, g2), ...]
    policy: indexed by (k, xk, b), stores the optimal control action for the input states
    """
    # Initialize the memo dictionary
    memo = _generate_memo(x0, season_index0)

    # Initialize the policy dictionary
    policy = {}

    # initialize b states
    num_b_states = min(max_num_b_states, int(Me / max_b_step_size) + 1)
    b_states = np.linspace(0, Me, num_b_states)
    b_states = np.round(b_states).astype(int)

    # Solve the optimization problem for each xk state
    for k in reversed(range(opt_horizon)):
        cur_season_index = (season_index0 + k) % m
        temp_memo = []
        for xk in memo[k].keys():
            if xk == 'num_intervals':
                continue
            else:
                # Unpack the state variables
                l, t, s = xk[0], xk[1], xk[2:]
                pk = l + t + s[cur_season_index] #TODO: consider making this more general later to suit other price models
                # compute the slopes and intercepts for the J function
                w_i, y_i = _compute_wi_yi(k, cur_season_index, xk, memo)

                for b in b_states:
                    # solve the single-step optimization problem
                    _, u_star, b_coef, intercept = _single_step_opt(b, w_i, y_i, pk)
                    policy[(k, xk, b)] = u_star
                    temp_memo.append([xk, b, (b_coef, intercept)])
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
    # start = time.time()
    # # Initialize the memo dictionary
    # memo, policy = dp_optimize([30, 0, -30, 10, -40, -20],0)
    # # # Initialize the policy dictionary
    # # policy = {}
    # # # Solve the optimization problem for each xk state
    # # _solve_xk(0, 3, 0, [100, 0, 0, 0, 0, 0, 0], memo, policy)
    # # print(policy)

    # # print memo keys in separate lines, one key per line
    # for subkey, subitem in memo[4].items():
    #     print(subkey, subitem)
    
    # # # print policy keys in separate lines, one key per line
    # # for key, item in policy.items():
    # #     print(key, item)
    
    # print(f"Time taken: {time.time()-start} seconds")
    _single_step_opt(4, [76.5, 0, 0],[0, 382.5, 382.5],80)