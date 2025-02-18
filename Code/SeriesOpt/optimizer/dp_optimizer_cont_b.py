import math
import copy
import pandas as pd
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from scipy.spatial import cKDTree
from itertools import product
from collections import defaultdict
from SeriesOpt.config import Config
from SeriesOpt.data_processing.randomness_models import *
from SeriesOpt.data_processing.holt_winters import HW_model
from SeriesOpt.data_processing import load_data
from multiprocessing import Pool
import os
import itertools
import pickle

Me = Config.get_param('Me')
Mc = Config.get_param('Mc')
Md = Config.get_param('Md')
eta = Config.get_param('eta')
max_num_x_states = Config.get_param('max_num_x_steps')
min_x_step_size = Config.get_param('min_x_step_size')
opt_horizon = Config.get_param('opt_horizon')

############################################################
# 1. Data structure for a PW-linear function in b
############################################################

class PiecewiseLinearFunction:
    """
    Represents a function f(b) that is piecewise linear on [0, B_max].
    Internally stored as a sorted list of segments:
        segments = [
          (b_left, b_right, slope, intercept),
          ...
        ]
    meaning that for b in [b_left, b_right],
       f(b) = slope * b + intercept.

    - Segments must be contiguous in b, with b_right of segment i
      = b_left of segment i+1 (except possibly for the last).
    - We assume no overlaps or gaps in the domain, and b_left < b_right.
    """

    def __init__(self, segments=None):
        if segments is None:
            self.segments = []
        else:
            # If you like, you can sort or validate here
            self.segments = segments

    def evaluate(self, b):
        """
        Evaluate the piecewise linear function at a scalar b.
        Assumes b is within the domain [segments[0].b_left, segments[-1].b_right].
        """
        # simple linear search (or you could do binary search)
        for (bL, bR, slope, intercept) in self.segments:
            if bL <= b <= bR:
                return slope * b + intercept
        # If b is out of range, handle or raise an error:
        raise ValueError(f"b={b} out of domain")

    def copy(self):
        return PiecewiseLinearFunction(segments=copy.deepcopy(self.segments))

    def __repr__(self):
        seg_strs = []
        for (bL, bR, slope, intercept) in self.segments:
            seg_strs.append(f"[{bL}, {bR}]: {slope} * b + {intercept}")
        return "PiecewiseLinearFunction(\n  " + "\n  ".join(seg_strs) + "\n)"


############################################################
# 2. Helper functions
############################################################

def _immediate_val_func(p, mode):
    """
    Immediate-cost helper: charging or discharging.
    u >= 0 -> charging revenue = - p * (u/eta)
    u <= 0 -> discharging revenue = - p * (u * eta)
    """
    if mode == 'charge':
        return -p / eta
    elif mode == 'discharge':
        return -p * eta
    else:
        raise ValueError("Invalid mode: {}".format(mode))

############################################################
# 3. Combine a bunch of J_{k+1} for different x_k+1 into 
# an expected J_{k+1} piecewise linear function
# similar to _compute_wi_yi in dp_optimizer.py
############################################################

def _build_state_kdtree(memo_kplus1):
    """
    Build a cKDTree from the keys of memo[k+1]. Used for fast nearest-neighbor lookup.
    Returns:
      kdtree:  a scipy.spatial.cKDTree object
      state_list: an array of shape (Nstates, dim)
      index_to_state: a list or dict mapping row index -> the actual state tuple
    """
    # 1) gather all states
    index_to_state = []
    for idx, state_tuple in enumerate(memo_kplus1.keys()):
        index_to_state.append(state_tuple)

    # 2) convert to np array
    state_array = np.array(index_to_state, dtype=float)  # shape (Nstates, dim)
    
    # 3) build KD-tree
    kdtree = cKDTree(state_array)
    
    return kdtree, index_to_state

def _combine_func(scenarios):
    """
    Combine a list of scenarios into a single expected PWL function.
    scenarios: a list of (weight, PWL) tuples.
    """
    # 1) Gather all the breakpoints
    all_breakpoints = set()
    for (weight, pwl) in scenarios:
        for (bL, bR, _, _) in pwl.segments:
            all_breakpoints.add(bL)
            all_breakpoints.add(bR)
    all_breakpoints = sorted(list(all_breakpoints))

    # 2) For each segment, compute the expected slope and intercept
    new_segments = []
    for i in range(len(all_breakpoints) - 1):
        bL, bR = all_breakpoints[i], all_breakpoints[i+1]
        slope_sum, intercept_sum = 0.0, 0.0
        for (weight, pwl) in scenarios:
            for (seg_bL, seg_bR, slope, intercept) in pwl.segments:
                if seg_bL <= bL and seg_bR >= bR:
                    # This segment covers [bL, bR]
                    slope_sum += weight * slope
                    intercept_sum += weight * intercept
                    break  # move to next scenario
        new_segments.append((bL, bR, slope_sum, intercept_sum))

    return PiecewiseLinearFunction(segments=new_segments)

def _interpolate_pwl_across_neighbors(x_kplus1_cont, 
                                      kdtree, 
                                      index_to_state, 
                                      memo_kplus1, 
                                      k=3):
    """
    Finds the k nearest discrete states to x_kplus1_cont, retrieves their 
    piecewise-linear cost-to-go, then merges (interpolates) them into a 
    single PWL function using distance-based weights.

    Returns: A PiecewiseLinearFunction instance representing the 
             weighted combination of the neighbors' PWL.
    """
    # 1) Query k neighbors
    # distances: shape (k,)
    # nn_indices: shape (k,) - indices into index_to_state
    distances, nn_indices = kdtree.query(x_kplus1_cont, k=k)  
    # If k=1, they are scalars. If k>1, arrays. Ensure they are arrays:
    if not hasattr(distances, '__len__'):
        # Means k=1 was used
        distances = np.array([distances])
        nn_indices = np.array([nn_indices])
    
    # 2) Compute interpolation weights (inverse-distance or similar)
    #    If any distance=0, to avoid divide-by-zero, 
    #    you might handle separately or add small epsilon.
    eps = 1e-8
    inv_d = 1.0 / (distances + eps)
    w_sum = np.sum(inv_d)
    neighbor_weights = inv_d / w_sum
    
    # 3) Retrieve each neighbor's PWL function
    neighbor_pwls = []
    for (idx_n, w) in zip(nn_indices, neighbor_weights):
        x_kplus1_disc = index_to_state[idx_n]
        segments = memo_kplus1[x_kplus1_disc]
        # Convert to your PWL class (assuming you have a constructor like below)
        neighbor_pwls.append((w, PiecewiseLinearFunction(segments=segments)))
    
    # 4) Combine them into one PWL using the same approach as _combine_func
    combined_pwl = _combine_func(neighbor_pwls)  # re-use the logic from your scenario combiner

    return combined_pwl

def _build_expected_pwl(
    xk, 
    memo_kplus1,   # dict: (x_{k+1} tuple) -> PiecewiseLinearFunction
    randomness_model,
    next_state_func,
    num_samples,
    ts_args=None,
    k_neighbors=3
):
    """
    Construct the PWL function for E[J_{k+1}(b, x_{k+1}(\epsilon))],
    using a sample-based approach and *k*-NN interpolation.

    Steps:
      1) Generate eps samples from randomness_model
      2) For each eps, compute x_{k+1} = next_state_func(xk, eps)
      3) Get *k* nearest neighbors in memo_kplus1, and interpolate them 
         to form a single PWL
      4) Weighted sum (by PDF or 1/num_samples) across samples to get an 
         expected PWL (via _combine_func).
    """
    # 1) Sample epsilons
    eps_samples = randomness_model.sample(num_samples)
    
    # 2) Probability weights for each sample
    pdf_values = [randomness_model.pdf(e) for e in eps_samples]
    pdf_sum = sum(pdf_values)

    # 3) Build KD-tree
    kdtree, index_to_state = _build_state_kdtree(memo_kplus1)

    # We'll collect (weight, PWL) for each scenario
    scenario_list = []

    for i, eps_val in enumerate(eps_samples):
        # 3a) Continuous next-state
        x_kplus1_cont = next_state_func(xk, eps_val, ts_args)

        # 3b) Interpolate among k neighbors
        pwl_next = _interpolate_pwl_across_neighbors(
            x_kplus1_cont,
            kdtree,
            index_to_state,
            memo_kplus1,
            k=k_neighbors
        )

        # 3c) Probability weight for this scenario
        weight_i = pdf_values[i] / pdf_sum
        scenario_list.append((weight_i, pwl_next))

    # 4) Combine scenario PWLs into one expected PWL
    Jkplus1_expected = _combine_func(scenario_list)
    return Jkplus1_expected

############################################################
# 4. Single step optimization for J_k
############################################################

def _solve_subproblem(Jkplus1Seg, p, mode, Mc_k, Md_k):
    """
    A unified function that handles EITHER the 'charging' or 'discharging' 
    sub-problem in a 'max' objective context.

    Within the segment b+u ∈ [bnext_L, bnext_R], J_{k+1}(b+u) = w*(b+u) + y.
    The value function is:
       F(b,u) = immediate_val + w*(b+u) + y, 
    with  u ∈ [0, Mc_k] or [-Md_k, 0], and  b+u ∈ [bnext_L, bnext_R],  b ∈ [0, Me],  b+u ≤ Me.

    We'll figure out the slope wrt u => w + immediate_val.
    If slope > 0 => we want u = upper boundary.
    If slope < 0 => we want u = lower boundary.
    If slope=0 => any feasible u => pick 0 for simplicity.

    Then we form a piecewise function in b. 

     Returns a list of piecewise segments in b:
       [ (bL, bR, slope_b, intercept_b, "somePolicyLabel"), ... ]

    Each segment is valid only where (b+u^*(b)) stays in [bnext_L, bnext_R].
    """
    bnext_L, bnext_R, w, y = Jkplus1Seg
    # F(b,u) = immediate_val*u + w*(b+u) + y
    im_val = _immediate_val_func(p, mode)
    slope_u = w + im_val
    pieces = [] # list of segments in b

    if mode == 'charge':
        # Charging: slope_u >= 0, choose feasible upper bound: the min of (u=Mc_k) or (u=Me-b) or (u=bnext_R-b)
        # implicitly there should be bnext_R <= Me, so just consider (u=Mc_k) or (u=bnext_R-b)
        if slope_u > 0: 
            # CASE 1: bnext_R-b < Mc_k => upper = bnext_R-b => b >= bnext_R-Mc_k
            bL = max(0, bnext_R - Mc_k)
            bR = bnext_R
            if bR >= bL:
                # F(b,u) = immediate_val*(Me-b) + w*(b+Me-b) + y
                # derivative of F(b,u) wrt b = - immediate_val
                slope_b = -im_val
                intercept = w*bnext_R + y + im_val*bnext_R
                # policy_label = "u=bnext_R-b"
                slope_u_star = -1 # slope and intercept of u* wrt b
                intercept_u_star = bnext_R
                pieces.append((bL, bR, slope_b, intercept, slope_u_star, intercept_u_star))
            # CASE 2: bnext_R-b >= Mc_k => upper = Mc_k => b <= bnext_R-Mc_k
            bL = max(0, bnext_L - Mc_k)   # for b+Mc_k>=bnext_L
            bR = min(Me, bnext_R - Mc_k)  # for b+Mc_k<=bnext_R and b<=bnext_R-Mc_k
            if bR >= bL:
                # F(b,u) = immediate_val*Mc_k + w*(b+Mc_k) + y
                # derivative of F(b,u) wrt b = w
                slope_b = w
                intercept = (im_val+w)*Mc_k + y
                # policy_label = "u=Mc_k"
                slope_u_star = 0
                intercept_u_star = Mc_k
                pieces.append((bL, bR, slope_b, intercept, slope_u_star, intercept_u_star))
        else:
            # Charging: slope_u < 0, choose lower bound (u=0)
            bL = max(0, bnext_L)   # for b+0>=bnext_L
            bR = min(Me, bnext_R)  # for b+0<=bnext_R
            if bR >= bL:
                # F(b,u) = immediate_val*0 + w*b + y
                # derivative of F(b,u) wrt b = w
                slope_b = w
                intercept = y
                # policy_label = "u=0"
                slope_u_star = 0
                intercept_u_star = 0
                pieces.append((bL, bR, slope_b, intercept, slope_u_star, intercept_u_star))
    
    elif mode == 'discharge':
        # Discharging: slope_u >0, choose upper bound (u=0)
        if slope_u > 0:
            bL = max(0, bnext_L)   # for b+0>=bnext_L
            bR = min(Me, bnext_R)  # for b+0<=bnext_R
            if bR >= bL:
                # F(b,u) = immediate_val*0 + w*b + y
                # derivative of F(b,u) wrt b = w
                slope_b = w
                intercept = y
                # policy_label = "u=0"
                slope_u_star = 0
                intercept_u_star = 0
                pieces.append((bL, bR, slope_b, intercept, slope_u_star, intercept_u_star))
        # Discharging: slope_u <= 0, choose feasible lower bound: the max of (u=-Md_k) or (u=-b) or (u=bnext_L-b)
        # implicitly there should be bnext_L >= 0, so just consider (u=-Md_k) or (u=bnext_L-b)
        else:
            # CASE 1: b-bnext_L < Md_k => lower = bnext_L-b 
            bL = bnext_L   # for u=bnext_L-b<=0
            bR = min(Me, bnext_L + Md_k)
            if bR >= bL:
                # F(b,u) = immediate_val*(bnext_L-b) + w*(b+bnext_L-b) + y
                # derivative of F(b,u) wrt b = - immediate_val
                slope_b = -im_val
                intercept = (im_val+w)*bnext_L + y
                # policy_label = "u=bnext_L-b"
                slope_u_star = -1
                intercept_u_star = bnext_L
                pieces.append((bL, bR, slope_b, intercept, slope_u_star, intercept_u_star))
            # CASE 2: b-bnext_L >= Md_k => lower = -Md_k
            bL = max(0, bnext_L + Md_k)   # for b-Md_k>=bnext_L
            bR = min(Me, bnext_R + Md_k)  # for b-Md_k<=bnext_R
            if bR >= bL:
                # F(b,u) = immediate_val*(-Md_k) + w*(b-Md_k) + y
                # derivative of F(b,u) wrt b = w
                slope_b = w
                intercept = (im_val+w)*(-Md_k) + y
                # policy_label = "u=-Md_k"
                slope_u_star = 0
                intercept_u_star = -Md_k
                pieces.append((bL, bR, slope_b, intercept, slope_u_star, intercept_u_star))
            
    return pieces

def _build_Jk_from_Jkplus1(Jkplus1_expected, p, Mc_k, Md_k):
    """
    For each segment in Jkplus1_expected, solve both the charging and discharging subproblem;
    combine the results to build the upper envelope of J_k.
    
    Build the upper envelope of a list of piecewise line segments in b.

    Inputs
    ------
    Jkplus1_expected: a PiecewiseLinearFunction in b, representing E[J_{k+1}(b)].
    
    First generate list of (bL, bR, slope, intercept, policy_label)
      meaning each segment is valid for b in [bL, bR], 
      and the line = slope * b + intercept.

    Returns
    -------
    A new list of merged segments: [ (bL,bR, slope, intercept, policy_label), ... ]
    representing the envelope on each sub-interval.
    """
    all_candidates = []
    for segment in Jkplus1_expected.segments:
        charge_pieces = _solve_subproblem(segment, p, mode="charge", Mc_k=Mc_k, Md_k=Md_k)
        all_candidates.extend(charge_pieces)
        discharge_pieces = _solve_subproblem(segment, p, mode="discharge", Mc_k=Mc_k, Md_k=Md_k)
        all_candidates.extend(discharge_pieces)
    
    # Now find the upper envelope of all_candidates
    # 1) Gather all breakpoints from the input segments
    bPoints = set()
    for (bL, bR, _, _, _, _) in all_candidates:
        bPoints.add(bL)
        bPoints.add(bR)
    sorted_b = sorted(bPoints)

    # 2) For each segment, compute the expected slope and intercept
    new_segments = []
    for i in range(len(sorted_b) - 1):
        left, right = sorted_b[i], sorted_b[i+1]
        # pick a test point in the sub-interval
        mid = (left + right) / 2
        # among all segments that cover [left,right], 
        # pick whichever line is largest (if mode="max") or smallest (if mode="min") at mid.
        best_line = None
        best_val = -math.inf
        for (bL, bR, slope, intercept, slope_u_star, intercept_u_star) in all_candidates:
            if bL <= mid <= bR:
                val = slope * mid + intercept
                if val > best_val:
                    best_val = val
                    best_line = (slope, intercept, slope_u_star, intercept_u_star)
                # if valuation is the same, pick the one with larger slope. Hopefully this doesn't happen b/c we use mid point.
                elif val == best_val and slope > best_line[0]:
                    best_line = (slope, intercept, slope_u_star, intercept_u_star)
        
        if best_line is not None:
            new_segments.append((left, right, best_line[0], best_line[1], best_line[2], best_line[3]))
        
    # 3) merge adjacent segments with the same slope and intercept
    merged_segments_J = []
    merged_segments_policy = []
    # sort by left boundary
    new_segments.sort(key=lambda x: x[0])
    cur = new_segments[0]
    for i in range(1, len(new_segments)):
        if new_segments[i][2:] == cur[2:]:
            cur = (cur[0], new_segments[i][1], cur[2], cur[3], cur[4], cur[5])
        else:
            merged_segments_J.append((cur[0], cur[1], cur[2], cur[3]))
            merged_segments_policy.append((cur[0], cur[1], cur[4], cur[5]))
            cur = new_segments[i]
    
    merged_segments_J.append((cur[0], cur[1], cur[2], cur[3]))
    merged_segments_policy.append((cur[0], cur[1], cur[4], cur[5]))

    return merged_segments_J, merged_segments_policy
    
############################################################
# 5. DP Optimization
############################################################

def _generate_memo(x0, ts_model, randomness_model, opt_horizon):
    """
    Initialize dictionary to store the value function J_k(b) for each state tuple. and policy
    """
    memo = defaultdict(dict) # store the value function J_k(b) for each state tuple
    state_ranges = ts_model.dp_generate_state_range(x0, randomness_model, opt_horizon) # list (period) of list of states
    for k in range(opt_horizon):
        state_keys_k = []
        # create states within ranges
        ## the min distance between two states within the same dimension is min_x_step_size
        ## the max number of states in each dimension is max_num_x_states
        # for dim in state_ranges[k]:
        #     num_states = min(max_num_x_states, int((dim[1] - dim[0]) / min_x_step_size) + 1)
        #     dim_states = np.linspace(dim[0], dim[1], num_states)
        #     state_keys_k.append(dim_states)
        # # combine to get all state tuples
        # state_keys_k = list(product(*state_keys_k))
        # for efficient implementation below
        state_keys_k = (
            itertools.product(*(
                np.linspace(dim[0], dim[1], min(max_num_x_states, int((dim[1] - dim[0]) / min_x_step_size) + 1))
                for dim in state_ranges[k]
            ))
        )
        memo[k] = {state: PiecewiseLinearFunction() for state in state_keys_k}
    
    policy = copy.deepcopy(memo) # store the optimal policy for each state tuple

    return memo, policy

# solve all xk states through parallel processing
def _process_state(k, state, memo, randomness_model, ts_model, num_samples, ts_args, Mc_k, Md_k):
    """
    helper function that solves the optimization problem for a given state xk
    
    Parameters:
    - k: current period
    - state: state tuple
    - memo: memoization table for Jk
    - randomness_model: randomness model object
    - ts_model: time series model object
    - num_samples: number of samples for expected PWL
    - ts_args: additional arguments for ts_model
    """
    # build J_{k+1} expected function
    if k == opt_horizon -  1: # special case for terminal condition
        Jkplus1_expected = PiecewiseLinearFunction([(0, Me, 0, 0)])
    else:
        Jkplus1_expected = _build_expected_pwl(
            state, 
            memo[k+1], 
            randomness_model, 
            ts_model.dp_func_transition, 
            num_samples,
            ts_args
        )
    # build J_k function
    p = ts_model.forecast(1, state, ts_args)[0] # 1 for current period forecast
    Jk, policy_k = _build_Jk_from_Jkplus1(Jkplus1_expected, p, Mc_k, Md_k)
    return state, Jk, policy_k


def dp_optimize_cont_b(x0, ts_model, randomness_model, num_samples, Mc_set=None, Md_set=None):
    """
    Perform dynamic programming optimization for continuous battery levels.

    Parameters:
    - x0: initial state (including initial battery level)
    - ts_model: time series model object with methods:
        - dp_generate_state_range(init_state, randomness_model, opt_horizon): returns list of list of states
        - forecast(xk): returns p for given state xk
        - next_state_func(xk, u, randomness_model): returns next state xk+1
        - opt_horizon: number of periods
    - randomness_model: randomness model object with attributes:
    - num_samples: number of samples for expected PWL
    - Mc_set: list of charging power limits if varying over time; default to Mc
    - Md_set: list of discharging power limits if varying over time; default to Md
    
    Returns:
    - policy_sequence: list of optimal controls (u*) from k=0 to k=opt_horizon-1
    """
    print(f"max_num_x_states is {max_num_x_states}")
    print(f"Me is {Me}")
    # 0. sepcify Mc and Md sets
    if Mc_set is None:
        Mc_set = [Mc] * opt_horizon
    if Md_set is None:
        Md_set = [Md] * opt_horizon
    # 1. initialization
    memo, policy = _generate_memo(x0, ts_model, randomness_model, opt_horizon)
    # 2. backward induction
    for k in reversed(range(opt_horizon)):
        # ts model specific arguments
        if hasattr(ts_model, 'cur_season_index'):
            ts_args = k % ts_model.m # season index
        ## TODO: arguments for other models
        else:
            ts_args = None
        print(f"Solving period {k}...")
        state_keys = list(memo[k].keys())
        arg_list = [(k, state, memo, randomness_model, ts_model, num_samples, ts_args, Mc_set[k], Md_set[k]) for state in state_keys]
        with Pool() as pool:
            print(pool._processes)
            results = pool.starmap(_process_state, arg_list)

        for state, Jk, policy_k in results:
            # print(Jk)
            memo[k][state] = Jk
            # print(memo[k][state])
            policy[k][state] = policy_k
            # print(memo[k][state])
        print(f"Period {k} solved.")

    return policy, memo

def save_policy(policy, file_name):
    """
    Save the policy to a file.
    """
    with open(file_name, 'wb') as f:
        pickle.dump(policy, f)

############################################################
# 6. Apply DP policy to a time series
############################################################

def apply_dp(real_prices, ts_model, dp_policy, b):
    """
    Apply the DP policy to a price series.

    Parameters:
    - real_prices: a pandas Series of real prices
    - ts_model: time series model object with methods:
        - initialized with historical data right before the optimization period
    - dp_policy: the policy returned by dp_optimize_cont_b
    - b: initial battery level

    Returns:
    - optimal_controls: a pandas Series of optimal controls
    """
    # make elements in DP policy to be PiecewiseLinearFunction
    for k in dp_policy.keys():
        for state in dp_policy[k].keys():
            dp_policy[k][state] = PiecewiseLinearFunction(segments=dp_policy[k][state])
    
    if hasattr(ts_model, 'cur_season_index'):
        xk = (ts_model.cur_l, ts_model.cur_d, *ts_model.cur_s)

    u_sequence = []
    profit_sequence = []
    b_sequence = []
    # check to ensure real_prices and dp_policy have the same length
    if len(real_prices) != len(dp_policy):
        raise ValueError("real_prices and dp_policy must have the same length")

    for k in range(len(real_prices)):
        price = real_prices[k]
        # Build the KD-tree for nearest neighbor search
        kdtree, index_to_state = _build_state_kdtree(dp_policy[k])
        # Find the nearest neighbor match for xk
        xk_match_idx = kdtree.query(xk)[1]
        xk_match = index_to_state[xk_match_idx]
        # Retrieve the optimal control
        # print(dp_policy)
        u = dp_policy[k][xk_match].evaluate(b)
        # Record the control
        u_sequence.append(u)
        # Record profit
        profit = max(u * eta, u / eta) * -price
        profit_sequence.append(profit)
        # Update the state
        b += u
        b_sequence.append(b)
        ts_model.update([price])
        if hasattr(ts_model, 'cur_season_index'):
            xk = (ts_model.cur_l, ts_model.cur_d, *ts_model.cur_s)
    
    return profit_sequence, u_sequence, b_sequence


############################################################
# 7. A small DEMO
############################################################

if __name__ == "__main__":
    import time
    from SeriesOpt.utils import *
    import csv
    import json
    from SeriesOpt.data_processing.holt_winters import HW_model
    from SeriesOpt.data_processing.randomness_models import DiscreteRandomness

    Config.set_params({'Me': 2, 'Mc':1, 'Md':1, 'eta':0.9,
                   'opt_horizon': 12})
    
    ############ Normal randomness; real prices ################
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

    price_train = [price_train.iloc[i*24:(i+1)*24] for i in range(len(price_train)//24)]
    price_test = [price_test.iloc[i*24:(i+1)*24] for i in range(len(price_test)//24)]
    # find the optimal segmentation for each 24-hour period based on the training data
    segments = load_data.find_opt_season_group(price_train, season)
    print(segments)
    # create Mc and Md sets based on segment length
    Mc_set = [min(Me,seg[2]*Mc) for seg in segments]
    Md_set = [min(Me,seg[2]*Md) for seg in segments]
    print(Mc_set)
    print(Md_set)
    # aggregate the training and testing data into segments
    price_train = load_data.aggregate_prices(price_train, segments)
    price_test = load_data.aggregate_prices(price_test, segments)

    # # save aggregated prices
    # pd.DataFrame(price_train).to_csv(os.path.dirname(wd)+'\\Data\\PJM_train_agg.csv', index=False, header=False)
    # pd.DataFrame(price_test).to_csv(os.path.dirname(wd)+'\\Data\\PJM_test_agg.csv', index=False, header=False)

    # # load aggregated prices
    # price_train = pd.read_csv(os.path.dirname(wd)+'\\Data\\PJM_train_agg.csv', header=None)
    # price_test = pd.read_csv(os.path.dirname(wd)+'\\Data\\PJM_test_agg.csv', header=None)
    # # convert to numpy array
    # price_train = price_train.values.flatten()
    # price_test = price_test.values.flatten()

    ############### fit the HW model ################
    hw_model = HW_model(season)
    hw_model.fit(price_train, hyperparams={'alpha': 0.1, 'beta': 0.1, 'gamma': 0.275})
    x0 = [hw_model.cur_l, hw_model.cur_d, *hw_model.cur_s]
    randomness_model = EmpiricalKDERandomness(hw_model.residuals)
    # ############### Generate memoization table ###############
    # # memo, policy = _generate_memo(x0, hw_model, randomness_model, opt_horizon)
    # # for k in range(opt_horizon):
    # #     print("k=", k, "num_states=", len(memo[k]))
    # #     print(memo[k].keys())

    ################# DP optimization #################
    # turn on when needed
    start = time.time()
    policy, memo = dp_optimize_cont_b(x0, hw_model, randomness_model, num_samples=100, Mc_set=Mc_set, Md_set=Md_set)
    end = time.time()
    print("DP optimization time:", end-start)
    save_policy(policy, "SeriesOpt/tests/dp_cont_policy_12seg.pkl")
    save_policy(memo, "SeriesOpt/tests/dp_cont_memo_12seg.pkl")
    # print("DP policy:", policy)

    ################# Apply DP policy #################
    # with open("SeriesOpt/tests/dp_cont_policy.pkl", 'rb') as f:
    #     policy = pickle.load(f)

    # print(policy[5])

    # real_prices = price_test[:opt_horizon]
    # profit_sequence, u_sequence, b_sequence = apply_dp(real_prices, hw_model, policy, 0)
    # print("Profit sequence:", profit_sequence)
    # print("Control sequence:", u_sequence)
    # print("Battery sequence:", b_sequence)
    # print("Real prices:", real_prices)