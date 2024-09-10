class Config:
    # define global variables
    PARAMS = {
        # storage power capacity (in MW)
        'Mc': 1,
        'Md': 1,
        # storage energy capacity and initial level (in MWh)
        'Me': 2,
        'b0': 0,
        # efficiency
        'eta': 0.9,
        # parameters for HW model
        'alpha': 0.01,
        'beta': 0.003,
        'gamma': 0.3,
        'm': 4,
        'sigma': 10,
        # parameters for periodic optimization
        'reopt_freq': 4, # re-optimization frequency, no larger than opt_horizon
        'opt_horizon': 5,
        # parameters for DP optimizer
        'max_num_b_states': 4, # number of b states for discretization
        'max_b_step_size': 1, # maximum step size for b states; the smaller the steps the more number of states
        'max_num_x_steps': 4, # number of x states for discretization; will explode by the order of m+2 for level, trend, seasonality
        'max_x_step_size': 1, # maximum step size for x states; the smaller the steps the more number of states
    }

    @staticmethod
    def set_params(custom_params):
        """
        Allows users to customize the global parameters.
        Accepts a dictionary with parameter names and values.
        """
        Config.PARAMS.update(custom_params)

    @staticmethod
    def get_param(param_name):
        """
        Returns the value of a parameter by name.
        """
        return Config.PARAMS.get(param_name, None)