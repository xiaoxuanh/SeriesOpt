from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.core.base import ConcreteModel
import pandas as pd
import numpy as np
from ..utils import get_solver_path
import os

from ..config import Config
# Access the parameters using the get_param function
Mc = Config.get_param('Mc')
Md = Config.get_param('Md')
Me = Config.get_param('Me')
eta = Config.get_param('eta')
# print(Mc, Md, Me, eta)

def lp_optimize(b0, p_forecast, H) -> pd.DataFrame:
    """
    A linear optimization model.
    Within each step, given knowledge of current charging state and price forecast,
    the model determines the optimal charging/discharging actions for the next H steps.
    """
    model = ConcreteModel()
    
    ### sets ###
    model.PERIODS = RangeSet(0,H-1)
    
    ### parameters ###
    model.Mc = Param(initialize=Mc)
    model.Md = Param(initialize=Md)
    model.Me = Param(initialize=Me)
    model.Eta = Param(initialize=eta)

    model.Price = Param(model.PERIODS, initialize=p_forecast)
    
    ### variables ###
    model.controls = Var(model.PERIODS,within=Reals)
    model.impact_to_grid = Var(model.PERIODS,within=Reals)
    
    ### inner rule definitions ###    
    def storage_charging_cap(model,period):
        return model.controls[period] <= model.Mc
    
    def storage_discharging_cap(model,period):
        return model.controls[period] >= -model.Md
    
    def storage_charging_ene(model,period):
        return model.controls[period] <= model.Me - model.Charging_State[period]
    
    def storage_discharging_ene(model,period):
        return model.controls[period] >= -model.Charging_State[period]
    
    def calc_charging_state(model):
        charging_state = dict()
        charging_state[0] = b0
        for period in range(1, H):
            charging_state[period] = (charging_state[period-1] 
                                    + model.controls[period-1])
        return charging_state

    def impact_to_grid_charging_bound(model,period):
        return model.impact_to_grid[period] >= model.controls[period] / model.Eta
    
    def impact_to_grid_discharging_bound(model,period):
        return model.impact_to_grid[period] >= model.controls[period] * model.Eta
    
    def obj_func(model):
        return sum(model.impact_to_grid[period] * model.Price[period]
                for period in model.PERIODS)

    
    ### expressions ###
    model.Charging_State = Expression(model.PERIODS, initialize=calc_charging_state(model))
    
    ### constraints ###
    model.Storage_Charging_Cap = Constraint(model.PERIODS, rule=storage_charging_cap)
    model.Storage_Discharging_Cap = Constraint(model.PERIODS, rule=storage_discharging_cap)
    model.Storage_Charging_Ene = Constraint(model.PERIODS, rule=storage_charging_ene)
    model.Storage_Discharging_Ene = Constraint(model.PERIODS, rule=storage_discharging_ene)
    model.Grid_Impact_Charging = Constraint(model.PERIODS, rule=impact_to_grid_charging_bound)
    model.Grid_Impact_Discharging = Constraint(model.PERIODS, rule=impact_to_grid_discharging_bound)
    
    ### objective ###
    model.obj = Objective(rule=obj_func)
    
    ### solve model ###
    cbc_executable = os.path.join(get_solver_path('cbc-win64'), 'cbc')
    opt = SolverFactory('cbc', executable=cbc_executable)
    results = opt.solve(model,tee=False)
    # print(results)
    # save results
    control_results = []
    index = 0
    for v in model.component_data_objects(Var, active=True):
        if 'controls' in v.name:
            control_results.append(["controls", index, value(v)])
            index += 1
    
    control_results = pd.DataFrame(control_results, columns=['name', 'index', 'value'])
    
    return control_results

if __name__ == "__main__":
    # Example usage
    b0 = 0
    p_forecast = np.array([10, 20, 10, 10, 20])
    H = 5
    
    control_results = lp_optimize(b0, p_forecast, H)
    print(control_results)