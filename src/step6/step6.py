import argparse, os
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB


from data.data import load_raw_data

from data.data import (
    load_raw_data, 
    process_generator_data, 
    process_generator_costs, 
    process_load_profile, 
    process_node_location_and_distribution, 
)

from step5.step5 import (
    get_balancing_consumers,
    get_wind_imbalance,
    get_conventional_imbalance,
    SingleHourModel,
    ImbalanceMarketModel
)



class ReserveMarketModel:
    def __init__(
            self, 
            up_needed: float, 
            down_needed: float, 
            up_offers: np.ndarray, 
            down_offers: np.ndarray, 
            up_max: np.ndarray, 
            down_max: np.ndarray
        ):
        self.up_needed = up_needed
        self.down_needed = down_needed
        self.up_offers = up_offers
        self.down_offers = down_offers
        self.up_max = up_max
        self.down_max = down_max

        model = gp.Model("Reserve Market")

        # Decision variables
        p_up = model.addVars(len(up_offers), lb=0, ub=up_max, vtype=GRB.CONTINUOUS, name="p_up")
        p_down = model.addVars(len(down_offers), lb=0, ub=down_max, vtype=GRB.CONTINUOUS, name="p_down")

        # Objective function
        model.setObjective(
            gp.quicksum(p_up[i] * up_offers[i] for i in range(len(up_offers))) +
            gp.quicksum(p_down[i] * down_offers[i] for i in range(len(down_offers))), 
            GRB.MINIMIZE
        )

        # Constraints
        model.addConstr(
            gp.quicksum(p_up) >= up_needed, 
            "up_needed"
        )
        model.addConstr(
            gp.quicksum(p_down) >= down_needed, 
            "down_needed"
        )
        
        self.vars = {
            "p_up": p_up,
            "p_down": p_down
        }

        self.model = model

    def optimize(self):
        self.model.optimize()
        assert self.model.status == GRB.OPTIMAL, "Model is not optimal"

    def get_clearing_outcome(self):
        # Get the up price and down price from the dual values of the constraints
        up_price = self.model.getConstrByName("up_needed").getAttr("Pi")
        down_price = self.model.getConstrByName("down_needed").getAttr("Pi")

        p_up = [self.vars["p_up"][i].X for i in range(len(self.up_offers))]
        p_down = [self.vars["p_down"][i].X for i in range(len(self.down_offers))]

        return {
            "p_up": p_up,
            "p_down": p_down,
            "up_price": up_price,
            "down_price": down_price,
            "obj": self.model.getObjective().getValue()
        }

if __name__ == "__main__":

    dataset_dir = "datasets/"
    wind_dir = dataset_dir + "wind_farms/"

    plot_dir = "results/step6/"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    (
        generator_data, 
        generator_costs, 
        load_profile, 
        node_location_and_distribution, 
        transmission_data, 
        wind_data
    ) = load_raw_data(dataset_dir, wind_dir)


    # GENERATOR DATA 
    (gen_unit, gen_node, gen_Pmax, gen_Pmin, gen_Rplus, gen_Rminus, gen_Ru, gen_Rd, gen_UT, gen_DT) = process_generator_data(generator_data)

    print(gen_node)

    # GENERATOR COSTS
    (gen_C, gen_Cu, gen_Cd, gen_Cplus, gen_Cminus, gen_Csu, gen_Pini, gen_Uini, gen_Tini) = process_generator_costs(generator_costs)

    # LOAD PROFILE
    system_demand_hours, system_demand = process_load_profile(load_profile)

    # LOAD LOCATION AND DISTRIBUTION
    (load_number, load_node, load_percentage) = process_node_location_and_distribution(node_location_and_distribution)


    # Find the peak hour for the single hour setting
    peak_demand = np.max(system_demand) # 2650.5 MW
    peak_idx = np.argmax(system_demand) # Hour 18
    print(f"peak_idx: {peak_idx}")
    wind_production_at_peak_demand = np.array([i[peak_idx] for i in wind_data]) # get the wind production for each wind farm at peak hour


    # Find total combined capacities and costs
    wind_costs = np.zeros(len(wind_data))
    offer_costs = np.concatenate((gen_C, wind_costs)) # Total generator costs
    offer_qty = np.concatenate((gen_Pmax, wind_production_at_peak_demand)) # Total generator capacities

    # Find the total MW on each load during hour 18 and bid prices
    bid_qty = load_percentage * peak_demand # the size of each load during hour 18 (sums to peak_demand)
    min_bid_price = min(offer_costs)   # Setting minimum bid price relatively high
    max_bid_price = max(offer_costs)   # Increase maximum bid price to be significantly higher than costs

    # Normalize the loads to the highest load 
    normalized_load = (bid_qty - min(bid_qty)) / (max(bid_qty) - min(bid_qty)) # Normalized loads
    bid_prices = min_bid_price + normalized_load * (max_bid_price - min_bid_price) # Bid prices - ensuring they're higher than costs

    num_gen = len(offer_qty) #number of generators
    num_loads = len(bid_qty) #number of loads


    ###########################################################
    ############ Part A: Clear the reserve market ###########
    ###########################################################
    print("-"*80)
    print("Reserve market clearing.")


    # Only conventional generators provide balancing services
    total_demand = np.sum(bid_qty)

    # Calculate the total up and down reserve needed
    up_needed = 0.15 * total_demand
    down_needed = 0.10 * total_demand

    # Only conventional generators provide balancing services
    bsp_indices = np.arange(len(offer_costs)-len(wind_data)) # all conventional generators


    up_reg_coef = 0.10 # 10% of the cost of the generator
    down_reg_coef = 0.15 # 15% of the cost of the generator

    up_offers = offer_costs[bsp_indices] * up_reg_coef
    down_offers = offer_costs[bsp_indices] * down_reg_coef


    # Create the reserve market model
    reserve_market_model = ReserveMarketModel(
        up_needed,
        down_needed,
        up_offers,
        down_offers,
        up_max = gen_Pmax[bsp_indices] * 0.15,
        down_max = gen_Pmax[bsp_indices] * 0.10
    )

    reserve_market_model.optimize()
    reserve_market_outcome = reserve_market_model.get_clearing_outcome()
    print(reserve_market_outcome)


    # Create a latex table of the clearing outcome 

    p_up = reserve_market_outcome['p_up']
    p_down = reserve_market_outcome['p_down']


    n_bsp = len(bsp_indices)

    print("-"*80)


    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\begin{{tabular}}{{{'c' * (n_bsp + 1)}}}")
    print(" & ".join(f"G{i}" for i in gen_node[bsp_indices]))
    print("\\hline")
    row_up = "p^{\-uparrow}"
    row_down = "p^{\-downarrow}"
    for i in range(len(bsp_indices)):
        row_up += f" & {p_up[i]:.1f} "
        row_down += f" & {p_down[i]:.1f} "
    row_up += "\\\\"
    row_down += "\\\\"

    print(row_up)
    print(row_down)

    print("\\end{tabular}")
    print("\\caption{Clearing outcome of the reserve market.}")
    print("\\label{tab:step6_reservation}")
    print("\\end{table}")
