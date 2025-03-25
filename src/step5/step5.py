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


class SingleHourModel:
    def __init__(self, offer_qty, bid_qty, bid_prices, offer_costs, num_gen, num_loads):

        self.offer_qty = offer_qty
        self.bid_qty = bid_qty
        self.bid_prices = bid_prices
        self.offer_costs = offer_costs
        self.num_gen = num_gen
        self.num_loads = num_loads


        model = gp.Model("SingleHourModel")

        # Decision variables
        pg = model.addVars(num_gen, lb=0, ub = offer_qty, vtype = GRB.CONTINUOUS, name = "p_g")
        pd = model.addVars(num_loads, lb=0, ub = bid_qty, vtype = GRB.CONTINUOUS, name = "p_d")


        # Objective function
        model.setObjective(
            gp.quicksum(bid_prices[d] * pd[d] for d in range(num_loads)) - 
            gp.quicksum(offer_costs[g] * pg[g] for g in range(num_gen)), GRB.MAXIMIZE
        )

        # Power Balance constraint
        model.addConstr(gp.quicksum(pg[g] for g in range(num_gen)) == gp.quicksum(pd[d] for d in range(num_loads)), "power_balance")

        self.vars = {
            "pg": pg,
            "pd": pd
        }

        self.model = model

    def optimize(self):
        self.model.optimize()
        assert self.model.status == GRB.OPTIMAL, "Model is not optimal"

    def get_clearing_outcome(self):
        assert hasattr(self, "model") and self.model.status == GRB.OPTIMAL, "Model is not optimal"
        mcp = -1*self.model.getConstrByName("power_balance").Pi #dual variable of the power balance

        pg = [self.vars["pg"][g].X for g in range(self.num_gen)]
        pd = [self.vars["pd"][d].X for d in range(self.num_loads)]

        # Calculate derived quantities
        demand_utility = np.dot(self.bid_prices - mcp, pd)
        gen_profit = np.dot(mcp - self.offer_costs, pg)
        social_welfare = demand_utility + gen_profit

        return {
            "mcp": mcp,
            "pg": pg,
            "pd": pd,
            "demand_utility": demand_utility,
            "gen_profit": gen_profit,
            "social_welfare": social_welfare
        }


class ImbalanceMarketModel:
    def __init__(
            self, 
            imbalance: float, 
            curtailment_prices: list[float], 
            curtailment_bounds: list[float], 
            up_offer_prices: list[float],
            down_offer_prices: list[float],
            scheduled_production: list[float], 
            max_capacity: list[float]
    ):
        self.imbalance = imbalance
        self.curtailment_prices = curtailment_prices
        self.curtailment_bounds = curtailment_bounds
        self.up_offer_prices = up_offer_prices
        self.down_offer_prices = down_offer_prices
        self.scheduled_production = scheduled_production
        self.max_capacity = max_capacity
        self.num_bsp = len(scheduled_production)
        self.num_consumers = len(curtailment_bounds)

        model = gp.Model("ImbalanceMarketModel")

        upward_bounds = max_capacity - scheduled_production
        downward_bounds = scheduled_production

        # Decision variables
        p_Up = model.addVars(self.num_bsp, lb=0, ub=upward_bounds, vtype=GRB.CONTINUOUS, name="p_Up")
        p_Down = model.addVars(self.num_bsp, lb=0, ub=downward_bounds, vtype=GRB.CONTINUOUS, name="p_Down")
        p_Curtailment = model.addVars(self.num_consumers, lb=0, ub=curtailment_bounds, vtype=GRB.CONTINUOUS, name="p_Curtailment")

        # Objective function
        model.setObjective(
            gp.quicksum(p_Up[i] * up_offer_prices[i] for i in range(self.num_bsp)) +
            gp.quicksum(p_Down[i] * down_offer_prices[i] for i in range(self.num_bsp)) +
            gp.quicksum(p_Curtailment[i] * curtailment_prices[i] for i in range(self.num_consumers)), 
            GRB.MINIMIZE
        )

        # Power balance constraint
        model.addConstr(
            gp.quicksum(p_Up) - gp.quicksum(p_Down) + gp.quicksum(p_Curtailment) == -imbalance, 
            "power_balance"
        )

        self.vars = {
            "p_Up": p_Up,
            "p_Down": p_Down,
            "p_Curtailment": p_Curtailment
        }

        self.model = model

    def optimize(self):
        self.model.optimize()
        assert self.model.status == GRB.OPTIMAL, "Model is not optimal"
    
    def get_clearing_outcome(self):
        assert hasattr(self, "model") and self.model.status == GRB.OPTIMAL, "Model is not optimal"

        mcp = -1*self.model.getConstrByName("power_balance").Pi #dual variable of the power balance
        obj = self.model.getObjective().getValue()
        p_Up = [self.vars["p_Up"][i].X for i in range(self.num_bsp)]
        p_Down = [self.vars["p_Down"][i].X for i in range(self.num_bsp)]
        p_Curtailment = [self.vars["p_Curtailment"][i].X for i in range(self.num_consumers)]


        return {
            "p_Up": p_Up,
            "p_Down": p_Down,
            "p_Curtailment": p_Curtailment,
            "obj": obj,
            "mcp": mcp
        }


def get_wind_imbalance(
    scheduled_wind: list[float],
    overproduction_rate: float = 1.15,
    underproduction_rate: float = 0.90,
) -> float:
    """ Now we select a subset of wind farms that over-produce and a subset of 
        wind farms that under-produce, relative to their forecasted production. """

    w_prod = np.array(scheduled_wind)
    n_farms = len(w_prod)

    # Split into overproducers and underproducers
    overproducers = np.arange(n_farms)[:n_farms//2]
    underproducers = np.arange(n_farms)[n_farms//2:]

    # print(f"Overproducers: {overproducers}")
    # print(f"Underproducers: {underproducers}")

    overproduction_qty = overproduction_rate * w_prod[overproducers]
    underproduction_qty = underproduction_rate * w_prod[underproducers]

    # print(f"Overproduction quantity: {overproduction_qty}")
    # print(f"Underproduction quantity: {underproduction_qty}")

    # Calculate the imbalance
    wind_imbalance = np.sum(overproduction_qty) - np.sum(underproduction_qty)
    print(f"Wind imbalance: {wind_imbalance}")


    # realized production per farm
    w_prod_realized = w_prod.copy()
    w_prod_realized[overproducers] = overproduction_qty
    w_prod_realized[underproducers] = underproduction_qty

    imbalance_per_farm = w_prod_realized - w_prod

    return wind_imbalance, imbalance_per_farm


def get_conventional_imbalance(
    scheduled_production: list[float],
    offer_costs: list[float],
    n_outages: int = 1,
) -> tuple[float, list[int]]:
    """
        We choose a subset of conventional generators that produce nothing due to outages.
        We select those with cheapest productions cost as these will definitely have been 
        scheduled to produce. 
    """
    idx_outages = np.argsort(offer_costs)[:n_outages]

    scheduled_production = np.array(scheduled_production.copy())
    realized_production = scheduled_production.copy()
    realized_production[idx_outages] = 0

    conventional_imbalance = np.sum(realized_production) - np.sum(scheduled_production)
    print(f"Conventional imbalance: {conventional_imbalance}")

    return conventional_imbalance, idx_outages
    
def get_balancing_consumers(scheduled_load: list[float], curtailment_price: float) -> tuple[list[int], list[float]]:
    """ Get the balancing consumers and their curtailment bounds. """
    curtailment_bounds = np.array(scheduled_load).copy() # Cannot curtail more than has been scheduled for each consumer
    balancing_consumers_idx = np.where(curtailment_bounds > 0)[0] # Those consumers that are actually scheduled for load.
    curtailment_bounds = curtailment_bounds[balancing_consumers_idx]
    curtailment_prices = curtailment_price * np.ones(len(balancing_consumers_idx))

    # print(f"balancing_consumers_idx: {balancing_consumers_idx}")
    # print(f"curtailment_bounds: {curtailment_bounds}")
    # print(f"curtailment_prices: {curtailment_prices}")

    return balancing_consumers_idx, curtailment_bounds, curtailment_prices
    
if __name__ == "__main__":

    dataset_dir = "datasets/"
    wind_dir = dataset_dir + "wind_farms/"

    plot_dir = "results/step5/"
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
    ############ Part A: Clear the day-ahead market ###########
    ###########################################################

    single_hour_model = SingleHourModel(offer_qty, bid_qty, bid_prices, offer_costs, num_gen, num_loads)
    single_hour_model.optimize()
    res = single_hour_model.get_clearing_outcome()

    print("-"*80)
    print("Day-ahead market clearing.")

    print("\n")
    print(f"Market clearing price: {res['mcp']:.4f} $/MWh")

    print("\n")
    print("Generator dispatch:")
    for g in range(num_gen):
        print(f"Generator {g}: {res['pg'][g]:.4f} MW")

    print("\n")
    print("Load dispatch:")
    for d in range(num_loads):
        print(f"Load {d}: {res['pd'][d]:.4f} MW")

    print("\n")
    print(f"Demand utility: {res['demand_utility']:.4f} USD")
    print(f"Generator profit: {res['gen_profit']:.4f} USD")
    print(f"Social welfare: {res['social_welfare']:.4f} USD")


    ###########################################################
    ############ Part B: Clear the imbalance market ###########
    ###########################################################
    print("-"*80)
    print("Imbalance market clearing.")

    # Calculate the power imbalance
    wind_imbalance, imbalance_per_farm = get_wind_imbalance(scheduled_wind=res['pg'][-len(wind_data):])
    conventional_imbalance, idx_outages = get_conventional_imbalance(
        scheduled_production=res['pg'][:-len(wind_data)],
        offer_costs=offer_costs[:-len(wind_data)]
    )
    imbalance = wind_imbalance + conventional_imbalance
    print(f"Power imbalance: {imbalance}")

    # Calculated prices of balancing services
    curtailment_price = 500 # $/MWh
    up_reg_coef = 0.10 # 10% of the cost of the generator
    down_reg_coef = 0.15 # 15% of the cost of the generator

    # Only conventional generators provide balancing services
    bsp_indices = np.arange(len(offer_costs)-len(wind_data)) # all conventional generators
    bsp_indices = np.delete(bsp_indices, idx_outages) # pop the dead generator
    up_offer_prices = res["mcp"] + offer_costs[bsp_indices] * up_reg_coef
    down_offer_prices = res["mcp"] - offer_costs[bsp_indices] * down_reg_coef

    dead_generators = [f"G{gen_node[i]}" for i in idx_outages]
    print(f"Eliminated generators: {dead_generators}")

    # print(f"up_offer_prices: {up_offer_prices}")
    # print(f"down_offer_prices: {down_offer_prices}")


    # Get the balancing consumers and their curtailment bounds
    balancing_consumers_idx, curtailment_bounds, curtailment_prices = get_balancing_consumers(
        scheduled_load=res['pd'],
        curtailment_price=curtailment_price
    )

    # Get the bounds of the BSPs
    scheduled_production = np.array([res['pg'][i] for i in bsp_indices])
    # print(f"scheduled_production: {scheduled_production}")
    max_capacity = np.array([offer_qty[i] for i in bsp_indices])
    # print(f"max_capacity: {max_capacity}")
    

    # Create the imbalance market model
    imbalance_market_model = ImbalanceMarketModel(
        imbalance, 
        curtailment_prices,
        curtailment_bounds,
        up_offer_prices,
        down_offer_prices,
        scheduled_production,
        max_capacity
    )
    imbalance_market_model.optimize()
    imbalance_res = imbalance_market_model.get_clearing_outcome()
    
    print(f"imbalance_res: {imbalance_res}")

    
    # Create a single row latex table with the following columns:
    # - mcp
    # obj
    print(f"imbalance_res['mcp']: {imbalance_res['mcp']}")
    print(f"imbalance_res['obj']: {imbalance_res['obj']}")

    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{cccccc|cc}")
    print(" BSP      & Node & Price     & Price     & Day-Ahead & Max      &    \multicolumn{2}{c}{\\textbf{Clearing outcome}} \\\\")
    print(" index  & tag    & Up        & Down      & Scheduled & Capacity &   res['p\_Up'] & res['p\_Down']    \\\\")
    print("         &       & (\$/MWh)  & (\$/MWh) & (MW)      & (MW)     & (MW) & (MW)   \\\\")
    
    print("\\hline")

    for i in range(len(bsp_indices)):
        print(f"{bsp_indices[i]} & G{gen_node[bsp_indices[i]]} & {up_offer_prices[i]:.3f} & {down_offer_prices[i]:.3f} & {scheduled_production[i]:.2f} & {max_capacity[i]:.0f} & {imbalance_res['p_Up'][i]:.1f} & {imbalance_res['p_Down'][i]:.1f} \\\\")
    print("\\end{tabular}")
    print("\\caption{Clearing inputs and outputs regarding conventional flexible producers in the balancing market (BSPs).}")
    print("\\label{tab:step5_bsps}")
    print("\\end{table}")



    # Create a latex table of the balancing consumers and their curtailment bounds
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{cccc|c}")
    print(" Load & Node & Price & Bounds & Curtailment \\\\")
    print(" index & tag & (\$/MWh) & (MW) & (MW) \\\\")
    print("\\hline")
    for i in range(len(balancing_consumers_idx)):
        print(f"{balancing_consumers_idx[i]} & D{int(load_number[balancing_consumers_idx[i]])} & {curtailment_prices[i]:.3f} & {curtailment_bounds[i]:.2f} & {imbalance_res['p_Curtailment'][i]:.1f} \\\\")
    print("\\end{tabular}")
    print("\\caption{Clearing inputs and outputs regarding balancing consumers in the balancing market. Notice that only consumers with positive curtailment bounds are included in the table. Consumer with no load scheduled from the day-ahead market are not able to provide upward regulation through curtailment.}")
    print("\\label{tab:step5_balancing_consumers}")


    print(f"{'---'*50}")
    # Create a latex table of the wind producers. Show the scheduled production and the actual production.
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{cccc|ccc}")
    print(" & W1 & W2 & W3 & W4 & W5 & W6 \\\\")
    print("\\hline")
    row = "Imbalance (MWh): & "
    print(f"{row} " + " & ".join(f"{imbalance:.2f}" for imbalance in imbalance_per_farm[:-1]) + f" & {imbalance_per_farm[-1]:.2f} \\\\")
    print("\\end{tabular}")
    print("\\caption{Imbalance per wind farm. Overproducers on the left, underproducers on the right.}")
    print("\\label{tab:step5_wind_imbalance}")
    print("\\end{table}")


    # One-price scheme
    print(f"{'---'*50}")
    print("One-price scheme.")

    day_ahead_price = res['mcp']
    imbalance_price = imbalance_res['mcp']
    print(f"day_ahead_price: {day_ahead_price}")
    print(f"imbalance_price: {imbalance_price}")

    # name_tags = [f"G{gen_node[i]}" for i in range(len(offer_costs))]

    # Calculate the profits from the day-ahead market
    day_ahead_profits = res['pg'] * (day_ahead_price - offer_costs)
    print(f"day_ahead_profits: {day_ahead_profits}")

    plt.figure(figsize=(10, 5))
    plt.bar(name_tags, day_ahead_profits)
    plt.xlabel("Generator")
    plt.ylabel("Profit")
    plt.title("Day-ahead profits")
    plt.savefig(plot_dir + "day_ahead_profits.png")
    plt.close()

    # Calculate the profits from the imbalance market
    imbalance_earnings = np.array(imbalance_res['p_Up']) * imbalance_price
    print(f"imbalance_earnings: {imbalance_earnings}")
    

    # Calculate the total profits
    # total_profits = day_ahead_profits + imbalance_profits
    


    