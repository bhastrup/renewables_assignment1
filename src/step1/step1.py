import os
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB


from data.data import (
    load_raw_data, 
    process_generator_data, 
    process_generator_costs, 
    process_load_profile, 
    process_node_location_and_distribution, 
    process_transmission_data
)

def plot_merit_order_curves_single_hour(total_gen_costs, total_gen_capacities, load_demand, bid_prices, plot_dir):
    
    # Supply curve
    sorted_idx_gen_costs = np.argsort(total_gen_costs) # Identify generator cost indices
    sorted_total_gen_costs = total_gen_costs[sorted_idx_gen_costs] # Generator costs in ascending order
    sorted_total_gen_capacities = total_gen_capacities[sorted_idx_gen_costs] # Corresponding generator capacity
    cumulative_capacity = np.cumsum(sorted_total_gen_capacities) # Add cumulative capacities
    
    # add initial dummy point
    cumulative_capacity = np.insert(cumulative_capacity, 0, 0)
    sorted_total_gen_costs = np.insert(sorted_total_gen_costs, 0, sorted_total_gen_costs[0])

    # Demand curve
    sorted_idx_bid_prices = np.argsort(bid_prices)[::-1] # Identify bid price indices
    sorted_bid_prices = bid_prices[sorted_idx_bid_prices] # Bid prices in descending order
    sorted_total_demand = load_demand[sorted_idx_bid_prices] # Corresponding load demand
    cumulative_load_demand = np.cumsum(sorted_total_demand) # Add cumulative load demand

    # add initial dummy point
    cumulative_load_demand = np.insert(cumulative_load_demand, 0, 0)
    sorted_bid_prices = np.insert(sorted_bid_prices, 0, sorted_bid_prices[0])
    
    
    plt.style.use("default")
    
    plt.figure(figsize=(6, 4))
    
    plt.step(cumulative_capacity, sorted_total_gen_costs, where='pre', 
             label='Cumulative generator capacity', color='blue', linewidth=2)
    plt.step(cumulative_load_demand, sorted_bid_prices, where='pre', 
             label='Cumulative load demand', color='red', linewidth=2)
    market_clearing_price = 10.2170 
    plt.axhline(y=market_clearing_price, color='black', linestyle='dotted', linewidth=2, label='Market clearing price')
    plt.text(2200, market_clearing_price - 2, 
             f'MCP: {market_clearing_price:.2f} $/MWh', 
             fontsize=12, color='black', fontweight='bold')
    
    plt.xlabel("Cumulative capacity / demand [MW]", fontsize=12)
    plt.ylabel("Offer / bid price [$/MWh]", fontsize=12)    
    plt.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, framealpha=0.8)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(plot_dir + "market_clearing.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":

    dataset_dir = "datasets/"
    wind_dir = dataset_dir + "wind_farms/"

    plot_dir = "results/step1/"
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

    # GENERATOR COSTS
    (gen_C, gen_Cu, gen_Cd, gen_Cplus, gen_Cminus, gen_Csu, gen_Pini, gen_Uini, gen_Tini) = process_generator_costs(generator_costs)

    # LOAD PROFILE
    system_demand_hours, system_demand = process_load_profile(load_profile)

    # LOAD LOCATION AND DISTRIBUTION
    (load_number, load_node, load_percentage) = process_node_location_and_distribution(node_location_and_distribution)

    # TRANSMISSION DATA
    (transmission_from, transmission_to, transmission_reactance, transmission_capacity) = process_transmission_data(transmission_data)



    # find the peak hour for the single hour setting
    peak_demand = np.max(system_demand) # 2650.5 MW
    peak_idx = np.argmax(system_demand) # Hour 18
    wind_production_at_peak_demand = np.array([i[peak_idx] for i in wind_data]) # get the wind production for each wind farm at peak hour

    # Find total combined capacities and costs
    wind_costs = np.zeros(len(wind_data))
    total_gen_costs = np.concatenate((gen_C, wind_costs)) # Total generator costs
    total_gen_capacities = np.concatenate((gen_Pmax, wind_production_at_peak_demand)) # Total generator capacities

    # Find the total MW on each load during hour 18 and bid prices
    load_demand = load_percentage * peak_demand # the size of each load during hour 18 (sums to peak_demand)
    min_bid_price = min(total_gen_costs)   # Setting minimum bid price relatively high
    max_bid_price = max(total_gen_costs)   # Increase maximum bid price to be significantly higher than costs

    # Normalize the loads to the highest load 
    normalized_load = (load_demand - min(load_demand)) / (max(load_demand) - min(load_demand)) # Normalized loads
    bid_prices = min_bid_price + normalized_load * (max_bid_price - min_bid_price) # Bid prices - ensuring they're higher than costs

    

    num_gen = len(total_gen_capacities) #number of generators
    num_loads = len(load_demand) #number of loads

    #set up Gurobi model
    model = gp.Model("Step1-Maximize SW")

    #decision variables
    pg = model.addVars(num_gen, lb=0, ub = total_gen_capacities, vtype = GRB.CONTINUOUS, name = "p_g")
    pd = model.addVars(num_loads, lb=0, ub = load_demand, vtype = GRB.CONTINUOUS, name = "p_d")

    #objective function
    model.setObjective(
        gp.quicksum(bid_prices[d] * pd[d] for d in range(num_loads)) - 
        gp.quicksum(total_gen_costs[g] * pg[g] for g in range(num_gen)), GRB.MAXIMIZE
    )

    # power Balance constraint
    model.addConstr(gp.quicksum(pg[g] for g in range(num_gen)) == gp.quicksum(pd[d] for d in range(num_loads)), "power_balance")

    # sum of loads must equal system demand
    #model.addConstr(gp.quicksum(pd[d] for d in range(num_loads)) == peak_demand, "demand_balance")

    #solve the model
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print("Optimization did not find an optimal solution.")
        exit()


    mcp = -1*model.getConstrByName("power_balance").Pi #dual variable of the power balance
    print("\n")
    print(f"Market clearing price: {mcp:.4f} $/MWh")
    
    print("\n")
    print("Generator dispatch:")
    for g in range(num_gen):
        print(f"Generator {g}: {pg[g].X:.4f} MW")
    
    print("\n")
    print("Load dispatch:")
    for d in range(num_loads):
        print(f"Load {d}: {pd[d].X:.4f} MW")

    
    # Social welfare
    demand_welfare = sum([(bid_prices[d] - mcp) * pd[d].X for d in range(num_loads)])
    print("\n")
    print(f"Demand welfare: {demand_welfare:.4f} USD")
    gen_welfare = sum([(mcp - total_gen_costs[g]) * pg[g].X for g in range(num_gen)])
    print(f"Generator welfare: {gen_welfare:.4f} USD")
    social_welfare = demand_welfare + gen_welfare



    print("-"*80)
    print("a) Determine the market clearing price under a uniform pricing scheme.")
    plot_merit_order_curves_single_hour(total_gen_costs, total_gen_capacities, load_demand, bid_prices, plot_dir)
    print("The market clearing price can be seen in the intersection point in the supply demand curve.")
    print(f"Market clearing price: {mcp:.4f} $/MWh")



    print("-"*80)
    print("b) Determine the social welfare of the system.")
    print(f"Social welfare: {social_welfare:.4f} USD")


    
    print("-"*80)
    print("c) Determine the profit of each generator")
    gen_profit = np.zeros(num_gen)
    for g in range(num_gen):
        pg_dispatched = pg[g].X
        gen_profit[g] = (mcp-total_gen_costs[g]) * pg_dispatched 
        print(f"Generator {g} profit: {gen_profit[g]:.2f} USD")



    print("-"*80)
    print("d) Determine the utility of each demand")
    demand_utilities = np.zeros(num_loads)
    for d in range(num_loads):
        consumed_power = pd[d].X
        demand_utilities[d] = (bid_prices[d] - mcp) * consumed_power
        print(f"Load {d} utility: {demand_utilities[d]:.2f} USD")