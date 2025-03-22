import os
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import argparse  # Add this import at the top


from data.data import (
    load_raw_data, 
    process_generator_data, 
    process_generator_costs, 
    process_load_profile, 
    process_node_location_and_distribution, 
    process_transmission_data
)


def get_battery_args():
    parser = argparse.ArgumentParser(description="Storage parameters for the renewable energy model.")
    parser.add_argument('--Pch', type=float, default=75, help='Max charging power (MW)')
    parser.add_argument('--Pdis', type=float, default=73, help='Max discharging power (MW)')
    parser.add_argument('--E', type=float, default=100, help='Storage capacity (MWh)')
    parser.add_argument('--eta_ch', type=float, default=0.95, help='Charging efficiency')
    parser.add_argument('--eta_dis', type=float, default=0.93, help='Discharging efficiency')

    return parser.parse_args()



# Function that maximizes total social welfare using Gurobi
def optimize_model(total_gen_costs, total_gen_capacities, demand_per_hour, bid_prices, with_storage=True,
                   Pch=75, Pdis=73, E=100, eta_ch=0.95, eta_dis=0.93):
    
    model = gp.Model("Social_Welfare_Maximization")
    
    G = len(total_gen_costs)  # Number of generators (18)
    D = len(demand_per_hour[0])  # Number of demand points (17)
    
    # Decision Variables
    p_g = model.addVars(G, T, lb=0, ub=total_gen_capacities[:, :G].T, vtype=GRB.CONTINUOUS, name="p_g")  # Generation
    p_d = model.addVars(D, T, lb=0, ub=demand_per_hour.T, vtype=GRB.CONTINUOUS, name="p_d")  # Consumption
    
    if with_storage:
        p_ch = model.addVars(T, lb=0, ub=Pch, vtype=GRB.CONTINUOUS, name="p_ch")  # Charging
        p_dis = model.addVars(T, lb=0, ub=Pdis, vtype=GRB.CONTINUOUS, name="p_dis")  # Discharging
        e = model.addVars(T, lb=0, ub=E, vtype=GRB.CONTINUOUS, name="e")  # Storage Energy Level
        model.addConstr(e[0] == 0, name="initial_energy")  # Force battery to start empty
        model.addConstr(p_dis[0] == 0, name="initial_discharge")
    else:
        p_ch = {t: 0 for t in range(T)}
        p_dis = {t: 0 for t in range(T)}
        e = {t: 0 for t in range(T)}

    # Objective: Maximize Social Welfare
    model.setObjective(
        gp.quicksum(bid_prices[t, d] * p_d[d, t] for d in range(D) for t in range(T)) -
        gp.quicksum(total_gen_costs[g] * p_g[g, t] for g in range(G) for t in range(T)),
        GRB.MAXIMIZE
    )

    # Constraints
    for g in range(G):
        for t in range(T):
            model.addConstr(p_g[g, t] <= total_gen_capacities[t, g], name=f"gen_limit_{g}_{t}")

    for d in range(D):
        for t in range(T):
            model.addConstr(p_d[d, t] <= demand_per_hour[t, d], name=f"demand_limit_{d}_{t}")

    for g in range(num_conv_gens):
        for t in range(1, T):
            model.addConstr(p_g[g, t] - p_g[g, t - 1] <= gen_Ru[g], name=f"ramp_up_{g}_{t}")
            model.addConstr(p_g[g, t] - p_g[g, t - 1] >= -gen_Rd[g], name=f"ramp_down_{g}_{t}")
        model.addConstr(p_g[g, 0] - gen_Pini[g] <= gen_Ru[g], name=f"ramp_up_initial_{g}")
        model.addConstr(p_g[g, 0] - gen_Pini[g] >= -gen_Rd[g], name=f"ramp_down_initial_{g}")

    power_balance_constraints = {}
    for t in range(T):
        if with_storage:
            power_balance_constraints[t] = model.addConstr(
                gp.quicksum(p_g[g, t] for g in range(G)) + p_dis[t] == gp.quicksum(p_d[d, t] for d in range(D)) + p_ch[t],
                name=f"power_balance_{t}"
            )
        else:
            power_balance_constraints[t] = model.addConstr(
                gp.quicksum(p_g[g, t] for g in range(G)) == gp.quicksum(p_d[d, t] for d in range(D)),
                name=f"power_balance_{t}"
            )

    if with_storage:
        for t in range(1, T):
            model.addConstr(e[t] == e[t-1] + eta_ch * p_ch[t] - (1/eta_dis) * p_dis[t], name=f"storage_balance_{t}")
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        market_prices = [-1 * power_balance_constraints[t].Pi for t in range(T)]
        dispatched_power = [[p_g[g, t].X for g in range(G)] for t in range(T)]
        consumed_power = [[p_d[d, t].X for d in range(D)] for t in range(T)]
        battery_energy = [e[t].X for t in range(T)] if with_storage else [0] * T
        charging_power = [p_ch[t].X for t in range(T)] if with_storage else [0] * T
        discharging_power = [p_dis[t].X for t in range(T)] if with_storage else [0] * T
    
    else:
        market_prices = [0] * T
        dispatched_power = [[0] * G for _ in range(T)]
        consumed_power = [[0] * D for _ in range(T)]
        battery_energy = [0] * T
        charging_power = [0] * T
        discharging_power = [0] * T
    
    print(f"Objective value: {model.getObjective().getValue()}")

    return market_prices, dispatched_power, consumed_power, battery_energy, charging_power, discharging_power, model



def plot_market_clearing_and_battery_power(market_prices, pg, SOC, p_ch, p_dis, T, plot_dir):
    # Generate plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # Now only 2 subplots

    # Market price plot with overlaid generation
    ax1 = axs[0]  # Primary y-axis for market price
    ax2 = ax1.twinx()  # Secondary y-axis for total generation

    ax1.step(range(T), market_prices, where="post", label="Market price with storage", color="blue", linewidth=2)
    pg_per_hour = np.array(pg).sum(axis=1)
    ax2.plot(range(T), pg_per_hour, label="Total generation", color="purple", linewidth=2, linestyle='dashed')

    ax1.set_xlabel("Hour", fontsize=12)
    ax1.set_ylabel("Market-clearing price [$/MWh]", fontsize=12, color="blue")    
    ax2.set_ylabel("Total generation [MW]", fontsize=12, color="purple")

    ax1.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)

    ax1.grid(True, linestyle='--', alpha=0.6)

    # Battery SOC plot
    axs[1].plot(range(T), SOC, label="Total energy [MWh]", color="green", linewidth=2)
    axs[1].plot(range(T), p_ch, label="Charge [MW]", color="orange", linewidth=2)
    axs[1].plot(range(T), p_dis, label="Discharge [MW]", color="red", linewidth=2)

    axs[1].set_xlabel("Hour", fontsize=12)
    axs[1].set_ylabel("Battery Power [MW]", fontsize=12)
    axs[1].legend(loc='upper right', fontsize=10)
    axs[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()  # Adjust layout to prevent overlap
    # add a little space between the two plots
    plt.subplots_adjust(wspace=0.35)
    plt.savefig(plot_dir + "step2.png", dpi=300, bbox_inches='tight')
    plt.show()



def get_curve_from_unsorted(quantities, prices, ascending=True):
    # We use ascending=True for supply curve and ascending=False for demand curve

    sorted_indices = np.argsort(prices)
    if not ascending:
        sorted_indices = sorted_indices[::-1]

    sorted_prices = prices[sorted_indices]
    sorted_quantities = quantities[sorted_indices]

    cumulative_quantities = np.cumsum(sorted_quantities)

    if ascending:
        # add initial dummy point
        insert_idx = 0
        sorted_prices = np.insert(sorted_prices, insert_idx, 0)
        sorted_quantities = np.insert(sorted_quantities, insert_idx, 0)
        cumulative_quantities = np.insert(cumulative_quantities, insert_idx, 0)

    return sorted_prices, sorted_quantities, cumulative_quantities



def plot_offer_shift(op1, oq1, bq, bp, op2, oq2, mcp, ax):
    
    sp1, _, csq1 = get_curve_from_unsorted(oq1, op1, ascending=True)
    sp2, _, csq2 = get_curve_from_unsorted(oq2, op2, ascending=True)
    
    dp, _, cdq = get_curve_from_unsorted(bq, bp, ascending=False)
    
    
    ax.step(csq1, sp1, where='pre', 
             label='Offer curve', color='blue', linewidth=2)
    ax.step(csq2, sp2, where='pre', 
             label='Offer curve with storage', color='green', linewidth=2)
    ax.step(cdq, dp, where='pre', 
             label='Demand curve', color='red', linewidth=2)
    
    ax.axhline(y=mcp, color='black', linestyle='dotted', linewidth=2, label='Market clearing price')
    ax.text(1600, mcp + 2, 
             f'MCP unaffected', 
             fontsize=12, color='black', fontweight='bold')
    
    ax.set_xlabel("Cumulative capacity / demand [MW]", fontsize=12)
    ax.set_ylabel("Offer / bid price [$/MWh]", fontsize=12)    
    ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, framealpha=0.8)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(700, 3000)

    return ax



def make_triple_plot(market_prices, market_prices_without_storage, pg, pg_without_storage, 
                     pd, pd_without_storage, p_dis, p_dis_without_storage, T, plot_dir):

    # Generate plots
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.7))  # Now only 2 subplots

    axs[0].step(range(T), market_prices, where="post", label="Market price with storage", color="blue", linewidth=2)
    axs[0].step(range(T), market_prices_without_storage, where="post", label="Market price without storage", color="red", linewidth=2)
    axs[0].set_xlabel("Hour", fontsize=12)
    axs[0].set_ylabel("Market-clearing price [$/MWh]", fontsize=12, )
    axs[0].legend(loc='center right', fontsize=10)
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # Dispatched power plot
    pg_per_hour = np.array(pg).sum(axis=1)
    pg_per_hour_without_storage = np.array(pg_without_storage).sum(axis=1)
    axs[1].plot(range(T), pg_per_hour, label="Total generation [MW]", color="blue", linewidth=2, linestyle='dashed')
    axs[1].plot(range(T), pg_per_hour_without_storage, label="Total generation without storage [MW]", color="red", linewidth=2)
    axs[1].set_xlabel("Hour", fontsize=12)
    axs[1].set_ylabel("Total generation [MW]", fontsize=12)
    axs[1].legend(loc='center right', fontsize=10)
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # Calculated the following quantities:
    # A) Total dispatched power
    total_dispatched_power = np.array(pg).sum(axis=1)
    total_dispatched_power_without_storage = np.array(pg_without_storage).sum(axis=1)

    # B) Total consumed power
    total_consumed_power = np.array(pd).sum(axis=1)
    total_consumed_power_without_storage = np.array(pd_without_storage).sum(axis=1)


    # find the peak hour of the battery discharge
    peak_discharge = np.max(p_dis)
    peak_hour = np.argmax(p_dis)


    bid_qty = demand_per_hour[peak_hour]
    bid_price = bid_prices[peak_hour]

    offer_qty = total_gen_capacities[peak_hour]
    offer_price = total_gen_costs

    new_offer_qty = np.concatenate((offer_qty, np.array([p_dis[peak_hour]])))
    new_offer_price = np.concatenate((offer_price, np.array([0])))

    plot_offer_shift(offer_price, offer_qty, bid_qty, bid_price, new_offer_price, new_offer_qty, market_prices[peak_hour], axs[2])


    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.subplots_adjust(wspace=0.20)
    plt.savefig(plot_dir + "step2_triple.png", dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":

    # STORAGE PARAMETERS
    args = get_battery_args()

    dataset_dir = "datasets/"
    wind_dir = dataset_dir + "wind_farms/"

    plot_dir = "results/step2/"
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




    # Number of hours
    T = 24

    # Number of conventional generators
    num_conv_gens = len(gen_unit)

    # Input data
    wind_production_data = np.array(wind_data).T
    total_gen_capacities = np.column_stack((np.tile(gen_Pmax, (T,1)), wind_production_data))
    demand_per_hour = np.array([load_percentage * system_demand[t] for t in range(T)])

    wind_costs = np.zeros(len(wind_data))
    total_gen_costs = np.concatenate([gen_C, wind_costs])
    min_bid_price = min(total_gen_costs)
    max_bid_price = max(total_gen_costs)
    normalized_load = (demand_per_hour - demand_per_hour.min()) / (demand_per_hour.max() - demand_per_hour.min())
    bid_prices = min_bid_price + normalized_load * (max_bid_price - min_bid_price)



    # Run the optimization
    (
        market_prices, 
        pg, 
        pd, 
        SOC, 
        p_ch, 
        p_dis,
        model
    ) = optimize_model(total_gen_costs, total_gen_capacities, demand_per_hour, bid_prices, with_storage=True, 
                       Pch=args.Pch, Pdis=args.Pdis, E=args.E, eta_ch=args.eta_ch, eta_dis=args.eta_dis)
    plot_market_clearing_and_battery_power(market_prices, pg, SOC, p_ch, p_dis, T, plot_dir)
    print("-"*80)
    print("a) The market-clearing prices (one per hour) under a uniform pricing scheme.")
    print(f"Market clearing prices: {market_prices}")


    print("-"*80)
    print("b) Social welfare")
    print(f"Social welfare: {model.getObjective().getValue()}")



    print("-"*80)
    print("c) The profit of each generator")
    generator_profits = []
    for g in range(len(total_gen_costs)):  
        profit = sum(market_prices[t] * pg[t][g] - total_gen_costs[g] * pg[t][g] for t in range(T))
        generator_profits.append(profit)

    for g, profit in enumerate(generator_profits):
        print(f"Generator {g+1}: ${profit:,.2f}")

    print(f"Total generator profits: ${sum(generator_profits):,.2f}")


    print("-"*80)
    print("d) Find battery profit over 24 hours")
    battery_profit = sum(market_prices[t] * SOC[t] - market_prices[t] * p_ch[t] / args.eta_ch for t in range(T))

    print(f"\nTotal battery profit: ${battery_profit:,.2f}")




    print("-"*80)
    print("e and f) Hourly market-clearing prices with and without storage.")
    # Run the optimization without storage
    (
        market_prices_without_storage, 
        pg_without_storage, 
        pd_without_storage, 
        SOC_without_storage, 
        p_ch_without_storage, 
        p_dis_without_storage,
        model_without_storage
    ) = optimize_model(total_gen_costs, total_gen_capacities, demand_per_hour, bid_prices, with_storage=False,
                       Pch=args.Pch, Pdis=args.Pdis, E=args.E, eta_ch=args.eta_ch, eta_dis=args.eta_dis)

    make_triple_plot(market_prices, market_prices_without_storage, pg, pg_without_storage, 
                     pd, pd_without_storage, p_dis, p_dis_without_storage, T, plot_dir)