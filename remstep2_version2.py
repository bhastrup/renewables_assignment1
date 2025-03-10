import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt


# **Step 2: Read Input Data**
print("Step 2: Copper Plate - Multiple Hours")

# Load CSV files into Pandas DataFrames
generator_data = pd.read_csv("generator_data_24.csv")
generator_costs = pd.read_csv("generator_costs_24.csv")
load_profile = pd.read_csv("load_profile_24.csv")
node_location_and_distribution = pd.read_csv("node_location_and_distribution_24.csv")
transmission_data = pd.read_csv("transmission_data_24.csv")

# WIND DATA for the day and wind costs

wind_files = [
    "wind_farm_1_DesMoines.csv",
    "wind_farm_2_Chicago.csv",
    "wind_farm_3_Minneapolis.csv",
    "wind_farm_4_Omaha.csv",
    "wind_farm_5_Wichita.csv",
    "wind_farm_6_SiouxFalls.csv"
]
def get_wind_data_for_the_day(wind_farms, day = "2019-12-30"):
    wind_production = []
    for i in wind_files:
        df = pd.read_csv(i, delimiter=";")
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]]) #convert first column to datetime
        df_filtered = df[df[df.columns[0]].dt.strftime('%Y-%m-%d') == day]
        wind_production.append(df_filtered.iloc[:,2].to_numpy()/1000) #kW to MW
    return wind_production

wind_data=get_wind_data_for_the_day(wind_files, day = "2019-12-30" ) # Wind prduction for each wind farm on the specified day
wind_costs = np.zeros(len(wind_data))

########################################################
########################################################
########################################################
########################################################
########################################################

# GENERATOR DATA 
gen_unit = generator_data.iloc[:, 0].to_numpy()
gen_node =generator_data.iloc[:, 1].to_numpy()
gen_Pmax = generator_data.iloc[:, 2].to_numpy()
gen_Pmin = generator_data.iloc[:, 3].to_numpy()
gen_Rplus = generator_data.iloc[:, 4].to_numpy()
gen_Rminus = generator_data.iloc[:, 5].to_numpy()
gen_Ru = generator_data.iloc[:, 6].to_numpy()
gen_Rd = generator_data.iloc[:, 7].to_numpy()
gen_UT = generator_data.iloc[:, 8].to_numpy()
gen_DT = generator_data.iloc[:, 9].to_numpy()

# GENERATOR COSTS
gen_C = generator_costs.iloc[:, 1].to_numpy()
gen_Cu = generator_costs.iloc[:, 2].to_numpy()
gen_Cd = generator_costs.iloc[:, 3].to_numpy()
gen_Cplus = generator_costs.iloc[:, 4].to_numpy()
gen_Cminus = generator_costs.iloc[:, 5].to_numpy()
gen_Csu = generator_costs.iloc[:, 6].to_numpy()
gen_Pini = generator_costs.iloc[:, 7].to_numpy()
gen_Uini = generator_costs.iloc[:, 8].to_numpy()
gen_Tini = generator_costs.iloc[:, 9].to_numpy()

# TOTAL SYSTEN DEMAND OVER 24 HOURS
system_demand_hours = np.concatenate([load_profile.iloc[:, 0].to_numpy(), load_profile.iloc[:, 2].to_numpy()])
system_demand = np.concatenate([load_profile.iloc[:, 1].to_numpy(), load_profile.iloc[:, 3].to_numpy()])
# LOAD LOCATION AND DISTRIBUTION
load_number = np.concatenate([node_location_and_distribution.iloc[:, 0].to_numpy(), node_location_and_distribution.iloc[:, 3].to_numpy()])[:-1]
load_node = np.concatenate([node_location_and_distribution.iloc[:, 1].to_numpy(), node_location_and_distribution.iloc[:, 4].to_numpy()])[:-1]
load_percentage = np.concatenate([node_location_and_distribution.iloc[:, 2].to_numpy(), node_location_and_distribution.iloc[:, 5].to_numpy()])[:-1]/100

#TRANSMISSION DATA
transmission_from = np.concatenate([transmission_data.iloc[:, 0].to_numpy(), transmission_data.iloc[:, 4].to_numpy()])[:-1]
transmission_to = np.concatenate([transmission_data.iloc[:, 1].to_numpy(), transmission_data.iloc[:, 5].to_numpy()])[:-1]
transmission_reactance = np.concatenate([transmission_data.iloc[:, 2].to_numpy(), transmission_data.iloc[:, 6].to_numpy()])[:-1]
transmission_capacity = np.concatenate([transmission_data.iloc[:, 3].to_numpy(), transmission_data.iloc[:, 7].to_numpy()])[:-1]


# STORAGE PARAMETERS
Pch = 75  # Max charging power (MW)
Pdis = 73  # Max discharging power (MW)
E = 100   # Storage capacity (MWh)
eta_ch = 0.95  # Charging efficiency
eta_dis = 0.93  # Discharging efficiency

########################################################
########################################################
########################################################
########################################################
########################################################


# Number of hours
T = 24

# Number of conventional generators
num_conv_gens = len(gen_unit)  

# Input data
wind_production_data = np.array(wind_data).T
total_gen_capacities = np.column_stack((np.tile(gen_Pmax, (T,1)), wind_production_data))
demand_per_hour = np.array([load_percentage * system_demand[t] for t in range(T)])
total_gen_costs = np.concatenate([gen_C, wind_costs])
min_bid_price = min(total_gen_costs)
max_bid_price = max(total_gen_costs)
normalized_load = (demand_per_hour - demand_per_hour.min()) / (demand_per_hour.max() - demand_per_hour.min())
bid_prices = min_bid_price + normalized_load * (max_bid_price - min_bid_price)

# Function that maximizes total social welfare using Gurobi
def optimize_model(with_storage=True):
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
    
    return market_prices, dispatched_power, consumed_power, battery_energy, charging_power, discharging_power


# Run the optimization
[market_prices, pg, pd, SOC, p_ch, p_dis] = optimize_model(with_storage=True)
[market_prices_without_storage, pg_without_storage, pd_without_storage, SOC_without_storage, p_ch_without_storage, p_dis_without_storage] = optimize_model(with_storage=False)

# Generate plots
plt.figure()
plt.step(range(T), market_prices, where="post", label="Market price with storage", color = "blue", linewidth=2)
#plt.step(range(T), market_prices_without_storage, where="post", label="Market Price With Storage", color = "red", linewidth=2, linestyle="--")

plt.xlabel("Hour", fontsize=12)
plt.ylabel("Market-clearing price [$/MWh]", fontsize=12)    
plt.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, framealpha=0.8, bbox_to_anchor=(0.98, 1.05))
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("market_prices_with_storage.png", dpi=300, bbox_inches='tight')
plt.show()


# Total generator profits

generator_profits = []
for g in range(len(total_gen_costs)):  
    profit = sum(market_prices[t] * pg[t][g] - total_gen_costs[g] * pg[t][g] for t in range(T))
    generator_profits.append(profit)


# find battery profit over 24 hours
battery_profit = sum(market_prices[t] * SOC[t] - market_prices[t] * p_ch[t] / eta_ch for t in range(T))

print("\nGenerator profits:")
for g, profit in enumerate(generator_profits):
    print(f"Generator {g+1}: ${profit:,.2f}")

print(f"\nTotal battery profit: ${battery_profit:,.2f}")
