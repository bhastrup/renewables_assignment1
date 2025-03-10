import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# **Step 1: Read Input Data**
print("Step 1: Single Hour Copper Plate")

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

########################################################
########################################################
########################################################
########################################################
########################################################

#-----------Market clearing price under a uniform pricing scheme in a a single hour-------#
print("-"*80)
print("Determine the market clearing price under a uniform pricing scheme")

# find the peak hour for the single hour setting
peak_demand = np.max(system_demand) # 2650.5 MW
peak_idx = np.argmax(system_demand) # Hour 18
wind_production_at_peak_demand = np.array([i[17] for i in wind_data]) # get the wind production for each wind farm at peak hour

# Find total combined capacities and costs
total_gen_costs = np.concatenate((gen_C, wind_costs)) # Total generator costs
total_gen_capacities = np.concatenate((gen_Pmax, wind_production_at_peak_demand)) # Total generator capacities

# Find the total MW on each load during hour 18 and bid prices
load_demand = load_percentage * peak_demand # the size of each load during hour 18 (sums to peak_demand)
min_bid_price = min(total_gen_costs)   # Setting minimum bid price relatively high
max_bid_price = max(total_gen_costs)   # Increase maximum bid price to be significantly higher than costs

# Normalize the loads to the highest load 
normalized_load = (load_demand - min(load_demand)) / (max(load_demand) - min(load_demand)) # Normalized loads
bid_prices = min_bid_price + normalized_load * (max_bid_price - min_bid_price) # Bid prices - ensuring they're higher than costs

    
def plot_merit_order_curves_single_hour():
    
    #Supply curve
    sorted_idx_gen_costs = np.argsort(total_gen_costs) #identify generator cost indices
    sorted_total_gen_costs = total_gen_costs[sorted_idx_gen_costs] #Generator costs in ascending order
    sorted_total_gen_capacities = total_gen_capacities[sorted_idx_gen_costs] #Corresponding generator capacity
    cumulative_capacity = np.cumsum(sorted_total_gen_capacities) # Add cumulative capacities
    
    #demand curve
    sorted_idx_bid_prices = np.argsort(bid_prices)[::-1] #identify bid price indices
    sorted_bid_prices = bid_prices[sorted_idx_bid_prices] # Bid prices in descending order
    sorted_total_demand = load_demand[sorted_idx_bid_prices] #corresponding load demand
    cumulative_load_demand = np.cumsum(sorted_total_demand) # add cumulative load demand
    
    
    plt.style.use("default")
    
    plt.figure(figsize=(9, 5))
    
    plt.step(cumulative_capacity, sorted_total_gen_costs, where='post', 
             label='Cumulative generator capacity', color='blue', linewidth=2)
    plt.step(cumulative_load_demand, sorted_bid_prices, where='post', 
             label='Cumulative load demand', color='red', linewidth=2)
    market_clearing_price = 10.2170 
    plt.axhline(y=market_clearing_price, color='black', linestyle='dotted', linewidth=2, label='Market clearing price')
    plt.text(min(cumulative_capacity) + 5, market_clearing_price + 1, 
             f'MCP: {market_clearing_price:.2f} $/MWh', 
             fontsize=12, color='black', fontweight='bold')
    
    plt.xlabel("Cumulative capacity / demand [MW]", fontsize=12)
    plt.ylabel("Offer / bid price [$/MWh]", fontsize=12)    
    plt.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, framealpha=0.8)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("supply_demand.png", dpi=300, bbox_inches='tight')
    plt.show()

    
    return 

plot_merit_order_curves_single_hour()

print("The market clearing price can be seen in the intersection point in the supply demand curve")
print("-"*80)
print("Determine the social welfare of the system")

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

if model.status == GRB.OPTIMAL:
    mcp = -1*model.getConstrByName("power_balance").Pi #dual variable of the power balance
    print(f"Market clearing price: {mcp:.4f}")
    print("Gnerator dispatch:")
    for g in range(num_gen):
        print(f"Generator {g}: {pg[g].X:.4f} MW")
        
    for d in range(num_loads):
        print(f"Load {d}: {pd[d].X:.4f} MW")
else:
    print("Optimization did not find an optimal solution.")

print("-"*80)
print("Find the profit of each producer")

gen_profit = np.zeros(num_gen)
for g in range(num_gen):
    pg_dispatched = pg[g].X
    gen_profit[g] = (mcp-total_gen_costs[g]) * pg_dispatched 
    print(f"Generator {g} profit: {gen_profit[g]:.2f} USD")


print("-"*80)
print("Find the utility of each demand")

demand_utilities = np.zeros(num_loads)
for d in range(num_loads):
    consumed_power = pd[d].X
    demand_utilities[d] = (bid_prices[d] - mcp) * consumed_power
    print(f"Load {d} utility: {demand_utilities[d]:.2f} USD")
    
    











