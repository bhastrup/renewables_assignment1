import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix


# **Step 1: Read Input Data**
print("Step 3: Network Constraints")

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
def get_wind_data_for_the_day(wind_farms, day = "2019-12-30", scale_factor = 1):
    wind_production = []
    for i in wind_files:
        df = pd.read_csv(i, delimiter=";")
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]]) #convert first column to datetime
        df_filtered = df[df[df.columns[0]].dt.strftime('%Y-%m-%d') == day]
        wind_production.append((df_filtered.iloc[:,2].to_numpy()/1000) * scale_factor) #kW to MW
    return wind_production

wind_data= get_wind_data_for_the_day(wind_files, day = "2019-12-30", scale_factor=1) # Wind prduction for each wind farm on the specified day
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
print(len(transmission_reactance))
transmission_reactance = np.full(33,0.002)
transmission_capacity = np.concatenate([transmission_data.iloc[:, 3].to_numpy(), transmission_data.iloc[:, 7].to_numpy()])[:-1]

# line capacity modifications
line_capacity_changes = {
    (15, 21): 400,
    (14, 16): 250,
    (13, 23): 250
}

for i in range(len(transmission_from)):
    bus_pair = (transmission_from[i], transmission_to[i])
    if bus_pair in line_capacity_changes:
        transmission_capacity[i] = line_capacity_changes[bus_pair]

#nodes where wind farms are located
wind_farm_nodes = np.array([3, 5, 7, 16, 21, 23])


########################################################
########################################################
########################################################
########################################################
########################################################

def run_dc_opf(line_15_21_capacity):
    # create a copy of original transmission data
    custom_capacity = transmission_capacity.copy()

    # locate index of line 15 21
    line_index = -1
    for i in range(len(transmission_from)):
        if (transmission_from[i] == 15 and transmission_to[i] == 21) or \
           (transmission_from[i] == 21 and transmission_to[i] == 15):
            line_index = i
            break

    if line_index == -1:
        raise ValueError("Line between buses 15 and 21 not found in the transmission data.")

    # Adjust the capacity for this line
    custom_capacity[line_index] = line_15_21_capacity
    capacity_pu = custom_capacity / base_MVA
    capacity_MW = capacity_pu * base_MVA  

    # Create and solve the model
    model = gp.Model(f"DC_OPF_line_15_21_{line_15_21_capacity}")
    model.setParam('OutputFlag', 0)  

    # decision variables
    pg = model.addVars(len(all_gen_capacities), lb=0, ub=all_gen_capacities, vtype=GRB.CONTINUOUS, name="p_g")
    pd = model.addVars(len(load_demand), lb=0, ub=load_demand, vtype=GRB.CONTINUOUS, name="p_d")
    theta = model.addVars(num_buses, lb=-np.pi, ub=np.pi, vtype=GRB.CONTINUOUS, name="theta")

    # objective function
    model.setObjective(
        gp.quicksum(bid_prices[d] * pd[d] for d in range(len(load_demand))) -
        gp.quicksum(all_gen_costs[g] * pg[g] for g in range(len(all_gen_capacities))), GRB.MAXIMIZE
    )

    # nodal power balance
    for b in range(num_buses):
        gen_power = gp.quicksum(pg[g] for g in range(len(all_gen_capacities)) if gen_bus_index[g] == b)
        load_power = gp.quicksum(pd[d] for d in range(len(load_demand)) if load_bus_index[d] == b)
        power_flow = gp.quicksum(B_matrix[b, m] * (theta[b] - theta[m]) for m in range(num_buses) if m != b)
        model.addConstr(gen_power - load_power == power_flow, f"nodal_balance_{b}")

    # reference bus
    model.addConstr(theta[0] == 0, "slack")

    #  power flow constraints
    for i in range(len(transmission_from)):
        from_bus = bus_map[transmission_from[i]]
        to_bus = bus_map[transmission_to[i]]
        flow = B_matrix[from_bus, to_bus] * (theta[from_bus] - theta[to_bus])
        model.addConstr(flow <= capacity_MW[i], f"flow_max_{i}")
        model.addConstr(flow >= -capacity_MW[i], f"flow_min_{i}")


    model.optimize()

    if model.status == GRB.OPTIMAL:
        mcp = {b: -model.getConstrByName(f"nodal_balance_{b}").Pi for b in range(num_buses)}
        gen_dispatch = [pg[g].X for g in range(len(all_gen_capacities))]
        load_consumption = [pd[d].X for d in range(len(load_demand))]
        theta_val = [theta[b].X for b in range(num_buses)]

        # power flow on line 15- 21
        from_bus = bus_map[15]
        to_bus = bus_map[21]
        line_flow = B_matrix[from_bus, to_bus] * (theta_val[from_bus] - theta_val[to_bus])

        # Total generator dispatch and load at bus 15
        bus15_idx = bus_map[15]
        gen_at_15 = sum(gen_dispatch[g] for g in range(len(all_gen_capacities)) if gen_bus_index[g] == bus15_idx)
        load_at_15 = sum(load_consumption[d] for d in range(len(load_demand)) if load_bus_index[d] == bus15_idx)
        g5_dispatch = pg[4].X  

        return {
            "mcp": mcp,
            "gen_dispatch": gen_dispatch,
            "load_consumption": load_consumption,
            "flow_15_21": line_flow,
            "bus15_gen": gen_at_15,
            "bus15_load": load_at_15,
            "g5_dispatch": g5_dispatch,
        }
    else:
        print(f"Optimization failed at {line_15_21_capacity}")
        return None

print("NODAL MARKET CLEARING PRICES")

# peak hour for the single hour setting
peak_demand = np.max(system_demand)
peak_idx = np.argmax(system_demand)
wind_production_at_peak_demand = np.array([i[17] for i in wind_data]) # get the wind production for each wind farm at peak hour

# total combined capacities and costs
total_gen_costs = np.concatenate((gen_C, wind_costs)) # Total generator costs
total_gen_capacities = np.concatenate((gen_Pmax, wind_production_at_peak_demand)) # Total generator capacities

# total MW on each load during hour 18 and bid prices
load_demand = load_percentage * peak_demand # the size of each load during hour 18 
min_bid_price = min(total_gen_costs)   
max_bid_price = max(total_gen_costs)   

# normalize the loads to the highest load 
normalized_load = (load_demand - min(load_demand)) / (max(load_demand) - min(load_demand)) # Normalized loads
bid_prices = min_bid_price + normalized_load * (max_bid_price - min_bid_price) # Bid prices - ensuring they're higher than costs

# System base MVA
base_MVA = 1000  

#  MVA to per unit 
transmission_capacity_pu = transmission_capacity / base_MVA

# convert back to MW using base MVA
transmission_capacity_MW = transmission_capacity_pu * base_MVA  # Since Base_MVA = 1000, this remains numerically the same

#network mapping
buses = np.unique(np.concatenate((transmission_from, transmission_to)))  # Unique bus numbers
num_buses = len(buses)
bus_map = {bus: i for i, bus in enumerate(buses)}  # Mmap bus numbers to indices

# include wind farms
all_gen_nodes = np.concatenate((gen_node, wind_farm_nodes))  #  wind farm buses
gen_bus_index = np.array([bus_map[b] if b in bus_map else -1 for b in all_gen_nodes])  # map generator buses
load_bus_index = np.array([bus_map[b] if b in bus_map else -1 for b in load_node])  #map load buses

# extend generation capacity and costs to include wind
all_gen_capacities = np.concatenate((gen_Pmax, wind_production_at_peak_demand))
all_gen_costs = np.concatenate((gen_C, wind_costs))

# susceptance matrix
B_matrix = np.zeros((num_buses, num_buses))

for i in range(len(transmission_from)):
    from_bus = bus_map[transmission_from[i]]
    to_bus = bus_map[transmission_to[i]]
    x = transmission_reactance[i]  # of the line reactance

    B_matrix[from_bus, to_bus] -= 1 / x
    B_matrix[to_bus, from_bus] -= 1 / x
    B_matrix[from_bus, from_bus] += 1 / x
    B_matrix[to_bus, to_bus] += 1 / x


# DC optimization problem
model = gp.Model("DC_OPF_Network")

pg = model.addVars(len(all_gen_capacities), lb=0, ub=all_gen_capacities, vtype=GRB.CONTINUOUS, name="p_g")
pd = model.addVars(len(load_demand), lb=0, ub=load_demand, vtype=GRB.CONTINUOUS, name="p_d")
theta = model.addVars(num_buses, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="theta")  # Phase angles

model.setObjective(
    gp.quicksum(bid_prices[d] * pd[d] for d in range(len(load_demand))) -
    gp.quicksum(all_gen_costs[g] * pg[g] for g in range(len(all_gen_capacities))), GRB.MAXIMIZE
)

for b in range(num_buses):
    gen_power = gp.quicksum(pg[g] for g in range(len(all_gen_capacities)) if gen_bus_index[g] == b)
    load_power = gp.quicksum(pd[d] for d in range(len(load_demand)) if load_bus_index[d] == b)

    power_flow = gp.quicksum(
        B_matrix[b, m] * (theta[b] - theta[m]) for m in range(num_buses) if m != b
    )
    model.addConstr(gen_power - load_power == power_flow, f"nodal_balance_{b}")

    
slack_bus = 0  # First bus as the slack reference
model.addConstr(theta[slack_bus] == 0, "slack_bus_angle")

for i in range(len(transmission_from)):
    from_bus = bus_map[transmission_from[i]]
    to_bus = bus_map[transmission_to[i]]

    flow = B_matrix[from_bus, to_bus] * (theta[from_bus] - theta[to_bus])

    model.addConstr(flow <= transmission_capacity_MW[i], f"flow_max_{i}")
    model.addConstr(flow >= -transmission_capacity_MW[i], f"flow_min_{i}")


model.optimize()

if model.status == GRB.OPTIMAL:
    nodal_mcp = {b: -model.getConstrByName(f"nodal_balance_{b}").Pi for b in range(num_buses)}

    print("\nNodal Market Clearing Prices (MCPs):")
    for b in range(num_buses):
        print(f"Bus {buses[b]}: MCP = {nodal_mcp[b]:.4f} $/MWh")

    print("\nGenerator Dispatch:")
    for g in range(len(all_gen_capacities)):
        print(f"Generator {g+1} at Bus {all_gen_nodes[g]}: {pg[g].X:.4f} MW")

    print("\nLoad Consumption Levels:")
    for d in range(len(load_demand)):
        print(f"Load Block {d+1} at Bus {load_node[d]}: {pd[d].X:.4f} MW")

    print("\nBus Voltage Angles:")
    for b in range(num_buses):
        print(f"Bus {buses[b]}: θ = {theta[b].X:.4f} rad")

else:
    print("Optimization did not find an optimal solution.")

nodal_mcp = {b+1: -model.getConstrByName(f"nodal_balance_{b}").Pi for b in range(num_buses)}


bus_indices_nodal = list(nodal_mcp.keys())
mcp_values_nodal = list(nodal_mcp.values())

bus_indices_nodal, mcp_values_nodal = zip(*sorted(zip(bus_indices_nodal, mcp_values_nodal)))

plt.figure(figsize=(10, 5))
plt.plot(bus_indices_nodal, mcp_values_nodal, marker='o', linestyle='-', color='b', markersize=6)
plt.xlabel("Bus Number")
plt.ylabel("Market Clearing Price (MCP) ($/MWh)")
plt.title("Nodal Market Clearing Prices (MCP) Across Buses")
plt.grid(True)
plt.xticks(bus_indices_nodal) 

plt.show()


# Sensitivity analysis (line 15-21)
capacities = np.linspace(50, 1000, 20)  # Try different line capacities (MW)
mcp_at_bus15 = []
flow_15_21 = []
bus15_gen = []
bus15_load = []
g5_dispatches = []
g7_dispatches = []


for cap in capacities:
    result = run_dc_opf(cap)
    if result:
        mcp_at_bus15.append(result["mcp"][bus_map[15]])
        flow_15_21.append(result["flow_15_21"])
        bus15_gen.append(result["bus15_gen"])
        bus15_load.append(result["bus15_load"])
        g5_dispatches.append(result["g5_dispatch"])
        g7_dispatches.append(result["g7_dispatch"])


plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(capacities, mcp_at_bus15, marker='o')
plt.title("Bus 15 market celaring price")
plt.xlabel("Line capacity [MW]")
plt.ylabel("MCP [USD/MWh]")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(capacities, flow_15_21, marker='s', color='orange')
plt.title("Power flow on line 15-21")
plt.xlabel("Line capacity [MW]")
plt.ylabel("Power flow [MW]")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(capacities, bus15_gen, marker='^', color='green')
plt.title("Bus 15 generator dispatch (G9)")
plt.xlabel("Line capacity [MW]")
plt.ylabel("Generation [MW]")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(capacities, bus15_load, marker='x', color='red')
plt.title("Bus 15 load (L13)")
plt.xlabel("Line capacity [MW]")
plt.ylabel("Load [MW]")
plt.grid(True)

plt.tight_layout()
plt.savefig("line_sensitivity_step3.png", dpi=300, bbox_inches='tight')

plt.show()

# === PROFIT AND SOCIAL WELFARE (NODAL) ===

gen_outputs = np.array([pg[g].X for g in range(len(all_gen_capacities))])
gen_costs = all_gen_costs.copy()

n_conventional = len(gen_C)
n_renewable = len(wind_costs)

gen_bus_prices = np.array([nodal_mcp[buses[gen_bus_index[g]]] for g in range(len(all_gen_capacities))])
gen_revenue = gen_outputs * gen_bus_prices
gen_total_cost = gen_outputs * gen_costs

profit_conventional = np.sum(gen_revenue[:n_conventional] - gen_total_cost[:n_conventional])
profit_renewable = np.sum(gen_revenue[n_conventional:])  # cost is zero

load_served = np.array([pd[d].X for d in range(len(load_demand))])
consumer_utility = np.sum(load_served * bid_prices)

social_welfare = consumer_utility - np.sum(gen_total_cost)

print("\n=== NODAL MARKET RESULTS ===")
print(f"Profit (Conventional): ${profit_conventional:.2f}")
print(f"Profit (Renewable):    ${profit_renewable:.2f}")
print(f"Total Generation Cost: ${np.sum(gen_total_cost):.2f}")
print(f"Total Consumer Utility: ${consumer_utility:.2f}")
print(f"Social Welfare:         ${social_welfare:.2f}")



#============ ZONAL =============

def run_zonal_market(atc_limit):
    print(f"\nRunning Zonal Market Clearing (ATC = {atc_limit} MW)...")

    zone1_buses = np.arange(1, 13)  
    zone2_buses = np.arange(13, 25)  
    zone_map = {bus: 1 if bus in zone1_buses else 2 for bus in buses}

    model = gp.Model("Zonal_Market")
    model.setParam('OutputFlag', 0)

    pg = model.addVars(len(all_gen_capacities), lb=0, ub=all_gen_capacities, vtype=GRB.CONTINUOUS, name="p_g")
    pd = model.addVars(len(load_demand), lb=0, ub=load_demand, vtype=GRB.CONTINUOUS, name="p_d")
    f12 = model.addVar(lb=-atc_limit, ub=atc_limit, vtype=GRB.CONTINUOUS, name="f12")  # Zonal flow from Zone 1 to 2

    model.setObjective(
        gp.quicksum(bid_prices[d] * pd[d] for d in range(len(load_demand))) -
        gp.quicksum(all_gen_costs[g] * pg[g] for g in range(len(all_gen_capacities))), GRB.MAXIMIZE
    )

    for zone in [1, 2]:
        gen_power = gp.quicksum(pg[g] for g in range(len(all_gen_capacities)) if zone_map[all_gen_nodes[g]] == zone)
        load_power = gp.quicksum(pd[d] for d in range(len(load_demand)) if zone_map[load_node[d]] == zone)

        if zone == 1:
            model.addConstr(gen_power - load_power == f12, name="zonal_balance_1")
        else:
            model.addConstr(gen_power - load_power == -f12, name="zonal_balance_2")

    model.optimize()

    if model.status == GRB.OPTIMAL:
        zonal_mcp = {
            "Zone 1": -model.getConstrByName("zonal_balance_1").Pi,
            "Zone 2": -model.getConstrByName("zonal_balance_2").Pi
        }

        gen_dispatch = np.array([pg[g].X for g in range(len(all_gen_capacities))])
        load_served = np.array([pd[d].X for d in range(len(load_demand))])

        gen_prices = np.array([
            zonal_mcp["Zone 1"] if zone_map[all_gen_nodes[g]] == 1 else zonal_mcp["Zone 2"]
            for g in range(len(all_gen_capacities))
        ])

        gen_revenue = gen_dispatch * gen_prices
        gen_cost = gen_dispatch * all_gen_costs

        n_conventional = len(gen_C)
        profit_conventional = np.sum(gen_revenue[:n_conventional] - gen_cost[:n_conventional])
        profit_renewable = np.sum(gen_revenue[n_conventional:])  # wind cost = 0

        consumer_utility = np.sum(load_served * bid_prices)

        social_welfare = consumer_utility - np.sum(gen_cost)

        print("\n=== ZONAL MARKET RESULTS ===")
        print(f"Zone 1 Price:           {zonal_mcp['Zone 1']:.2f} $/MWh")
        print(f"Zone 2 Price:           {zonal_mcp['Zone 2']:.2f} $/MWh")
        print(f"Profit (Conventional): ${profit_conventional:.2f}")
        print(f"Profit (Renewable):    ${profit_renewable:.2f}")
        print(f"Total Generation Cost: ${np.sum(gen_cost):.2f}")
        print(f"Total Consumer Utility: ${consumer_utility:.2f}")
        print(f"Social Welfare:         ${social_welfare:.2f}")

        return zonal_mcp
    else:
        print("Zonal optimization failed.")
        return None


def simulate_zonal_prices_for_demand(total_demand):
    ld = load_percentage * total_demand
    gen_cap = np.concatenate((gen_Pmax, wind_production_at_peak_demand))
    gen_cost = np.concatenate((gen_C, wind_costs))

    min_bid_price = min(gen_cost)
    max_bid_price = max(gen_cost)
    norm_ld = (ld - min(ld)) / (max(ld) - min(ld))
    bid_p = min_bid_price + norm_ld * (max_bid_price - min_bid_price)

    global load_demand, all_gen_capacities, all_gen_costs, bid_prices
    load_demand = ld
    all_gen_capacities = gen_cap
    all_gen_costs = gen_cost
    bid_prices = bid_p

    atc_values = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000, 2400]
    prices = []
    for atc in atc_values:
        result = run_zonal_market(atc)
        if result:
            prices.append((atc, result["Zone 1"], result["Zone 2"]))
    return zip(*prices)  # returns atc_vals, zone1_prices, zone2_prices

# Run for peak demand (original max)
#peak_demand = np.max(system_demand)
#atc_vals_peak, z1_peak, z2_peak = simulate_zonal_prices_for_demand(peak_demand)

# Run for reduced demand (first hour)
#reduced_demand = system_demand[0]
#atc_vals_low, z1_low, z2_low = simulate_zonal_prices_for_demand(reduced_demand)

# Plot side-by-side
fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

"""axs[0].plot(atc_vals_peak, z1_peak, label="Zone 1 market price", marker='o')
axs[0].plot(atc_vals_peak, z2_peak, label="Zone 2 market price", marker='s')
axs[0].set_title("Demand = 2650.5 MW")
axs[0].set_xlabel("ATC [MW]")
axs[0].set_ylabel("Zonal market price [USD/MWh]")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(atc_vals_low, z1_low, label="Zone 1 market price", marker='o')
axs[1].plot(atc_vals_low, z2_low, label="Zone 2 market price", marker='s')
axs[1].set_title(f"Demand = {int(reduced_demand)} MW")
axs[1].set_xlabel("ATC [MW]")
axs[1].legend()
axs[1].grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("zonal_step3.png", dpi=300, bbox_inches='tight')

plt.show()
"""


atc_values = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000, 2400]
zonal_price_results = []

for atc in atc_values:
    print(f"\n--- Zonal Market: ATC = {atc} MW ---")
    zonal_prices = run_zonal_market(atc)
    if zonal_prices:
        zonal_price_results.append((atc, zonal_prices["Zone 1"], zonal_prices["Zone 2"]))

atc_vals, zone1_prices, zone2_prices = zip(*zonal_price_results)

plt.figure(figsize=(10, 6))
plt.plot(atc_vals, zone1_prices, marker='o', label="Zone 1 Price")
plt.plot(atc_vals, zone2_prices, marker='s', label="Zone 2 Price")
plt.xlabel("ATC between Zone 1 and Zone 2 [MW]")
plt.ylabel("Zonal Market Price [$/MWh]")
plt.title("Zonal Market Prices vs ATC Capacity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()













