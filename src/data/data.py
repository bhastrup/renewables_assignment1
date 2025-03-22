import pandas as pd
import numpy as np

def load_raw_data(dataset_dir, wind_dir): 

    # Load CSV files into Pandas DataFrames
    generator_data = pd.read_csv(dataset_dir + "generator_data_24.csv")
    generator_costs = pd.read_csv(dataset_dir + "generator_costs_24.csv")
    load_profile = pd.read_csv(dataset_dir + "load_profile_24.csv")
    node_location_and_distribution = pd.read_csv(dataset_dir + "node_location_and_distribution_24.csv")
    transmission_data = pd.read_csv(dataset_dir + "/transmission_data_24.csv")

    
    # WIND DATA for the day and wind costs
    wind_files = [
        wind_dir + "wind_farm_1_DesMoines.csv",
        wind_dir + "wind_farm_2_Chicago.csv",
        wind_dir + "wind_farm_3_Minneapolis.csv",
        wind_dir + "wind_farm_4_Omaha.csv",
        wind_dir + "wind_farm_5_Wichita.csv",
        wind_dir + "wind_farm_6_SiouxFalls.csv"
    ]
    def get_wind_data_for_the_day(wind_files, day = "2019-12-30"):
        wind_production = []
        for i in wind_files:
            df = pd.read_csv(i, delimiter=";")
            df[df.columns[0]] = pd.to_datetime(df[df.columns[0]]) #convert first column to datetime
            df_filtered = df[df[df.columns[0]].dt.strftime('%Y-%m-%d') == day]
            wind_production.append(df_filtered.iloc[:,2].to_numpy()/1000) #kW to MW
        return wind_production

    wind_data=get_wind_data_for_the_day(wind_files=wind_files, day = "2019-12-30" ) # Wind prduction for each wind farm on the specified day


    return generator_data, generator_costs, load_profile, node_location_and_distribution, transmission_data, wind_data


def process_generator_data(generator_data):
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

    return gen_unit, gen_node, gen_Pmax, gen_Pmin, gen_Rplus, gen_Rminus, gen_Ru, gen_Rd, gen_UT, gen_DT

def process_generator_costs(generator_costs):
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

    return gen_C, gen_Cu, gen_Cd, gen_Cplus, gen_Cminus, gen_Csu, gen_Pini, gen_Uini, gen_Tini

def process_load_profile(load_profile):
    # TOTAL SYSTEN DEMAND OVER 24 HOURS
    system_demand_hours = np.concatenate([load_profile.iloc[:, 0].to_numpy(), load_profile.iloc[:, 2].to_numpy()])
    system_demand = np.concatenate([load_profile.iloc[:, 1].to_numpy(), load_profile.iloc[:, 3].to_numpy()])

    return system_demand_hours, system_demand

def process_node_location_and_distribution(node_location_and_distribution):
    # LOAD LOCATION AND DISTRIBUTION
    load_number = np.concatenate([node_location_and_distribution.iloc[:, 0].to_numpy(), node_location_and_distribution.iloc[:, 3].to_numpy()])[:-1]
    load_node = np.concatenate([node_location_and_distribution.iloc[:, 1].to_numpy(), node_location_and_distribution.iloc[:, 4].to_numpy()])[:-1]
    load_percentage = np.concatenate([node_location_and_distribution.iloc[:, 2].to_numpy(), node_location_and_distribution.iloc[:, 5].to_numpy()])[:-1]/100

    return load_number, load_node, load_percentage

def process_transmission_data(transmission_data):
    #TRANSMISSION DATA
    transmission_from = np.concatenate([transmission_data.iloc[:, 0].to_numpy(), transmission_data.iloc[:, 4].to_numpy()])[:-1]
    transmission_to = np.concatenate([transmission_data.iloc[:, 1].to_numpy(), transmission_data.iloc[:, 5].to_numpy()])[:-1]
    transmission_reactance = np.concatenate([transmission_data.iloc[:, 2].to_numpy(), transmission_data.iloc[:, 6].to_numpy()])[:-1]
    transmission_capacity = np.concatenate([transmission_data.iloc[:, 3].to_numpy(), transmission_data.iloc[:, 7].to_numpy()])[:-1]

    return transmission_from, transmission_to, transmission_reactance, transmission_capacity



if __name__ == "__main__":

    dataset_dir = "datasets/"
    wind_dir = dataset_dir + "wind_farms/"

    (
        generator_data, 
        generator_costs, 
        load_profile, 
        node_location_and_distribution, 
        transmission_data, 
        wind_data
    ) = load_raw_data(dataset_dir, wind_dir)


    print(generator_data)
    print(generator_costs)
    print(load_profile)
    print(node_location_and_distribution)
    print(transmission_data)


    # Process the data

    # GENERATOR DATA 
    (
        gen_unit, gen_node, gen_Pmax, gen_Pmin, gen_Rplus, gen_Rminus, gen_Ru, gen_Rd, gen_UT, gen_DT
    ) = process_generator_data(generator_data)

    # GENERATOR COSTS
    (
        gen_C, gen_Cu, gen_Cd, gen_Cplus, gen_Cminus, gen_Csu, gen_Pini, gen_Uini, gen_Tini
    ) = process_generator_costs(generator_costs)

    # LOAD PROFILE
    system_demand_hours, system_demand = process_load_profile(load_profile)

    # LOAD LOCATION AND DISTRIBUTION
    (
        load_number, load_node, load_percentage
    ) = process_node_location_and_distribution(node_location_and_distribution)

    # TRANSMISSION DATA
    (
        transmission_from, transmission_to, transmission_reactance, transmission_capacity
    ) = process_transmission_data(transmission_data)



