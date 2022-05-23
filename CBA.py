import numpy as np
import pandas as pd
from sqlalchemy import intersect
import streamlit as st
import inputs
import demand_calcs

#TODO see if there's a neater way to do this
def outer_product(df1,df2):
    """Return the matrix outer product of two single column, single index dataframes with a 2-level Multiindex
    matching the column index names."""
    matrix = np.outer(df1,df2)
    matrix_dataframe = pd.DataFrame(
        matrix,
        index=df1.index.tolist(),
        columns=df2.index.tolist()
        )
    flat = matrix_dataframe.stack()
    flat_dataframe = pd.DataFrame(flat,columns=df1.columns.tolist())
    flat_dataframe.index.set_names([df1.index.name,df2.index.name],inplace=True)
    return flat_dataframe

#Function for writing benefits
def write_to_benefit_table(benefit,benefit_name,benefit_table):
    """Return the benefit table with an additional benefit concatenated to it."""
    benefit['benefit'] = benefit_name
    benefit.set_index('benefit',append=True,inplace=True)
    return pd.concat([benefit_table,benefit])

def calculate_benefits(demand, trip_distance_sensitivity=1, transport_share_sensitivity=0):
    """Calculates CBA benefits based on an input demand frame and input variables in inputs.py"""
    
    
    ### set up change in distance by surface type frames

    trip_distance = inputs.trip_distance_raw * trip_distance_sensitivity

    if inputs.subtract_project_length:
        #Apply surface distances to trip distance MINUS facility length
        off_facility_trip_distance = trip_distance-inputs.facility_length
        on_facility_trip_distance = trip_distance*0 + inputs.facility_length


        #If trip length is less than facility length assume whole trip is on facility
        for mode in off_facility_trip_distance.index:
            if off_facility_trip_distance.loc[mode,'value'] < 0:
                
                on_facility_trip_distance.loc[mode,'value'] = (
                    on_facility_trip_distance.loc[mode,'value']
                    + off_facility_trip_distance.loc[mode,'value']
                    )
               
                off_facility_trip_distance.loc[mode,'value'] = 0

        #Multiply off-facility distance by surface proportions
        off_facility_distance_perceived_base = outer_product(
            off_facility_trip_distance,
            inputs.surface_distance_prop_base/100
            )

        
        #Set up frame of on-facility distance by surface
        on_facility_trip_distance_base = on_facility_trip_distance.copy()
        on_facility_trip_distance_base['surface']=inputs.facility_type_existing
        on_facility_trip_distance_base.set_index('surface',append=True,inplace=True)
        

        #Add on and off-facility distance
        distance_perceived_base = (
            off_facility_distance_perceived_base
            .add(on_facility_trip_distance_base,
            fill_value=0)
            )

        #Multiply off-facility distance by surface proportions
        off_facility_distance_perceived_project = outer_product(
            off_facility_trip_distance+inputs.trip_distance_change
            ,inputs.surface_distance_prop_project/100
            )

        
        #Set up frame of on-facility distance by surface
        on_facility_trip_distance_project = on_facility_trip_distance.copy()
        on_facility_trip_distance_project['surface']=inputs.facility_type_new
        on_facility_trip_distance_project.set_index('surface',append=True,inplace=True)

        
        #Add on and off-facility distance
        distance_perceived_project = (
            off_facility_distance_perceived_project.
            add(on_facility_trip_distance_project,
            fill_value=0)
            )

    else:
        #Apply surface distances to whole trip distance
        distance_perceived_base = outer_product(
        trip_distance,
        inputs.surface_distance_prop_base/100
        )
        distance_perceived_project = outer_product(
        trip_distance + inputs.trip_distance_change,
        inputs.surface_distance_prop_project/100
        )

    distance_perceived_change = distance_perceived_project - distance_perceived_base

    ### Change in travel time costs (perceived)

    travel_time_perceived_change = distance_perceived_change/inputs.speed_active
    travel_time_perceived_change_per_trip = (
        travel_time_perceived_change.groupby(level='mode').sum()
        + inputs.time_saving/60/60
        )
    change_in_travel_time_cost_per_trip = travel_time_perceived_change_per_trip*inputs.vott

    ### Change in crash risk (perceived)

    change_in_crash_cost_per_trip = ((
        distance_perceived_change
        *inputs.relative_risk
        *inputs.crash_cost_active
        )
        ).groupby(level='mode').sum()



    ### Change in VOC (perceived)

    change_in_voc_per_trip = (
        distance_perceived_change.groupby(level='mode').sum()
        *inputs.voc_active
        )

    ###Set up the rule of a half
    rule_of_a_half = pd.DataFrame(np.full(
        inputs.diversion_rate.loc['Bicycle'].shape,
        0.5),
        index=inputs.diversion_rate.loc['Bicycle'].index,columns=['value']
        )

    rule_of_a_half.loc['Reassign']=1

    benefits = pd.DataFrame()

    ## Get total perceived benefits (apply trip numbers, transport share and the rule of a half)

    #ensure transport share sensitivity doesn't push transport share outside 0 and 100
    adjusted_transport_share = inputs.transport_share+transport_share_sensitivity
    for mode in adjusted_transport_share.index:
        if adjusted_transport_share.loc[mode,'value'] > 100:
            adjusted_transport_share.loc[mode,'value'] = 100
        if adjusted_transport_share.loc[mode,'value'] < 0:
            adjusted_transport_share.loc[mode,'value'] = 0

    travel_time_benefits = (
        demand
        *(adjusted_transport_share/100)
        *change_in_travel_time_cost_per_trip*rule_of_a_half
        *-1
        ) #travel time benefits only for transport purpose trips

    crash_cost_benefits = (
        demand
        *change_in_crash_cost_per_trip
        *rule_of_a_half
        *-1
        )

    voc_cost_benefits = (
        demand
        *(adjusted_transport_share/100)
        *change_in_voc_per_trip
        *rule_of_a_half
        *-1
        ) #voc cost benefits only for transport purpose trips

    benefits = write_to_benefit_table(travel_time_benefits,'Travel time',benefits)
    benefits = write_to_benefit_table(crash_cost_benefits,'Crash costs',benefits)
    benefits = write_to_benefit_table(voc_cost_benefits,'Vehicle operating costs',benefits)

    #Intersection safety
    if inputs.number_of_intersections>0:
        intersection_safety =[]

        
        for i in list(range(inputs.number_of_intersections)):
            paras = inputs.intersection_inputs.loc[i]

            
            
            intersection_safety.append(
                (
                    paras.loc['Expected 10-year fatalities (base case)','value']
                    *inputs.injury_cost.loc['Fatality','value']
                    +paras.loc['Expected 10-year injuries (base case)','value']
                    *inputs.injury_cost.loc['Serious Injury','value']
                    )
                *paras.loc['Risk reduction % (project case)','value']
                /100
            )
        
        intersection_safety = (sum(intersection_safety)/10).values[0]
        intersection_safety = intersection_safety/demand.groupby('year').sum()
        intersection_safety = intersection_safety*demand*rule_of_a_half

        benefits = write_to_benefit_table(intersection_safety,'Intersection safety',benefits)

    ### Health system benefits (unpercieved)

    health_system_benefits = (
        distance_perceived_project.groupby(level='mode').sum()
        *inputs.health_system
        *demand
        )
    
    # Health system benefits are zero for reassign
    health_system_benefits = health_system_benefits.swaplevel('mode','from_mode')
    health_system_benefits.loc['Reassign'] = 0
    health_system_benefits = health_system_benefits.swaplevel('from_mode','mode')

    benefits = write_to_benefit_table(health_system_benefits,'Health system costs',benefits)

    ## Avoided car costs (unperceived)

    from_car_demand = demand.swaplevel('mode','from_mode').loc['Car']

    per_trip_avoided_car_externalities= outer_product(trip_distance,inputs.car_externalities)
    car_benefits = from_car_demand*per_trip_avoided_car_externalities
    car_benefits.index.rename(['mode','year','benefit'],inplace=True)

    avoided_congestion_per_km = ((
        (inputs.diversion_congestion/100)
        *inputs.congestion_cost)
        .sum()
        )

    avoided_congestion_per_trip = avoided_congestion_per_km*trip_distance
    congestion_benefits = from_car_demand*avoided_congestion_per_trip
    congestion_benefits['benefit'] = 'Congestion'
    congestion_benefits.set_index('benefit',append=True,inplace=True)

    # road_provision_benefits = from_car_demand*inputs.road_provision
    # road_provision_benefits['benefit'] = 'Road provision costs'
    # road_provision_benefits.set_index('benefit',append=True,inplace=True)

    car_benefits = pd.concat([car_benefits,congestion_benefits])
    car_benefits = car_benefits.reset_index()
    car_benefits.insert(1,'from_mode','Car')
    car_benefits = car_benefits.set_index(['mode','from_mode','year','benefit'])

    benefits = pd.concat([benefits,car_benefits])

    return benefits



def discount_benefits(benefits_frame,discount_rate):
    """Applies the discount rate to a benefits frame"""

    #Set up discount factors
    discount_factor = pd.DataFrame()
    for yr in inputs.year:
        discount_factor.loc[yr,'value'] = (
            1/
            ((1+discount_rate/100)
            **(yr-int(inputs.start_year)))
            )
    discount_factor = discount_factor.rename_axis('year')

    #discount benefits
    discounted_benefits = discount_factor*benefits_frame

    return discounted_benefits




def discount_costs(costs_frame,discount_rate):
    """Applies the discount rate to a costs frame"""

    #Set up discount factors
    discount_factor = pd.DataFrame()
    for yr in inputs.year:
        discount_factor.loc[yr,'value'] = (
            1/
            ((1+discount_rate/100)
            **(yr-int(inputs.start_year)))
            )
    discount_factor = discount_factor.rename_axis('year')
    
    discounted_costs = discount_factor*costs_frame

    return discounted_costs



def calculate_results(discounted_benefits_frame,discounted_costs_frame,benefits_sensitivity=1,opex_sensitivity=1,capex_sensitivity=1):
    """Return NPV, BCR1 and BCR2 from discounted benefits and costs frames"""
    
    #Calculate NPV and BCR
    ben = discounted_benefits_frame.sum()*benefits_sensitivity
    opex = discounted_costs_frame.groupby('cost').sum().loc['operating_cost']*opex_sensitivity
    capex = discounted_costs_frame.groupby('cost').sum().loc['capital_cost']*capex_sensitivity

    NPV = ben - opex - capex
    BCR1 = ben / (opex + capex)
    BCR2 = (ben + opex) / capex

    results = {
        "NPV" : NPV['value'],
        "BCR1" : BCR1['value'],
        "BCR2" : BCR2['value']
    }

    return results




def do_sensitivity_CBA(demand_sensitivity=1, trip_distance_sensitivity=1, transport_share_sensitivity=0, new_trips_sensitivity=1,discount_rate=None,benefits_sensitivity=1,opex_sensitivity=1,capex_sensitivity=1):
    """Do a CBA and return NPV, BCR1 and BCR2. Input a single parameter value for a single sensitivity test"""
    
    if discount_rate is None:
        discount_rate = inputs.discount_rate

    basic_demand = demand_calcs.get_basic_demand_frame(demand_sensitivity)

    demand = demand_calcs.get_demand_frame(basic_demand, new_trips_sensitivity)

    bens = calculate_benefits(
        demand,
        trip_distance_sensitivity=trip_distance_sensitivity,
        transport_share_sensitivity=transport_share_sensitivity,
        )
    
    discount_b = discount_benefits(bens,discount_rate=discount_rate)

    discount_c = discount_costs(inputs.costs,discount_rate=discount_rate)

    res = calculate_results(
        discount_b,
        discount_c,
        benefits_sensitivity=benefits_sensitivity,
        opex_sensitivity=opex_sensitivity,
        capex_sensitivity=capex_sensitivity
        )

    return res


# def get_user_flows(demand, discounted_benefits):
#     """Returns breakdown of total benefits from CBA results. Perceived benefits are calculated compared to the base case mode
#     instead of using the rule of a half. Implicit Mode Benefit is a balancing item to ensure total CBA net benefits by mode
#     are equal to user flows by mode"""

#     reassign_distance_base = outer_product(
#         inputs.trip_distance_raw,
#         inputs.surface_distance_prop_base/100
#         )
   
#     project_trip_distance = inputs.trip_distance_raw + inputs.trip_distance_change
    
#     distance_project = outer_product(
#         project_trip_distance,
#         inputs.surface_distance_prop_project/100
#         )

#     user_flows = pd.DataFrame()

#     #Difference in travel time between project case and base case (as if travel time were unperceived)
   
#     #calculate trip times for non mode-shift in base case
#     reassign_time_base = (
#         (reassign_distance_base/inputs.speed_active)
#         .groupby(level='mode').sum()) 
#     reassign_time_base.insert(0,'from_mode','Reassign') #insert "from_mode" index
#     reassign_time_base.set_index('from_mode',append=True,inplace=True)

#     #Calculate trip times for mode shift in base case
#     from_time_base = outer_product(inputs.trip_distance_raw,1/inputs.speed_from_mode) 
#     time_base = from_time_base.append(reassign_time_base).sort_index()

#     #Calculate trip times in project case
#     time_project = (distance_project/inputs.speed_active).groupby(level='mode').sum()

#     time_change = time_project - time_base
#     per_trip_travel_time_cost_change = time_change*inputs.vott

#     travel_time_benefits = (
#         demand
#         *(inputs.transport_share/100)
#         *per_trip_travel_time_cost_change
#         *-1
#         )

#     user_flows = write_to_benefit_table(travel_time_benefits,'Travel time',user_flows)

#     #Perceived health benefits

#     #Calculate reassign health benefits in base case
#     reassign_perceived_health_base = inputs.trip_distance_raw*inputs.health_private 
#     reassign_perceived_health_base.insert(0,'from_mode','Reassign')
#     reassign_perceived_health_base.set_index('from_mode',append=True,inplace=True)

#     from_mode_perceived_health_base = from_time_base*0 #health benefits are nil in the base case for from_modes
#     perceived_health_base = from_mode_perceived_health_base.append(reassign_perceived_health_base).sort_index()
#     perceived_health_project = project_trip_distance*inputs.health_private
#     per_trip_perceived_health_change = perceived_health_project - perceived_health_base

#     perceived_health_benefits = demand*per_trip_perceived_health_change

#     user_flows = write_to_benefit_table(perceived_health_benefits,'Mortality and morbidity benefits',user_flows)

#     #Change in safety

#     #Calculate per trip crash cost for reassign in base case
#     reassign_crash_cost_base = (
#         (reassign_distance_base
#         *inputs.relative_risk
#         *inputs.crash_cost_active)
#         .groupby(level='mode').sum())
    
#     #Copy this DF to use later for induced crash costs
#     induced_crash_cost_base = reassign_crash_cost_base.copy()

#     reassign_crash_cost_base.insert(0,'from_mode','Reassign') #insert "from_mode" index
    
#     reassign_crash_cost_base.set_index('from_mode',append=True,inplace=True)

#     #Per trip crash cost for pt and car
#     from_mode_crash_cost_base = outer_product(inputs.trip_distance_raw,inputs.crash_cost_from_mode) 

#     #Crash costs are zero for induced trips (It's very safe at home)
#     induced_crash_cost_base = induced_crash_cost_base*0 

#     induced_crash_cost_base.insert(0,'from_mode','Induced') #insert "from_mode" index
#     induced_crash_cost_base.set_index('from_mode',append=True,inplace=True)

#     #append all from_modes
#     crash_cost_per_trip_base = (from_mode_crash_cost_base
#     .append(reassign_crash_cost_base)
#     .append(induced_crash_cost_base)
#     .sort_index()
#     )

#     crash_cost_per_trip_project = (
#         (distance_project
#         *inputs.relative_risk
#         *inputs.crash_cost_active)
#         .groupby(level='mode').sum()
#         )

#     change_in_crash_cost_per_trip = crash_cost_per_trip_project - crash_cost_per_trip_base

#     crash_cost = demand*change_in_crash_cost_per_trip*-1

#     user_flows = write_to_benefit_table(crash_cost,'Crash cost',user_flows)



#     #Change in money cost

#     money_cost_per_km = inputs.diversion_rate.copy()
#     money_cost_per_km = money_cost_per_km*0 # Set per_km to zero to start (marginal pt resource cost = 0, induced cost = 0)
#     for mode in inputs.voc_active.index.tolist():
#         money_cost_per_km.loc[(mode,'Reassign')] = inputs.voc_active.loc[mode] 
#         money_cost_per_km.loc[(mode,'Car')] = inputs.voc_car + inputs.parking_cost

#     money_cost_per_trip_base = inputs.trip_distance_raw*money_cost_per_km

#     money_cost_per_trip_project = project_trip_distance*inputs.voc_active

#     money_cost_change = money_cost_per_trip_project - money_cost_per_trip_base

#     money_cost_benefits = demand*money_cost_change*-1

#     user_flows = write_to_benefit_table(money_cost_benefits,'User monetary costs',user_flows)

#     discounted_user_flows = discount_benefits(user_flows,inputs.discount_rate).groupby(level=('mode','from_mode','benefit')).sum()

#     benefit_flows = discounted_benefits.groupby(level=('mode','from_mode','benefit')).sum()
#     user_benefit_flows = (
#         benefit_flows.xs('Travel time',level='benefit')
#         +benefit_flows.xs('Vehicle operating costs',level='benefit')
#         +benefit_flows.xs('Crash costs',level='benefit')
#         )
       
#     implicit_mode_benefit = user_benefit_flows - discounted_user_flows.groupby(level=('mode','from_mode')).sum()
#     implicit_mode_benefit.insert(0,'benefit','Mode preference')
#     implicit_mode_benefit.set_index('benefit',append=True,inplace=True)
#     discounted_user_flows = discounted_user_flows.append(implicit_mode_benefit).sort_index()

#     external_benefit_list = [
#         'Air quality',
#         'Congestion',
#         'Greenhouse gas emissions',
#         'Health system costs',
#         'Noise',
#         'Road provision costs'
#         ]
#     for externality in external_benefit_list:
#         discounted_user_flows = discounted_user_flows.append(benefit_flows.xs(externality,level='benefit',drop_level=False))
#     discounted_user_flows.sort_index(inplace=True)
#     return discounted_user_flows

