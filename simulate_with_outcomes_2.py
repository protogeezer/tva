import numpy as np
import pandas as pd
#from copy import copy

#def find_swappers(df):
#    switch_people = df.loc[df['switch'], 'person'].values
#    new_people = copy(switch_people)
#    np.random.shuffle(new_people)
#    return dict(zip(switch_people, new_people))

#def simulate(df, seed_increment, switch_fraction, time_var):
#             
#    np.random.seed(seed_increment)
#    times = sorted(set(df[time_var]))
#    df['switch'] = np.random.random(len(df)) < switch_fraction
#    
#    for t in times[1:]:
#        # find out who needs to swap with who
#        # first column of swaps is original, second is replacement
#        swaps = df[df[time_var] == t].groupby('state').apply(find_swappers).values
#        swaps = {k:v for d in swaps for k, v in d.items()}
#        
#        # Replace bureaucrats in all future times
#        indices = [s >= t and p in swaps for s, p in 
#                   df[[time_var, 'person']].values]
#        df.loc[indices, 'person'] = [swaps[p] 
#                                   for p in df.loc[indices, 'person'].values]
#        
#    return df
    
def subset_shuffle(an_array, shuffle_fraction):
    assert(type(an_array) == np.ndarray)
    shuffle_indices = np.random.rand(len(an_array)) < shuffle_fraction # Boolean indexing
    shuffle_part = an_array[shuffle_indices]
    np.random.shuffle(shuffle_part)
    an_array[shuffle_indices] = shuffle_part
    return an_array
    
def reassign(state_df, time_var, district, switch_fraction):
    times = sorted(set(state_df[time_var]))
    last_df = state_df[state_df[time_var] == times[0]]
    last_assignments = dict(zip(last_df['person'], last_df[district]))
    last_people = set(last_df['person'])
    
    for t in times[1:]:
        indices = pd.Series(state_df[time_var] == t)        
        current_df = state_df[indices]
            
        current_people = set(current_df['person'])
        current_districts = set(current_df[district])
        current_assignments = dict(zip(current_df['person'], current_df[district]))
        
        """
        Find people who are in the state in the current period and last period,
        AND are assigned to the same district in both period. 
        Find the districts they go with.
        """
        people_continuing_in_district \
                  = list(filter(lambda person: last_assignments[person] \
                                           == current_assignments[person],
                               current_people & last_people))
        districts_with_continuing_people = \
             [last_assignments[person] for person in people_continuing_in_district]
                          
        other_people = list(set(current_people) \
                       - set(people_continuing_in_district))
        other_districts = list(set(current_districts) \
                          - set(districts_with_continuing_people))
        
        """ 
        Create new assignments:
            - People who continue in the same district do so
            - Everyone else is assigned to one o fthe other districts
        """
        person_assignments = people_continuing_in_district + other_people
        district_assignments = districts_with_continuing_people + other_districts
        # Randomly switch some people
        new_person_assignments = subset_shuffle(np.array(person_assignments), 
                                                switch_fraction)
        assert(set(new_person_assignments) == current_people)
        assert(set(district_assignments) == current_districts)
        current_assignments = dict(zip(new_person_assignments, district_assignments))
        # Update data with new changes
        state_df.loc[indices, district] = \
                       [current_assignments[person] for person in current_df['person']]
        
        last_assignments, last_people = current_assignments, current_people
    return state_df
    
def simulate(df, seed_increment, switch_fraction, time_var):
    np.random.seed(seed_increment)
    district = 'clean district name'
    return df.groupby('state').apply(lambda x: \
                                     reassign(x, time_var, district, switch_fraction))
    
#    for state in set(df['state']):
#        times = sorted(set(df.loc[df['state'] == state, time_var]))
#        last_df = df[(df['state'] == state) & (df[time_var] == times[0])]
#        last_assignments = dict(zip(last_df['person'], last_df[district]))
#        last_people = set(last_df['person'])
#        
#        for t in times[1:]:
#            indices = pd.Series((df['state'] == state) & (df[time_var] == t))            
#            current_df = df[indices]
#                
#            current_people = set(current_df['person'])
#            current_districts = set(current_df[district])
#            current_assignments = dict(zip(current_df['person'], current_df[district]))
#            
#            """
#            Find people who are in the state in the current period and last period,
#            AND are assigned to the same district in both period. 
#            Find the districts they go with.
#            """
#            people_continuing_in_district \
#                      = list(filter(lambda person: last_assignments[person] \
#                                               == current_assignments[person],
#                                   current_people & last_people))
#            districts_with_continuing_people = \
#                 [last_assignments[person] for person in people_continuing_in_district]
#                              
#            other_people = list(set(current_people) \
#                           - set(people_continuing_in_district))
#            other_districts = list(set(current_districts) \
#                              - set(districts_with_continuing_people))
#            
#            """ 
#            Create new assignments:
#                - People who continue in the same district do so
#                - Everyone else is assigned to one o fthe other districts
#            """
#            person_assignments = people_continuing_in_district + other_people
#            district_assignments = districts_with_continuing_people + other_districts
#            # Randomly switch some people
#            new_person_assignments = subset_shuffle(np.array(person_assignments), 
#                                                    switch_fraction)
#            assert(set(new_person_assignments) == current_people)
#            assert(set(district_assignments) == current_districts)
#            current_assignments = dict(zip(new_person_assignments, district_assignments))
#            # Update data with new changes
#            df.loc[indices, district] = \
#                           [current_assignments[person] for person in current_df['person']]
#            
#            last_assignments, last_people = current_assignments, current_people
#    return df

