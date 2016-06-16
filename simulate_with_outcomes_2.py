import numpy as np
import pandas as pd
from copy import copy
    
def subset_shuffle(an_array, shuffle_fraction):
    assert(type(an_array) == np.ndarray)
    shuffle_indices = np.random.rand(len(an_array)) < shuffle_fraction # Boolean indexing
    shuffle_part = an_array[shuffle_indices]
    np.random.shuffle(shuffle_part)
    an_array[shuffle_indices] = shuffle_part
    return an_array
    
def reassign(state_df, time_var, district, switch_fraction):
    class Text(object): 
        def __init__(self, string = ''):
            self.text = str(string)
        def append(self, string):
            self.text = self.text + str(string) + '\n'
        def __str__(self):
            return '\n\n\n' + self.text
    
    times = sorted(set(state_df[time_var]))
    last_df = state_df[state_df[time_var] == times[0]]
    last_assignments_from_orig = dict(zip(last_df['person'], last_df[district]))
    last_people = set(last_df['person'])
    
    text = Text(state_df['state'].values[0])
    text.append(last_assignments_from_orig)
    
    for t in times[1:]:
        indices = pd.Series(state_df[time_var] == t)        
        current_df = state_df[indices]
            
        current_people = set(current_df['person'])
        current_districts = set(current_df[district])
        current_assignments_from_orig = dict(zip(current_df['person'], 
                                                 current_df[district]))
        text.append(t)
        text.append('original assignments')
        text.append(current_assignments_from_orig)
        """
        Find people who are in the state in the current period and last period,
        AND are assigned to the same district in both period. 
        Find the districts they go with.
        """
        people_continuing_in_district \
                  = list(filter(lambda person: last_assignments_from_orig[person] \
                                            == current_assignments_from_orig[person],
                               current_people & last_people))
        text.append('people continuing in district: ')
        text.append(people_continuing_in_district)
        districts_with_continuing_people = [last_assignments_from_orig[person] 
                                    for person in people_continuing_in_district]
                          
        other_people = list(set(current_people) \
                       - set(people_continuing_in_district))
        text.append('new people:')
        text.append(other_people)
        other_districts = list(set(current_districts) \
                          - set(districts_with_continuing_people))
        
        text.append('old districts')
        text.append(districts_with_continuing_people)
        text.append('new districts')
        text.append(other_districts)
        """ 
        Create new assignments:
            - People who continue in the same district do so
            - Everyone else is randomly assigned to one of the other districts
        """
        np.random.shuffle(other_people)
        person_assignments = people_continuing_in_district + other_people
        district_assignments = districts_with_continuing_people + other_districts
#       Randomly switch some people
        new_person_assignments = subset_shuffle(np.array(person_assignments), 
                                                switch_fraction)
        assert(set(new_person_assignments) == current_people)
        assert(set(district_assignments) == current_districts)
        assignments = dict(zip(new_person_assignments, district_assignments))
        text.append('simulated assignments')
        text.append(assignments)
        # Update data with new changes
        state_df.loc[indices, district] = \
                       [assignments[person] for person in current_df['person']]
        
        last_assignments_from_orig = current_assignments_from_orig
        last_people = current_people

#    print(text)
    return state_df


#def reassign(state_df, time_var, district, switch_fraction):
#    state_df = state_df.reset_index()
#    times = sorted(set(state_df[time_var]))
#    last_df = state_df[state_df[time_var] == times[0]]
#    last_assignments = dict(zip(last_df['person'], last_df[district]))
#    last_people = set(last_df['person'])
#    
#    
#    for t in times[1:]:
#        current_df = state_df[state_df[time_var] == t]
#        indices = current_df.index.values
#        current_people = current_df['person']
#        current_districts = current_df[district]
#        current_assignments = dict(zip(current_people, current_districts))
#        
#        """
#        Find people who are in the state in the current period and last period,
#        AND are assigned to the same district in both period. 
#        Find the districts they go with.
#        """
#        people_continuing_in_district \
#                  = list(filter(lambda person: last_assignments[person] \
#                                           == current_assignments[person],
#                               set(current_people) & set(last_people)))

#        # Boolean indices, then integer indices
#        continuation_indices = [person in people_continuing_in_district 
#                                for person in current_people]
#        other_indices = [not x for x in continuation_indices]
#        continuation_indices = indices[np.array(continuation_indices)]
#        other_indices = indices[np.array(other_indices)]

# 
#        """ 
#        Create new assignments:
#            - People who continue in the same district do so
#            - Everyone else is randomly assigned to one of the other districts
#        """
#        district_indices = np.concatenate((continuation_indices, other_indices))
#        np.random.shuffle(other_indices)
#        person_indices = np.concatenate((continuation_indices, other_indices))
#        # Randomly switch some people
#        person_indices = subset_shuffle(np.array(person_indices), switch_fraction)
#        # Update data with new changes
#        last_people = current_people[person_indices]
#        last_districts = current_districts[district_indices]
#        state_df.iloc[indices][['person', district]] = np.vstack((last_people,
#                                                                  last_districts)).T
#        last_assignments = dict(zip(last_people, last_districts))

#        
#    return state_df

    
def simulate(df, seed_increment, switch_fraction, time_var):
    np.random.seed(seed_increment)
    district = 'clean district name'
    return df.groupby('state').apply(lambda x: \
                                     reassign(x, time_var, district, switch_fraction))
    
