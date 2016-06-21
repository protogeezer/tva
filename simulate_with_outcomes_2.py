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
    last_assignments_from_orig = dict(zip(last_df[district], last_df['person']))
    last_districts = set(last_df[district])
    
    text = Text(state_df['state'].values[0])
    text.append(last_assignments_from_orig)
    
    for t in times[1:]:
        indices = pd.Series(state_df[time_var] == t)        
        current_df = state_df[indices]
            
        current_people = set(current_df['person'])
        if len(current_people) != len(current_df): # someone is duplicated; drop duplicates
            _, ind = np.unique(current_df['person'].values, return_index = True)
#            print(ind)
#            print(current_df[['person', district]])
#            print(current_df.index)
#            print(current_df.index[ind])
            current_df = current_df.iloc[ind]
#            print(current_df[['person', district]])
        
        current_districts = set(current_df[district])
        assert len(current_people) == len(current_df)
        assert len(current_districts) == len(current_df)
        current_assignments_from_orig = dict(zip(current_df[district],
                                                 current_df['person']))
        text.append(t)
        text.append('original assignments')
        text.append(current_assignments_from_orig)
        """
        Find people who are in the state in the current period and last period,
        AND are assigned to the same district in both period. 
        Find the districts they go with.
        """
        districts_with_continuing_people = \
                     list(filter(lambda dist: last_assignments_from_orig[dist] \
                                         == current_assignments_from_orig[dist],
                                 current_districts & last_districts))
        people_continuing_in_district = [last_assignments_from_orig[dist]
                                         for dist in districts_with_continuing_people]
                          
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
        assert(len(district_assignments) == len(new_person_assignments))
        assignments = dict(zip(district_assignments, new_person_assignments))
        text.append('simulated assignments')
        text.append(assignments)
        # Update data with new changes
        try:
            state_df.loc[indices, 'person'] = \
                            [assignments.get(dist, 'NaN') 
                             for dist in state_df.loc[indices, district]]
#                            [assignments[dist] for dist in current_df[district]]
        except ValueError:
            print(text)
            print(current_df[['person', district]])
            assert False
        last_assignments_from_orig = current_assignments_from_orig
        last_districts = current_districts

#    print(text)
    return state_df
    
def simulate(df, seed_increment, switch_fraction, time_var):
    np.random.seed(seed_increment)
    district = 'clean district name'
    return df.groupby('state').apply(lambda x: \
                                     reassign(x, time_var, district, switch_fraction))
