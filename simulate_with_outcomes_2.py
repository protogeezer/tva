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
    
#@profile
def reassign(state_df, time_var, district, switch_fraction, seed):
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
    last_assignments_from_sim = dict(zip(last_df[district], last_df['person']))
    last_districts = set(last_df[district])
    last_district_from_sim = dict(zip(last_df['person'], last_df[district]))
    
    text = Text(state_df['state'].values[0])
    text.append(last_assignments_from_orig)
    
    for t in times[1:]:
        indices = pd.Series(state_df[time_var] == t)        
        current_df = state_df[indices]
            
        current_people = set(current_df['person'])
        if len(current_people) != len(current_df): # someone is duplicated; drop duplicates
            _, ind = np.unique(current_df['person'].values, return_index = True)
            current_df = current_df.iloc[ind]
        
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
        
        continuation_people_districts = [last_district_from_sim[p] 
                                         for p in people_continuing_in_district
                                 if last_district_from_sim[p] in current_districts]
        text.append('people continuing in district')
        text.append(people_continuing_in_district)
                          
        other_people = current_people - set(people_continuing_in_district)
        assert(other_people | set(people_continuing_in_district) == current_people)
        text.append('people not continuing in district:')
        text.append(other_people)
        other_districts = current_districts - set(continuation_people_districts)
        assert(other_districts | set(continuation_people_districts) == current_districts)   
        text.append('districts with continuing people')
        text.append(districts_with_continuing_people)
        text.append('other districts')
        text.append(other_districts)
        """ 
        Create new assignments:
            - People who continue in the same district do so
            - Everyone else is randomly assigned to one of the other districts
        """
        other_people = list(other_people)
        
        np.random.shuffle(other_people)
        person_assignments = people_continuing_in_district + list(other_people)
        district_assignments = continuation_people_districts + list(other_districts) 

        assert(set(person_assignments) == current_people)
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
        state_df.loc[indices, 'person' + str(seed)] = \
                            [assignments.get(dist, 'NaN') 
                             for dist in state_df.loc[indices, district]]
        last_assignments_from_sim = assignments
        last_assignments_from_orig = current_assignments_from_orig
        last_district_from_sim = dict(zip(new_person_assignments, district_assignments))
        last_districts = current_districts

    #print(text)
    return state_df
    
def simulate(df, seed_increment, switch_fraction, time_var):
    np.random.seed(seed_increment)
    print(seed_increment)
    district = 'clean district name'
    result = df.groupby('state').apply(lambda x: reassign(x, time_var, 'clean district name', switch_fraction, seed_increment)).drop('person', axis=1)
    print(result.head())
    return result
