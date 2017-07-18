import numpy as np
import pandas as pd
from va_functions import Text, normalize, remove_duplicates
from scipy.stats import burr
from config import *
import sys
sys.path.append(hdfe_dir)
from hdfe import Groupby
       
def reassign(state_df, time_var, switch_frequency):
    times = sorted(set(state_df[time_var]))
    last_df = state_df[state_df[time_var] == np.min(times)]
    last_assignments_from_orig = dict(zip(last_df['distcode'], last_df['person']))
    last_assignments_from_sim = dict(zip(last_df['distcode'], last_df['person']))
    last_districts = remove_duplicates(last_df['distcode'])
    last_district_from_sim = dict(zip(last_df['person'], last_df['distcode']))
    
    text = Text(state_df['state'].values[0], last_assignments_from_orig)
    
    for t in times[1:]:
        indices = pd.Series(state_df[time_var] == t)        
        current_df = state_df[indices]
        current_districts = remove_duplicates(current_df['distcode'])
        assert set(current_districts) == set(current_df['distcode'])
        assert len(current_districts) == len(current_df)
        current_assignments_from_orig = dict(zip(current_df['distcode'],
                                                 current_df['person']))
        """
        Find people who are in the state in the current period and last period,
        AND are assigned to the same district in both period. 
        Find the districts they go with.
        """
        districts_with_continuing_people = \
                [dist for dist in current_districts 
                        if dist in last_districts and
                       last_assignments_from_orig[dist] == current_assignments_from_orig[dist]]
        assert len(districts_with_continuing_people) == len(set(districts_with_continuing_people))
        assert set(districts_with_continuing_people).issubset(set(current_df['distcode']))
        people_continuing_in_district = [last_assignments_from_orig[dist]
                                         for dist in districts_with_continuing_people]
        assert len(people_continuing_in_district) == len(set(people_continuing_in_district))
        assert set(people_continuing_in_district).issubset(set(current_df['person']))
       
        continuation_people_districts = [last_district_from_sim[p] 
                                         for p in people_continuing_in_district
                                         if last_district_from_sim[p] in current_districts]
        assert len(continuation_people_districts) == len(set(continuation_people_districts))
        assert set(continuation_people_districts).issubset(set(current_df['distcode']))
       
        other_people = [p for p in remove_duplicates(current_df['person'])
                        if p not in people_continuing_in_district]
        assert set(other_people) | set(people_continuing_in_district) == set(current_df['person'])
        if t % switch_frequency == 0:
            """ 
            Create new assignments:
                - People who continue in the same district do so
                - Everyone else is randomly assigned to one of the other districts
            """
            np.random.shuffle(other_people)

        person_assignments = people_continuing_in_district + other_people
        assert set(person_assignments) == set(current_df['person'])
        other_districts = [d for d in current_districts
                           if d not in continuation_people_districts]
        assert len(other_districts) == len(set(other_districts))
        assert set(other_districts).issubset(set(current_districts))
        district_assignments = continuation_people_districts + other_districts
        assert set(district_assignments) == set(current_districts)
        assert len(district_assignments) == len(person_assignments)

        assignments = dict(zip(district_assignments, person_assignments))
        text.append('simulated assignments', assignments)
        # Update data with new changes
        state_df.loc[indices, 'person'] = state_df['distcode'].map(assignments)
        last_assignments_from_orig = current_assignments_from_orig.copy()
        last_district_from_sim = dict(zip(person_assignments, district_assignments))
        last_districts = current_districts.copy()

    return state_df
    
def simulate(df, seed_increment, switch_frequency, time_var):
    np.random.seed(seed_increment)
    print(seed_increment)
    no_duplicates = df.set_index(['state', 'month_id', 'person']).index.is_unique
    if not no_duplicates:
        raise ValueError('Some months, the same person was in a state twice')
    grouped = df.groupby('state')
    df = grouped.apply(lambda x: reassign(x, time_var, switch_frequency))
    people = set(df['person'])
    person_effect = dict(zip(people, burr.rvs(6, 10**5, size = len(people))))
    df['person_effect'] = df['person'].map(person_effect)
    df['person_effect'] = normalize(df['person_effect'])
    return df[['person', 'person_effect']]
