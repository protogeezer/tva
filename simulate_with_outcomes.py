from va_functions import remove_duplicates
import numpy as np
import pandas as pd
from random import shuffle
from copy import copy

"""
Creates simulated data for testing VA algorithm or doing permutation tests.

Preserved from initial data:
-- Which state bureaucrat a bureaucrat is in.
-- How long each bureaucrat has been in her current district.
"""

# TODO: Deal with consecutiveness. A posting is all years a bureaucrat works in a certian district.

# binary: was the bureaucrat serving in the same district last year?
def create_switch_status(df):
    districts = df['clean district name'].values
    
    df['switch status'] = [int(current != last) for current, last
                   in zip(districts, np.concatenate(([0], districts[:-1])))]
    return df    

# df holds data for one state in one year
def shuffle_bureaucrats(df):
    switch_indices = pd.Series(df['switch status'] == 1)
    switch_people = df.loc[switch_indices, 'person'].values
    shuffle(switch_people)
    df.loc[switch_indices, 'person'] = switch_people
    return df
    
def find_swappers(df):
    switch_indices = pd.Series(df['switch status'] == 1)
    switch_people = df.loc[switch_indices, 'person'].values
    new_switch_people = copy(switch_people)
    shuffle(new_switch_people)
    return dict(zip(switch_people, new_switch_people))

def simulate(df, seed_increment, outcomes):
             
    np.random.seed(seed_increment)
    df = df.groupby('person').apply(create_switch_status)
    times = sorted(set(df['year'].values))
    
    # TODO: Speed up by doing swaps within a state, then looking for unmatched
    # people elsewhere
    for t in times[1:]:
        # find out who needs to swap with who
        # first column of swaps is original, second is replacement
        swaps = df[df['year'] == t].groupby('state').apply(find_swappers).values
        swaps = {k:v for d in swaps for k, v in d.items()}
        
        # Replace bureaucrats in all future times
        indices = [s >= t and p in swaps for s, p in 
                   df[['year', 'person']].values]
        df.loc[indices, 'person'] = [swaps[p] 
                                   for p in df.loc[indices, 'person'].values]
        
    return df
