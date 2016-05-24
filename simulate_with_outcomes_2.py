import numpy as np
import pandas as pd
from copy import copy

def find_swappers(df):
    switch_people = df.loc[df['switch'], 'person'].values
    new_people = copy(switch_people)
    np.random.shuffle(new_people)
    return dict(zip(switch_people, new_people))

def simulate(df, seed_increment, outcomes, switch_fraction):
             
    np.random.seed(seed_increment)
    times = sorted(set(df['year'].values))
    df['switch'] = np.random.random(len(df)) < switch_fraction
    
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
