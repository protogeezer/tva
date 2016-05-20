from va_functions import remove_duplicates
import numpy as np
import pandas as pd
from random import shuffle

"""
TODO: Would be better to create balanced panel outside this function. Would be even better to have a check_balanced function that only creates a balanced panel if it is missing.
"""

# df should have columns person and district
# nan = not in sample, 1 = switching to new district, 0 = continuing
def create_switch_status(df):
    districts = df[district].values
    
    df['switch status'] = [current if type(current) == float 
                                   and np.isnan(current) 
                                  else int(current != last)
                           for current, last
                   in zip(districts, np.concatenate(([0], districts[:-1])))]
    return df    

# df holds data for one state in one year
def shuffle_bureaucrats(df):
    switch_indices = pd.Series(df['switch status'] == 1)
    #print(switch_indices)
    switch_people = df.loc[switch_indices, person].values
    shuffle(switch_people)
    #print(switch_people)
    df.loc[switch_indices, person] = switch_people
    return df

def simulate(df, seed_increment, outcomes, district='district',
             person='person', time='year', state='state'):
    np.random.seed(seed_increment)
    
    district = 'clean district name'
    person = 'person'
    time = 'year'
    state = 'state'
    df = pd.read_csv('/home/lizs/Documents/ias/data/indicus_cleaned.csv')
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df[person] = [int(x) for x in df[person]]
    
    # Create a balanced panel by first making complete panel, then merging
    """
    times = sorted(set(df[time].values))
    T = len(times)
    collectors = remove_duplicates(df[person].values)
    N = len(collectors)
    
    balanced_panel = pd.DataFrame(np.array([np.tile(collectors, T)
                                          , np.array(times).repeat(N)]).T
                                , columns=[person, time])
    df = df.merge(balanced_panel, how = 'right')
    """
    """
     Give each collector a "status" variable tells whether they switch,
     enter, or exit
     e.g. [NA, A, A, B, B, C, NA]
      ->  [NA, 'enter', 'continue', 'switch', 'continue', 'switch', 'exit']
    """
    df = df.groupby(person).apply(create_switch_status)
    
    """
    Also track a state variable for each bureaucrat

    Randomly order bureacurats so that "status" and "state" stay the same,
    but district *within* a state is random.
    """
    print(df[df[district] == 'Adilabad'])
    df = df.groupby((state, time)).apply(shuffle_bureaucrats)
    print(df[df[district] == 'Adilabad'])
    
    return df
