import numpy as np
from va_functions import Text, remove_duplicates
from config import *
import sys
sys.path.append(hdfe_dir)
sys.path.append('/Users/lizs/Documents/ias')
from find_connected_sets import find_biggest_connected_set


def reassign(state_df):
    times = sorted(set(state_df['month_id']))
    last_df = state_df[state_df['month_id'] == np.min(times)]
    assert times[0] == np.min(times)
    last_assignments_from_orig = dict(zip(last_df['distcode'], last_df['person']))
    last_assignments_from_sim = dict(zip(last_df['distcode'], last_df['person']))
    last_districts = remove_duplicates(last_df['distcode'])
    
    text = Text(state_df['state'].values[0], last_assignments_from_orig)
    state_df['sim_person'] = np.nan
    initial_idx = state_df['month_id'] == times[0]
    state_df.loc[initial_idx, 'sim_person'] = state_df.loc[initial_idx, 'person']
    
    for t in times[1:]:
        indices = pd.Series(state_df['month_id'] == t)
        current_df = state_df[indices]
        current_districts = remove_duplicates(current_df['distcode'])
        current_people = remove_duplicates(current_df['person'])
        assert set(current_districts) == set(current_df['distcode'])
        assert len(current_districts) == len(current_df)
        assert len(current_people) == len(current_df)
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
        other_districts = [dist for dist in current_districts 
                           if dist not in districts_with_continuing_people]
        continuing_people = [last_assignments_from_sim[d] for d in districts_with_continuing_people]
        other_people = [p for p in current_people if p not in continuing_people]
        np.random.shuffle(other_people)
        last_assignments_from_sim = dict(zip(districts_with_continuing_people + other_districts,
                                             continuing_people + other_people))

        text.append('simulated assignments', last_assignments_from_sim)
        # Update data with new changes
        state_df.loc[indices, 'sim_person'] = state_df['distcode'].map(last_assignments_from_sim)
        assert np.all(np.isfinite(state_df.loc[indices, 'sim_person']))
        last_assignments_from_orig = current_assignments_from_orig.copy()
        last_districts = current_districts.copy()

    state_df['sim_person'] = state_df['sim_person'].astype(int)
    return state_df


def simulate(df, seed_increment):
    np.random.seed(seed_increment)
    print(seed_increment)
    no_duplicates = df.set_index(['state', 'month_id', 'person']).index.is_unique
    if not no_duplicates:
        raise ValueError('Some months, the same person was in a state twice')
    grouped = df.groupby('state')
    df = grouped.apply(reassign)
    # States connected in original
    target_connected_states = set(df['state'])
    # Find connected set and return indicator for it
    conn = find_biggest_connected_set(df[['sim_person', 'distcode']], person='sim_person')
    print('Size of largest connected set', np.sum(conn))
    connected_states = set(df.loc[conn, 'state'])
    number_missing = len(target_connected_states) - len(connected_states)
    print('Number missing', number_missing)
    missing_from_conn = target_connected_states - connected_states
    print('States missing:', missing_from_conn)

    # Try to fill these in
    for state in missing_from_conn:
        print(state)
        t = np.random.choice(df.loc[df['state'] == state, 'month_id'])
        t_idx = df['month_id'] == t
        one = np.random.choice(df.loc[t_idx & (df['state'] == state), 'sim_person'])
        two = np.random.choice(df.loc[t_idx & conn, 'sim_person'])
        one_loc = (df['month_id'] > t ) & (df['sim_person'] == one)
        two_loc = (df['month_id'] > t) & (df['sim_person'] == two)
        df.loc[one_loc, 'sim_person'] = two
        df.loc[two_loc, 'sim_person'] = one

    conn = find_biggest_connected_set(df[['person', 'distcode']])
    connected_states = set(df.loc[conn, 'state'])
    missing_from_conn = set(target_connected_states) & set(connected_states) \
                                                       - set(connected_states)
    print('States missing:', missing_from_conn)
    return df['sim_person'].astype(int)
