import sys
sys.path = sys.path + ['/home/lizs/Documents/tva/algorithm/']
from va_functions import remove_duplicates
import numpy as np
import pandas as pd

# Given a vector of identifiers like [a a b b b],
# Returns a vector of random variables 
# such that if identifiers[x] = identifiers[y],
# effects[x] = effects[y]
def fill_effects(identifiers, st_dev):
    no_dup_ids = remove_duplicates(identifiers)
    id_effect_dict = dict(zip(no_dup_ids
                        , np.random.normal(0, st_dev, len(no_dup_ids))))
    return [id_effect_dict[id_] for id_ in identifiers]

def simulate(params, assignments, seed_increment):
    # unpack parameters
    var_mu, var_theta, var_delta, rho  = params['var mu'], params['var theta'] \
                                       , params['var delta'], params['ar1 param']

    np.random.seed(seed_increment)
    
    var_epsilon = 1 - var_theta - var_mu
    std_epsilon = var_epsilon**.5
    
    assignments['delta'] = fill_effects(assignments['distcode'].values, var_delta**.5)
    assignments['mu']    = fill_effects(assignments['person'].values, var_mu**.5)
    postings = list(zip(assignments['distcode'].values, assignments['person'].values))
    assignments['theta'] = fill_effects(postings, var_theta**.5)
    
    ## Create panel in which districts are always there
    # And use it to create serial correlation
    districts = remove_duplicates(assignments['distcode'].values)
    times = remove_duplicates(assignments['month_id'].values)
    T = len(times)
    D = len(districts)
    
    
    # Introduce serially correlated errors
    all_errors = np.empty((T, D))
    current_error = np.random.normal(0, std_epsilon, D)
    all_errors[0, :] = current_error
    
    for t in range(1, T):
        current_error = rho*current_error + np.random.normal(0, var_epsilon**.5, D)
        all_errors[t, :] = current_error
        
        
    balanced_panel = pd.DataFrame(np.array([
                                             np.tile(districts, T)
                                           , np.array(times).repeat(D)
                                           , all_errors.flatten()
                                          ]).T
                        , columns=['distcode', 'month_id', 'error'])
        

    assignments = pd.merge(assignments, balanced_panel, how='left')
    assignments['outcome'] = assignments['mu'] + assignments['delta'] + assignments['theta'] + assignments['error']
    return assignments
