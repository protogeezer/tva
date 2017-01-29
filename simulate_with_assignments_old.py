import sys
sys.path = sys.path + ['/home/lizs/Documents/tva/algorithm/']
from va_functions import remove_duplicates
import numpy as np
import pandas as pd

def convert_vector_to_index_dict(vector):
    return {value:index for index, value in enumerate(remove_duplicates(vector))}

# Given a vector of identifiers like [a a b b b],
# Returns a vector of random variables 
# such that if identifiers[x] = identifiers[y],
# effects[x] = effects[y]
def fill_effects(identifiers, st_dev):
    no_dup_ids = remove_duplicates(identifiers)
    id_effect_dict = dict(zip(no_dup_ids
                        , np.random.normal(0, st_dev, no_dup_ids)))
    return [id_effect_dict[id_] for id_ in identifiers]

def simulate(params, assignments, seed_increment):
    # unpack parameters
    var_mu, var_theta, var_delta, rho  = params['var mu'], params['var theta'] \
                                       , params['var delta'], params['ar1 param']

    np.random.seed(seed_increment)
    
    std_theta = var_theta**.5
    var_epsilon = 1 - var_theta - var_mu
    std_epsilon = var_epsilon**.5
    
    district_effects = fill_effects(assignments['distcode'].values, var_delta**.5)
    bureaucrat_effects = fill_effects(assignments['person'].values, var_mu**.5)
    postings = zip(assignments['distcode'].values, assignments['person'].values)
    posting_effects = fill_effects(postings)
    
#    districts = convert_vector_to_index_dict(assignments['distcode'].values)
#    months = convert_vector_to_index_dict(assignments['month_id'].values)
#    bureaucrats = convert_vector_to_index_dict(assignments['person'].values)
    

    
#    
#    n_districts = len(districts)
#    T = len(months)
#    
#    assignments.rename(columns={'distcode':'old distcode'
#                              , 'month_id':'old month id'
#                              , 'person':'old person'}, inplace=True)
#    assignments.loc[:, 'distcode'] = \
#                        [districts[d] for d in assignments['old distcode']]
#    assignments.loc[:, 'month_id'] = \
#                        [months[m] for m in assignments['old month id']]
#    assignments.loc[:, 'person']   = \
#                        [bureaucrats[b] for b in assignments['old person']]
#    
#    balanced_panel = pd.DataFrame(np.array([np.tile(np.arange(n_districts), T)
#                                          , np.arange(T).repeat(n_districts)]).T
#                                , columns=['distcode', 'month_id'])
#    
#    assert len(balanced_panel) == n_districts * T
#    # merge in bureaucrats
#    balanced_panel = pd.merge(balanced_panel
#                            , assignments[['distcode', 'month_id', 'person']]
#                            , how='left')
#    assert len(balanced_panel) == n_districts * T
#    
#    district_effects = np.random.normal(0, var_delta**.5, n_districts)
#    district_effects_vector = np.tile(district_effects, T)
#    
#    bureaucrat_effects = np.random.normal(0, var_mu**.5, len(bureaucrats))
#    bureaucrat_effects_vector = np.array([bureaucrat_effects[b] 
#                                                    if not np.isnan(b) 
#                                                    else b 
#                                             for b in balanced_panel['person']])
        
#    posting_vector = zip(balanced_panel['distcode'], balanced_panel['month_id'])
#    postings = remove_duplicates(posting_vector)
#                                                   
#    posting_effects = {hash(p):np.random.normal(0, std_theta) for p in postings}
#    print(posting_effects)
#    posting_effects_vector = np.array([posting_effects[hash(p)] 
#                                                for p in posting_vector])
    postings = zip(balanced_panel['distcode'], balanced_panel['month_id'])
    posting_effects_vector = fill_effects(postings, 
    
    # Introduce serially correlated errors
    all_errors = np.empty((T, n_districts))
    current_error = np.random.normal(0, std_epsilon, n_districts)
    all_errors[0, :] = current_error
    
    for t in range(1, T):
        current_error = rho*current_error \
                        + np.random.normal(0, var_epsilon**.5, n_districts)
        all_errors[t, :] = current_error
        
    balanced_panel['true va'] = bureaucrat_effects_vector
    balanced_panel['outcome'] = bureaucrat_effects_vector \
                                + district_effects_vector \
                                + posting_effects_vector + all_errors.flatten()
    print(balanced_panel)
    print(balanced_panel.dropna())
    return balanced_panel.dropna()
