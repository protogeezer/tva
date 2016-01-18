import sys
sys.path = sys.path + ['/home/lizs/Dropbox/lizs_backup/Documents/tva/algorithm/']
from va_functions import remove_duplicates
import numpy as np
import pandas as pd


def simulate(params, assignments):
    # unpack parameters
    var_mu, var_theta, var_delta = params['var mu'], params['var theta'], params['var delta']
    rho = params['ar1 param']
    np.random.seed()
    
    std_theta = var_theta**.5
    var_epsilon = 1 - var_theta - var_mu
    std_epsilon = var_epsilon**.5
    
    districts = {d:index for index, d in enumerate(remove_duplicates(assignments['distcode'].values))}
    months    = {m:index for index, m in enumerate(remove_duplicates(assignments['month_id'].values))}
    bureaucrats = {b:index for index, b in enumerate(remove_duplicates(assignments['person'].values))}
    
    n_districts = len(districts)
    T = len(months)
    
    assignments.rename(columns={'distcode':'old distcode', 'month_id':'old month id', 'person':'old person'}, inplace=True)
    assignments.loc[:, 'distcode'] = [districts[d] for d in assignments['old distcode']]
    assignments.loc[:, 'month_id'] = [months[m] for m in assignments['old month id']]
    assignments.loc[:, 'person']   = [bureaucrats[b]       for b in assignments['old person']]
    
    balanced_panel = pd.DataFrame(
                        np.array([np.tile(np.arange(n_districts), T), np.arange(T).repeat(n_districts)]).T,
                        columns=['distcode', 'month_id'])
    
    assert len(balanced_panel) == n_districts * T
    # merge in bureaucrats
    balanced_panel = pd.merge(balanced_panel, assignments[['distcode', 'month_id', 'person']], how='left')
    assert len(balanced_panel) == n_districts * T
    
    district_effects = np.random.normal(0, var_delta**.5, n_districts)
    district_effects_vector = np.tile(district_effects, T)
    
    bureaucrat_effects = np.random.normal(0, var_mu**.5, len(bureaucrats))
    bureaucrat_effects_vector = np.array([bureaucrat_effects[b] if not isnan(b) else float('nan')
                                          for b in balanced_panel['person']])
        
    postings_with_duplicates = zip(balanced_panel['distcode'], balanced_panel['month_id'])
    postings = remove_duplicates(zip(balanced_panel['distcode'], balanced_panel['month_id']))
    posting_effects = {hash(p):np.random.normal(0, std_theta) for p in postings}
    posting_effects_vector = np.array([posting_effects[hash(p)] for p in zip(balanced_panel['distcode'], balanced_panel['month_id'])])
    
    all_errors = np.empty((T, n_districts))
    current_error = np.random.normal(0, std_epsilon, n_districts)
    all_errors[0, :] = current_error
    
    for t in range(1, T):
        current_error = rho*current_error + np.random.normal(0, var_epsilon**.5, n_districts)
        all_errors[t, :] = current_error
        
    balanced_panel['true va'] = bureaucrat_effects_vector
    balanced_panel['outcome'] = bureaucrat_effects_vector + district_effects_vector + posting_effects_vector + all_errors.flatten()
    return balanced_panel.dropna()
    
#if __name__ == "__main__":
#    params = {'var mu': .024, 'var theta': .178, 'var delta': .4, 'ar1 param':0}
#    assignments = pd.read_csv('merged.csv', usecols=['month_id', 'distcode', 'person'])
#    simulate(params, assignments).to_csv('simulated_data_real_assignments')
