import pandas as pd
import numpy as np
from basic_va_alg_2 import calculate_va
from basic_va_alg import calculate_va as calculate_va_orig


sigma_mu_squared = 3
data = pd.read_csv('/Users/lizs/Documents/ias/data/indicus_cleaned.csv',
        index_col=0)

n_people = len(set(data['person']))
_, person = np.unique(data['person'], return_inverse = True)
np.random.seed(20616)
person_effects = np.random.normal(0, sigma_mu_squared**.5, n_people)
data['outcome'] = -7 + data['fiscal year'] * 3 + person_effects[person]
data['constant'] = 1

# Try each of three different estimators
print('Original')
est_orig = calculate_va_orig(data.copy(), ['fiscal year', 'constant'], False, 
        column_names={'class id': 'clean district name', 'outcome':'score', 
            'fiscal year':'fiscal year', 'constant':'constant', 'person':'teacher'}, moments_only=True,
        categorical_controls=['class id'])
print('\n\n\n KS')
est_ks = calculate_va(data.copy(), 'outcome', 'person', ['fiscal year'],
                      ['person', 'clean district name'], categorical_controls=['clean district name'],
                      method='ks', add_constant=False)
print('\n\n\n CFR')
est_cfr = calculate_va(data.copy(), 'outcome', 'person', ['fiscal year'],
                      ['person', 'clean district name'], categorical_controls=['clean district name'],
                      method='cfr', add_constant=False)
print(est_orig)
print(est_ks)
print(est_cfr)
#est_fk = calculate_va(data.copy(), 'outcome', 'person', ['fiscal year'],
#                      ['person', 'clean district name'], categorical_controls=['clean district name'],
#                      method='fk')
