import pandas as pd
import numpy as np
from basic_va_alg import calculate_va


sigma_mu_squared = 3
data = pd.read_csv('/Users/lizs/Documents/ias/data/indicus_cleaned.csv',
        index_col=0)

n_people = len(set(data['person']))
_, person = np.unique(data['person'], return_inverse = True)
np.random.seed(20616)
person_effects = np.random.normal(0, sigma_mu_squared**.5, n_people)
data['outcome'] = -7 + data['fiscal year'] * 3 + person_effects[person]
data['constant'] = 1
data['fy2'] = data['fiscal year'] ** 2

data['id'] = data['person'].astype(str) + '_' + data['clean district name'].astype(str)

# Try each of three different estimators
print('KS')
est_ks = calculate_va(data.copy(), 'outcome', 'person', ['fiscal year', 'fy2'],
                      ['id'], categorical_controls=None,
                      method='mle')
sigma_mu_squared, sigma_theta_squared, sigma_epsilon_squared, beta, lambda_ = est_ks
x = data[['fiscal year', 'fy2']]
x -= np.mean(x, 0)
variance = np.var(x.dot(lambda_)) + sigma_mu_squared
print(variance)


#print('CFR')
#est_cfr = calculate_va(data.copy(), 'outcome', 'person', ['fiscal year'],
#                      ['person', 'clean district name'], categorical_controls=['clean district name'],
#                      method='cfr')[0]
#print('FK')
#est_fk = calculate_va(data.copy(), 'outcome', 'person', ['fiscal year'],
#                      ['person', 'clean district name'], categorical_controls=['clean district name'],
#                      method='fk')[0]
#
#print(est_ks)
#print(est_cfr)
#print(est_fk)
