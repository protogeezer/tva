from f_va_alg import calculate_va
import pandas as pd
import numpy as np

sigma_mu_squared = 3

data = pd.read_csv('/Users/lizs/Documents/ias/data/indicus_cleaned.csv', \
              usecols=['fiscal year', 'person'])

n_people = len(set(data['person']))
_, data['person']  = np.unique(data['person'], return_inverse = True)
person_effects = np.random.normal(0, sigma_mu_squared**.5, n_people)
data['outcome'] = data['fiscal year'] * 3 + person_effects[data['person']]

est = calculate_va(data, ['fiscal year'], False, outcome='outcome', teacher='person')
print(est)
