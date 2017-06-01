from simulate_baseline import simulate
from basic_va_alg import calculate_va
import numpy as np
import pandas as pd
from hdfe import Groupby
import time

params = {'num teachers':100, 'beta':[2, 3], 'sd mu':.024**.5, 'sd theta':.178**.5, 
          'sd epsilon':(1-.024-.178)**.5, 'mean class size':20, 'mean classes taught':3}


data = simulate(params, 2)

# idx = data['teacher'] == 0
# data.loc[idx, 'teacher'] = 1000
data['id'] = data['teacher'] * np.max(data['class id']) + data['class id']
assert np.all(np.diff(data['id']) >= 0)

print('Real total varaince ', np.var(data['true va']))

est_cfr = calculate_va(data.copy(), 'score', 'teacher', ['x1', 'x2'], 
                       ['class id', 'teacher'], categorical_controls=None, method='cfr')
print('CFR', est_cfr[0])

est_ks = calculate_va(data.copy(), 'score', 'teacher', ['x1', 'x2'], 
                       ['class id', 'teacher'], categorical_controls=None, method='ks')
print('KS', est_ks[0])

start = time.time()
est_mle = calculate_va(data.copy(), 'score', 'teacher', ['x1', 'x2'], 
                       ['id'], categorical_controls=None, method='mle')
print(time.time() - start)
print('total variance', est_mle['total var'])
print('total variance', est_mle['sigma mu squared'] + est_mle['predictable var'])
# Should be negative definite! uh-oh
asymp_var = np.linalg.inv(est_mle['hessian']) / np.sqrt(len(set(data['teacher'])))
asymp_var_lambda = asymp_var[5:7, 5:7]
print(asymp_var_lambda)



