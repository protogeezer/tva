from simulate_baseline import simulate
from basic_va_alg import calculate_va
import numpy as np
import pandas as pd
from hdfe import Groupby
import time

params = {'num teachers':1000, 'beta':[2, 3], 'sd mu':.024**.5, 'sd theta':.178**.5, 
          'sd epsilon':(1-.024-.178)**.5, 'mean class size':20, 'mean classes taught':3}


data = simulate(params, 2)

# idx = data['teacher'] == 0
# data.loc[idx, 'teacher'] = 1000
data['id'] = data['teacher'] * np.max(data['class id']) + data['class id']
assert np.all(np.diff(data['id']) >= 0)

# # Do things properly here
# est_cfr = calculate_va(data.copy(), 'score', 'teacher', ['x1', 'x2'], 
#                        ['class id', 'teacher'], categorical_controls=None, method='cfr')
# print(est_cfr[0])
# 
# est_ks = calculate_va(data.copy(), 'score', 'teacher', ['x1', 'x2'], 
#                        ['class id', 'teacher'], categorical_controls=None, method='ks')
# print(est_ks[0])

start = time.time()
est_mle = calculate_va(data.copy(), 'score', 'teacher', ['x1', 'x2'], 
                       ['id'], categorical_controls=None, method='mle')
print(time.time() - start)
print('sigma_mu_squared', est_mle[0])
lambda_ = est_mle[4] - est_mle[3]
print(lambda_)
mean = data[['x1', 'x2']].dot(lambda_)
print('other variance', np.var(mean))

