from simulate_baseline import simulate
from basic_va_alg import calculate_va
import numpy as np
import pandas as pd
from hdfe import Groupby

params = {'num teachers':1000, 'beta':[2, 3], 'sd mu':.024**.5, 'sd theta':.178**.5, 
          'sd epsilon':(1-.024-.178)**.5, 'mean class size':20, 'mean classes taught':3}

data = simulate(params, 0)
data['id'] = data['class id'].astype(str) + '_' + data['teacher'].astype(str)
data['constant'] = 1

# Do things properly here
beta = np.linalg.lstsq(data[['x1', 'x2']], data['score'])
print(beta[0])

est_cfr = calculate_va(data.copy(), 'score', 'teacher', ['x1', 'x2'], 
                       ['class id', 'teacher'], categorical_controls=None, method='cfr')

est_ks = calculate_va(data.copy(), 'score', 'teacher', ['x1', 'x2'], 
                       ['class id', 'teacher'], categorical_controls=None, method='ks')

est_mle = calculate_va(data.copy(), 'score', 'teacher', ['x1', 'x2'], 
                       ['id'], categorical_controls=None, method='mle')

