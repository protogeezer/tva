import pandas as pd
import numpy as np
import va_functions
from two_type_covariance import calculate_covariances
import time
import copy

# Basic algorithm: Effect is constant within teacher
directory =  '/home/lizs/Dropbox/lizs_backup/Documents/tva/algorithm/'
filename = 'two_type_simulated_data'
output_file = open('two_types', 'w')

controls = ['x1', 'x2']

data = pd.read_csv(directory+filename+'.csv', sep=',', nrows=100000)
start = time.time()
data.loc[:, 'residual'], _ = va_functions.residualize(data, 'score', controls, 'teacher')
output_file.write('Time to residualize: ' + str(time.time() - start))
output_file.write('\nNumber of observations with non-null residual : ' + str(sum(data['residual'].notnull())))

data.loc[:, 'year'] = data['class id']

# Run the algorithm
params, n_obs, _, _ = calculate_covariances(copy.copy(data), [], residual='residual')
output_file.write('\nTime: ' + str(time.time() - start))

for k in params:
    output_file.write('\n' + k + ' ' + str(params[k]))
