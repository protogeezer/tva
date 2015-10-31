import time
import pandas as pd
import numpy as np
import copy
import va_functions
from guppy import hpy
from two_type_covariance import calculate_covariances

h = hpy()

# Basic algorithm: Effect is constant within teacher
data_directory = '/n/regal/chetty_lab/esantorella/'
filename = 'students_with_controls'
output_directory = '/n/home09/esantorella/algorithm/'
n = 10**5
output_file = open(output_directory+'two_types_'+str(n), 'w')

controls = ['sped', 'limited_english', 'student_grade', 'ETHNethnici_1', 'ETHNethnici_2', 'ETHNethnici_3', 'ETHNethnici_4', 'PreZl_score', 'PreOl_other', 'l1_days_absent', 'l1_stu_suspension', 'female']

start = time.time()
data = pd.read_csv(data_directory+filename+'.csv', sep=',', usecols=['score', 'stuid', 'teacher', 'class', 'year']+controls, nrows=n)
output_file.write('Time to read in file: ' + str(time.time() - start))

start = time.time()
data.loc[:, 'residual'], _ = va_functions.residualize(data, 'score', controls, 'teacher')
output_file.write('\nTime to residualize: ' + str(time.time() - start))

output_file.write('\nNumber of observations with non-null residual : ' + str(sum(data['residual'].notnull())))
data.loc[:, 'lag score binary'] = [int(score - np.mean(data['PreZl_score'].values) > 0) for score in data['PreZl_score'].values]
data.loc[:, 'lag absence binary'] = [int(score - np.mean(data['l1_days_absent'].notnull().values) > 0) for score in data['l1_days_absent'].values]

# Run the algorithm
for type_var in ['female', 'lag score binary', 'limited_english', 'lag absence binary']:
    column_names = {'student id':'stuid', 'type':type_var, 'class id':'class'}
    output_file.write('\n\n\nSplitting on variable: ' + type_var) 
    start = time.time()
    params, n_obs, _, _ = calculate_covariances(copy.copy(data), [], residual='residual', column_names=column_names)
    output_file.write('\nTime: ' + str(time.time() - start))

    for k in params:
        output_file.write('\n' + k + ' ' + str(params[k]))

output_file.close()

print(h.heap())
