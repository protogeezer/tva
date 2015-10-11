import pandas as pd
import numpy as np
import va_functions
#from two_type_covariance import calculate_covariances
from va_alg_two_groups import calculate_va

# Basic algorithm: Effect is constant within teacher
directory = '~/Documents/tva/algorithm/'
filename = 'two_type_simulated_data'
output_file = open('two_types', 'w')

data = pd.read_csv(directory+filename+'.csv', sep=',', nrows = 100000)
output_file.write('Number of observations: ' + str(len(data.index)))
data.loc[:, 'year'] = data['class id']

# Run the algorithm
class_type_df, var_mu_hat, corr_mu_hat, var_theta_hat, var_epsilon_hat = calculate_va(data, ['x1', 'x2'], jackknife = False)

corr_mu = .7
corr_theta = .9
params = {'cov mu':[[.018, corr_mu*(.018*.012)**(.5)], [corr_mu*(.018*.012)**(.5), .012]], 'cov theta':[[.03, corr_theta*(.03*.02)**.5], [corr_theta*(.03*.02)**.5, .02]], 'mean class size':24, 'mean classes taught':3}

output_file.write('\nCorr mu actual value; measured: ' + '.7' + '; ' + str(corr_mu_hat))
output_file.write('\nVar mu actual value; measured: ' + str(params['cov mu'][0][0]) +', ' + str(params['cov mu'][1][1]) + '; ' + str(var_mu_hat))
output_file.write('\nVar theta actual value; measured: ' + str(params['cov theta'][0][0]) +', ' + str(params['cov theta'][1][1]) + '; ' + str(var_theta_hat))
output_file.write('\nVar epsilon actual value; measured: ' + '.2455; ' + str(var_epsilon_hat))
