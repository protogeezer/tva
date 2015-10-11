import pandas as pd
import basic_va_alg
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from save_figure import save
import va_functions

# Basic algorithm: Effect is constant within teacher
directory = '~/Documents/tva/algorithm/'
filename = 'baseline_simulated_data'
output_file = open('output_baseline', 'w')

data = pd.read_csv(directory+filename+'.csv', sep=',', nrows = 500000)
output_file.write('Number of observations: ' + str(len(data.index)))

data.loc[:, 'year'] = data['class id']

# Run the algorithm
[data, var_mu_hat, var_theta_hat, var_epsilon_hat] \
    = basic_va_alg.calculate_va(data, ['x1', 'x2'], True, parallel=True)

#data.to_csv('tva_basic_alg_'+filename)

## Summary stats that help check if it is working
#output_file.write('\nVariance of epsilon, true: ' + str(.2455))
#output_file.write('\nVariance of epsilon, computed:'+str(var_epsilon_hat))

#output_file.write('\n\nVariance of value-added, true: ' + str(.0135))
#output_file.write('\nVariance of value-added, computed:'+str(var_mu_hat))

#output_file.write('\n\nVariance of theta, true: ' + str(.0295))
#output_file.write('\nVariance of theta, computed: '+str(var_theta_hat))

#output_file.write('\n\nempirical mean of value-added '+str(np.ma.average(data['va'].values)))
#output_file.write('\nempirical variance of value-added '+str(np.var(data['va'].values)))

## Make plots to show that va is correct on average
#print('about to binscatter')
#va_bins, predictions = va_functions.binscatter(data['va'], data['true va'], 20)
#plt.plot(va_bins, predictions)

#plt.plot(np.arange(-.3, .3, .1), np.arange(-.3, .3, .1))

#plt.show()

## Test confidence interval
#data.loc[:, 'outside ci'] = [true_va > va + 1.96 * se or true_va < va - 1.96 * se 
#                                       for true_va, va, se in zip(data['true va'].values, \
#                                           data['va'].values, np.sqrt(data['variance'].values))]

### Regressions
#result = sm.OLS(data['mean score'], data['va'], hasconst = True, missing = 'drop')
#result = result.fit()
#output_file.write('\n'+str(result.summary()))   

## Binned scatter plot
#class_level_data['missing'] = pd.isnull(class_level_data['va']+class_level_data['va other'])
#data_nomissing = class_level_data[~class_level_data['missing']]
#plt.hist2d(data_nomissing['va'].as_matrix(), data_nomissing['va other'].as_matrix(), bins=100)
#save('figures/2d_scatter_tva_basic_'+ filename)

#bins, y = binscatter(data_nomissing['va'].as_matrix(), data_nomissing['va other'].as_matrix(), 100)
#plt.plot(bins, y, 'o')
#save('figures/1d_scatter_basic_'+filename)
