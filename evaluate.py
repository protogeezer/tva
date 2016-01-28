import sys
import pandas as pd
import numpy as np
from basic_va_alg import calculate_va
from multiprocessing import cpu_count, Pool
from simulate_baseline import simulate
import matplotlib.pyplot as plt
from va_functions import binscatter

def compute_estimates_once(i):
    print(i)
    params = {'sd mu':.0135**.5, 'sd theta':.0295**.5, 'mean class size':20, 'mean classes taught':3, 'num teachers': 1000, 'sd epsilon':.2455**.5, 'beta':[2, 3]}
    data = simulate(params)
    data.loc[:, 'year'] = data['class id']
    #return calculate_va(data, ['x1', 'x2'], False, moments_only=True)
    return calculate_va(data, ['x1', 'x2'], False, categorical_controls=['year'], moments_only=True)

n_iters = 16

num_cores = min(cpu_count(), n_iters)
pool = Pool(num_cores)
results = pool.map(compute_estimates_once, range(n_iters))
pool.close()
pool.join()

results = np.array(results)

np.save('parameter_estimates/baseline_'+sys.argv[1]+'.npy', results)

############

#results = np.load('parameter_estimates/baseline_0.npy')

def bins(y, bin_size):
    return np.arange(min(y), max(y)+bin_size, bin_size)


def parameter_hist(y, param_name, true_value):
    bin_size = .001
    fig, ax = plt.subplots(1)
    ax.hist(y, bins=bins(y, bin_size))
    ax.set_title('Estimates of '+param_name)
    
    ax.axvline(true_value, color='k', linewidth='3') # true value
    ax.axvline(np.mean(y), color='r', linewidth='1') # mean
    # 95% CI of mean
    se1 = (np.var(y)/len(y))**(.5)
    ax.axvline(np.mean(y) - 1.96*se1, color='r', linewidth='1')
    ax.axvline(np.mean(y) + 1.96*se1, color='r', linewidth='1')
    print('red lines:')
    print(np.mean(y))
    print(np.mean(y) - 1.96*se1)
    print(np.mean(y) + 1.96*se1)
    # 5th perentile
    ax.axvline(sorted(y)[int(.05*len(y))], color='g', linewidth='1')
    # 95th perentile
    ax.axvline(sorted(y)[int(.95*len(y))], color='g', linewidth='1')
    print('green lines:')
    print(sorted(y)[int(.05*len(y))])
    print(sorted(y)[int(.95*len(y))])
        
    fig.savefig('parameter_estimates/'+param_name+'_sim_estimates_ac')

params = {'sd mu':.0135**.5, 'sd theta':.0295**.5, 'mean class size':20, 'mean classes taught':3, 'num teachers': 1000, 'sd epsilon':.2455**.5, 'beta':[2, 3]}

parameter_hist(results[:, 0], 'variance of mu', .0135)
parameter_hist(results[:, 1], 'variance of theta', .0295)
parameter_hist(results[:, 2], 'variance of epsilon', .2455)



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
