import sys
import pandas as pd
import numpy as np
from basic_va_alg import calculate_va
from multiprocessing import cpu_count, Pool
from simulate_with_assignments import simulate
import matplotlib.pyplot as plt

def compute_estimates_once(input_tuple):
    i, assignments = input_tuple
    print(i)
    params = {'var mu': 0, 'var theta': .178, 'var delta': .4, 'ar1 param':0}
    column_names = {'person':'teacher', 'distcode':'class id', 
                    'month_id':'student id', 'outcome':'score'}
    data = simulate(params, assignments, i)
    return calculate_va(data, ['month_id'], False, 
                        categorical_controls=['distcode'], moments_only=False, 
                        class_level_vars = ['person', 'distcode'],
                        column_names=column_names)

if __name__ == '__main__':
    #iters = list(range(4, 10))
    iters = [4]

    assignments = pd.read_csv('/Users/lizs/Documents/ias/data/indicus_cleaned.csv', \
                              usecols=['fiscal year', 'clean district name', 'person'])
    assignments.rename(columns = {'clean district name': 'distcode', 
                                  'fiscal year':'month_id'}, 
                       inplace = True)

    parallel = False
    tuples = ((i, assignments) for i in iters)
    if parallel:
        num_cores = min(cpu_count(), len(iters))
        pool = Pool(num_cores)
        results = pool.map(compute_estimates_once, tuples)
        pool.close()
        pool.join()
    else:
        results = [compute_estimates_once(t) for t in tuples]
    
    var_mu_hat = np.array([elt[1] for elt in results])
    
    print(var_mu_hat)
    print(np.mean(var_mu_hat))
    print(np.var(var_mu_hat))
    np.save('parameter_estimates/simulate_with_assignments_0.npy', var_mu_hat)
    
"""
    results = [[elt[0], elt[1], elt[2], elt[3][0], elt[3][1]] for elt in results]
    results = np.array(results)
    np.save('parameter_estimates/baseline_'+sys.argv[1]+'.npy', results)
"""
############

#results = np.load('parameter_estimates/baseline_0.npy')
"""
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

parameter_hist(results[:, 0], 'variance of mu', params['var mu'])
parameter_hist(results[:, 1], 'variance of theta', params['var theta'])
parameter_hist(results[:, 2], 'variance of epsilon', 1-params['var mu'] - params['var theta'])

# Test average squared error, normalized by standard deviation
error = (results[:, 0] - .0135)**2 / results[:, 3]**2
print(np.mean(error))
"""
