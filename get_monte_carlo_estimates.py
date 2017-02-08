from simulate_baseline import simulate
from basic_va_alg import calculate_va
import numpy as np
import datetime
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count

"""
This file runs Monte Carlo simulations, evaluates several estimators
on simulated data sets, and makes CDFs of the estimates.
It then repeats this for different data sizes.
"""
# TODO: test more parameters than sigma_mu_squared

def simulate_and_evaluate(input_tuple):
    n_teachers, i = input_tuple
    params = {'num teachers': n_teachers, 'beta':[2, 3], 'sd mu':.01**.5,
          'sd theta':.2**.5, 'sd epsilon':(1-.01-.2)**.5,
          'mean class size':20, 'mean classes taught':4}
    data = simulate(params, i)
    est_ks = calculate_va(data.copy(), 'score', 'teacher', ['x1', 'x2'], 
                          ['teacher', 'class id'], ['c1'], method='ks', 
                          add_constant=False)[0]
    est_cfr = calculate_va(data.copy(), 'score', 'teacher', ['x1', 'x2'], 
                          ['teacher', 'class id'], ['c1'], method='cfr', 
                          add_constant=False)[0]
    est_fk = calculate_va(data, 'score', 'teacher', ['x1', 'x2'], 
              ['teacher', 'class id'], ['c1'], method='fk', add_constant=False)
    return [est_ks, est_cfr, est_fk]


def get_empirical_cdf_points(vec):
    """
    e.g. if vec is [1, 6, 2], want to plot
                x = [1, 1,    2,   2,  6,  6]
                y = [0, 1/3, 1/3, 2/3, 2/3, 1]
    """
    k = len(vec)
    vec = np.repeat(np.sort(vec), 2)
    y_points = np.repeat(np.arange(0, 1 + 1/k, 1/k), 2)[1:-1]
    return vec, y_points

recalculate_results = True
make_figure = True
parallel = True
date = datetime.date.today().strftime('%B_%d_%Y')
if recalculate_results:
    n_iters = 400 
    if parallel:
       num_cores = min(cpu_count(), n_iters)
       pool = Pool(num_cores)
       small_sample_results = np.array(pool.map(simulate_and_evaluate, 
                                       ((100, i) for i in range(n_iters))))
       large_sample_results = np.array(pool.map(simulate_and_evaluate, 
                                        ((1000, i) for i in range(n_iters))))
    else:
        small_sample_results = np.vstack((simulate_and_evaluate((100, i)) 
                             for i in range(n_iters)))
        large_sample_results = np.vstack((simulate_and_evaluate((1000, i))
                                          for i in range(n_iters)))
    np.save('monte_carlo_results_small' + date + '.npy', small_sample_results)
    np.save('monte_carlo_results_large' + date + '.npy', large_sample_results)

if make_figure:
    if not recalculate_results:
        small_sample_results = np.load('monte_carlo_results_small' + date + '.npy')
        large_sample_results = np.load('monte_carlo_results_large' + date + '.npy')
    small_sample_results = small_sample_results.T
    large_sample_results = large_sample_results.T
    to_plot_small = [get_empirical_cdf_points(elt) for elt in small_sample_results]
    to_plot_large = [get_empirical_cdf_points(elt) for elt in large_sample_results]

    plt.figure()
    labels = ['KS', 'CFR', 'FK']
    for points, label in zip(to_plot_small, labels):
        plt.plot(points[0], points[1], label=label + 'small')
    for points, label in zip(to_plot_large, labels):
        plt.plot(points[0], points[1], label=label+'large')

    plt.xlabel('Variance of teacher effects')
    plt.title('Empirical CDF of estimated bureaucrat effects')
    plt.legend()
    plt.axvline(x = .01, color='k')
    plt.savefig('monte_carlo_figure' + date)
