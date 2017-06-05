from simulate_baseline import simulate
from basic_va_alg import calculate_va
import numpy as np
import pandas as pd
from hdfe import Groupby
import time

var_mu = .024

params = {'num teachers': 100, 'beta':[2, 3], 'sd mu': var_mu**.5, 'sd theta':.178**.5, 
          'sd epsilon':(1-.024-.178)**.5, 'mean class size':20, 'mean classes taught':3}

n_iters = 8 
#n_covariate_numbers = 3 
#n_covariate_list = [0, 5, 50, 90]
n_covariate_list = [5]
n_covariate_numbers = len(n_covariate_list)

var_mu_hat = np.zeros((n_covariate_numbers, n_iters))
se_var_mu_hat = np.zeros((n_covariate_numbers, n_iters))
predictable_var_error = np.zeros((n_covariate_numbers, n_iters))
bias_correction = np.zeros((n_covariate_numbers, n_iters))
total_var_error = np.zeros((n_covariate_numbers, n_iters))
total_var_se = np.zeros((n_covariate_numbers, n_iters))

# hat_variance = np.zeros((n_covariate_numbers, n_iters))
error_lambda = [[[] for _ in range(n_iters)]
                for _ in range(n_covariate_numbers)]
se_lambda = [[[] for _ in range(n_iters)]
             for _ in range(n_covariate_numbers)]

for i, n_covariates in enumerate(n_covariate_list):
    print('\n')
    print(i)
    covariates = ['x' + str(elt) for elt in range(1, n_covariates + 3)]
    lambda_ = np.zeros(len(covariates))
    lambda_[1] = 1

    for n in range(n_iters):
        print(n)

        data = simulate(params, n)

        data['id'] = data['teacher'] * np.max(data['class id']) + data['class id']
        assert np.all(np.diff(data['id']) >= 0)

        # Generate useless covariates
        np.random.seed(n)
        for elt in range(n_covariates):
            data['x' + str(elt + 3)] = np.random.normal(0, 1, len(data))

        est_mle = calculate_va(data, 'score', 'teacher', covariates, 
                               ['id'], categorical_controls=None, method='mle')

        var_mu_hat[i, n] = est_mle['sigma mu squared']
        se_var_mu_hat[i, n] = est_mle['asymp var'][0, 0]
        predictable_var_error[i, n] = est_mle['predictable var'] - np.var(est_mle['x bar bar'].dot(lambda_))
        bias_correction[i, n] = est_mle['bias correction']

        error_lambda[i][n] = est_mle['lambda'] - lambda_
        s = -1 - len(lambda_)
        se_lambda[i][n] = np.sqrt(np.diag(est_mle['asymp var'][s:-1, s:-1]))
        total_var_error[i,n] = est_mle['total var'] - np.var(data['true va'])
        total_var_se[i, n] = est_mle['total var se']
        
    var_mu_error = var_mu_hat[i, :] - var_mu
    t_stat_variance_error = var_mu_error - se_var_mu_hat[i, :]
    print('Mean variance error', np.mean(var_mu_error))
    print('Variance of t stat', np.var(t_stat_variance_error))
    t_stat = np.array(error_lambda[i]) / np.array(se_lambda[i])
    print('Mean t statistic', np.mean(t_stat))
    print('Var t statistic', np.var(t_stat))
    print('Mean error in predictable variance', np.mean(predictable_var_error[i, :]))
    print('Mean bias correction', np.mean(bias_correction[i, :]))
    print('Bias correction overshoots by', 
            np.mean(bias_correction[i, :]) - np.mean(predictable_var_error[i, :]))
    print('Mean error in total var', total_var_error[:, n])
    t_stat = total_var_error[i, :] / total_var_se
    print('Mean t stat', np.mean(t_stat))
    print('T stat variance', np.var(t_stat))

