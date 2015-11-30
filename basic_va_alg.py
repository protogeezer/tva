from multiprocessing import cpu_count
from multiprocessing import Pool as ThreadPool
from va_functions import *
import pandas as pd
import copy
import warnings
import random


def estimate_mu_variance(data, n_iters):
    def f(vector):
        est = 0
        for i, first in enumerate(vector):
            for second in vector[i+1:]:
                est += first * second

        n = len(vector) * (len(vector) - 1) / 2
        return est, n
        
    def get_var_mu_hat(array, quietly=True):
        val = np.sum(array[:, 0]) / np.sum(array[:, 1])
        if val <= 0:
            if not quietly:
                warnings.warn('Var mu hat is negative. Measured to be ' + str(val))
            val = 0
        return val
        
    mu_estimates = np.array(list(data.groupby('teacher')['mean score'].apply(f).values))
    var_mu_hat =  get_var_mu_hat(mu_estimates, quietly=False)
    # bootstrap
    var_mu_hat_distribution = [get_var_mu_hat(get_bootstrap_sample(mu_estimates)) for i in range(n_iters)]
        
    return var_mu_hat, \
          (np.sum([(elt - var_mu_hat)**2 for elt in var_mu_hat_distribution]) / (n_iters - 1))**.5

def get_each_va(df, var_theta_hat, var_epsilon_hat, var_mu_hat, jackknife):
    def f(data):
        return get_va(data, var_theta_hat, var_epsilon_hat, var_mu_hat, jackknife)
        
    if jackknife:
        results = df.groupby('teacher')[['size', 'mean score']].apply(f).values
        print(results)
        df.loc[:, 'va'] = np.hstack(results)
    else:
        results = pd.DataFrame(df.groupby('teacher')[['size', 'mean score']].apply(f).reset_index())
        results.columns = ['teacher', 'va']
        df = pd.merge(df, results)
    print(df)
    return df

# Returns VA's and important moments
# a residual can be specified
# Covariates is a list like ['prev score', 'free lunch']
# moments contains 'var epsilon', 'var mu', 'cov mu', 'var theta', 'cov theta'
# Column names can specify 'class id', 'student id', and 'teacher'
# class_level_vars can contain any variables are constant at the class level and will stay in the final data set
def calculate_va(data, covariates, jackknife, residual=None, moments=None, column_names=None, class_level_vars=['teacher', 'class id'], categorical_controls = None, moments_only = False, n_bootstrap_samples = 1000):
    ## First, a bunch of data processing
    if moments is None:
        moments = {}

    # Fix column names
    if column_names is not None:
        data.rename(columns=column_names, inplace=True)

    # If a residual was not included, residualize scores
    if residual is None:
        data.loc[:, 'residual'], _ = residualize(data, 'score', covariates, 'teacher', categorical_controls)
    else:
        data.rename(columns={residual: 'residual'}, inplace=True)

    data = data[data['residual'].notnull()]  # Drop students with missing scores
    assert len(data) > 0
        
    ssr = np.var(data['residual'].values)  # Calculate sum of squared residuals

    # Collapse data to class level
    # Count number of students in class
    class_df = data.groupby(class_level_vars).size().reset_index()
    class_df.columns = class_level_vars + ['size']

    # Calculate mean and merge it back into class-level data
    class_df.loc[:, 'mean score'] = data.groupby(class_level_vars)['residual'].mean().values
    class_df.loc[:, 'var'] = data.groupby(class_level_vars)['residual'].var().values
    assert len(class_df) > 0
    
    if jackknife: # Drop teachers teaching only one class
        class_df = drop_one_class_teachers(class_df)

    ## Second, calculate a bunch of moments
    # Calculate variance of epsilon
    if 'var epsilon' in moments:
        var_epsilon_hat = moments['var epsilon']
    else:
        var_epsilon_hat = estimate_var_epsilon(class_df)
    assert var_epsilon_hat > 0 

    # Estimate TVA variances and covariances
    if 'var mu' in moments:
        var_mu_hat = moments['var mu']
    else:
        var_mu_hat, var_mu_hat_se = estimate_mu_variance(class_df, n_bootstrap_samples)

    # Estimate variance of class-level shocks
    var_theta_hat = moments.get('var theta', ssr - var_mu_hat - var_epsilon_hat)
    if var_theta_hat < 0:
        warnings.warn('Var theta hat is negative. Measured to be ' + str(var_theta_hat))
        var_theta_hat = 0

    if moments_only:
        return var_mu_hat, var_theta_hat, var_epsilon_hat, var_mu_hat_se
        
    results = get_each_va(class_df, var_theta_hat, var_epsilon_hat, var_mu_hat, jackknife)
    if column_names is not None:
        results.rename(columns={column_names[key]:key for key in column_names}, inplace=True)
                   
    return results, var_mu_hat, var_theta_hat, var_epsilon_hat, var_mu_hat_se, 0
