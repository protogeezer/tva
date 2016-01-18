from va_functions import *
import pandas as pd
import warnings
import random


def estimate_mu_variance(data):
    def f(vector):
        try:
            score_1, score_2 = random.sample(set(vector), 2)
            return [score_1 * score_2, 1]
        except ValueError:
            return [0, 0]
            
    def weighted_mean(arr):
        return np.sum(arr[:, 0]) / np.sum(arr[:, 1])
        
    mu_estimates = np.array(list(data.groupby('teacher')['mean score'].apply(f).values))
    mu_hat = weighted_mean(mu_estimates)
    
    bootstrap_samples = [weighted_mean(get_bootstrap_sample(mu_estimates)) for i in range(1000)]
    return mu_hat, [np.percentile(bootstrap_samples, 2.5), np.percentile(bootstrap_samples, 97.5)]

def get_each_va(df, var_theta_hat, var_epsilon_hat, var_mu_hat, jackknife):
    def f(data):
        return get_va(data, var_theta_hat, var_epsilon_hat, var_mu_hat, jackknife)
        
    if jackknife:
        results = df.groupby('teacher')[['size', 'mean score']].apply(f).values
        df.loc[:, 'va'] = np.hstack(results)
    else:
        results = pd.DataFrame(df.groupby('teacher')[['size', 'mean score']].apply(f).reset_index())
        results.columns = ['teacher', 'va']
        df = pd.merge(df, results)

    return df

# Returns VA's and important moments
# a residual can be specified
# Covariates is a list like ['prev score', 'free lunch']
# moments contains 'var epsilon', 'var mu', 'cov mu', 'var theta', 'cov theta'
# Column names can specify 'class id', 'student id', and 'teacher'
# class_level_vars can contain any variables are constant at the class level and will stay in the final data set
def calculate_va(data, covariates, jackknife, residual=None, moments=None, column_names=None, class_level_vars=['teacher', 'class id'], categorical_controls = [], moments_only = False):
    ## First, a bunch of data processing
    if moments is None:
        moments = {}

    # Fix column names
    if column_names is not None:              
        data.rename(columns=column_names, inplace=True)

    # If a residual was not included, residualize scores
    if residual is None:
        data.loc[:, 'residual'], beta = residualize(data, 'score', covariates, 'teacher', categorical_controls)
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
        var_mu_hat, var_mu_hat_ci = estimate_mu_variance(class_df)
    if var_mu_hat <= 0:
        warnings.warn('Var mu hat is negative. Measured to be ' + str(var_mu_hat))
        var_mu_hat = 0
        

    # Estimate variance of class-level shocks
    var_theta_hat = moments.get('var theta', ssr - var_mu_hat - var_epsilon_hat)
    if var_theta_hat < 0:
        warnings.warn('Var theta hat is negative. Measured to be ' + str(var_theta_hat))
        var_theta_hat = 0

    if moments_only:
        return var_mu_hat, var_theta_hat, var_epsilon_hat, var_mu_hat_ci
        
    results = get_each_va(class_df, var_theta_hat, var_epsilon_hat, var_mu_hat, jackknife)
    if column_names is not None:
        results.rename(columns={column_names[key]:key for key in column_names}, inplace=True)
    
    return results, var_mu_hat, var_theta_hat, var_epsilon_hat, var_mu_hat_ci
