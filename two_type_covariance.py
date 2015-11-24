## This file should be run as the first step of va_alg_two_groups.py
from va_functions import *
import warnings
import random

def estimate_mu_covariances(data):
    @profile
    def f(df):
        if len(df) <= 2:
            return [0, 0, 0, 0, 0]
        else:
            df = df.values
            i, j = random.sample(range(0, len(df), 2), 2)
            assert df[i, 0] == 0 # check that types are correct
            assert df[j, 0] == 0
            assert df[i+1, 0] == 1
            assert df[j+1, 0] == 1

            score00 = df[i, 1]
            score01 = df[i+1, 1]
            score10 = df[j, 1]
            score11 = df[j+1, 1]

        return [score00*score10, score01*score11, score00*score11+score01*score10, score00*score10+score10*score11, 1]

        
    estimates = np.array(list(data.groupby('teacher')[['type', 'mean score']].apply(f).values))
    print(estimates.shape)
    n = np.sum(estimates[:, 4])
    # TODO: warning if these are negative
    cov_mu_00 = np.sum(estimates[:, 0])/n
    cov_mu_11 = np.sum(estimates[:, 1])/n
    cov_mu_01 = np.sum(estimates[:, 2])/(2*n)
    cov_theta_01 = np.sum(estimates[:, 3])/n - cov_mu_01
    return [cov_mu_00, cov_mu_11], cov_mu_01, cov_theta_01

## Ben thinks Liz is the bestest ever

## Returns VA's and important moments
## a residual can be specified
## Covariates is a list like ['prev score', 'free lunch']
## Column names can specify 'class id', 'student id', and 'type'
def calculate_covariances(data, covariates, residual = None, moments = None, column_names = None, class_type_level_vars = []):
    # Fix column names
    if column_names is not None:
        data.rename(columns={column_names[k]: k for k in column_names}, inplace=True)
        if 'type' in column_names and column_names['type'] in covariates:
            covariates[covariates.index(column_names['type'])] = 'type'

    for var in ['score', 'student id', 'teacher', 'class id', 'year', 'type']:
        try:
            assert var in data.columns
        except AssertionError:
            raise Exception(var + ' must be in column names')

    if moments is None:
        moments = {}

    # If a residual was not included, residualize scores
    if residual is None:
        data.loc[:, 'residual'], _ = residualize(data, 'score', covariates, 'teacher')
    else:
        data.rename(columns={residual: 'residual'}, inplace=True)
    
    data = data[data['residual'].notnull()] # Drop students with missing scores   
    ssr = [np.var(data[data['type'] == i]['residual'].values) for i in [0,1]] # sum of squared residuals
    
    # Reduce data to class level
    # Count number of students in class
    class_type_level_vars = ['teacher', 'class id', 'type', 'year'] + class_type_level_vars
    class_type_df = data.groupby(class_type_level_vars)['student id'].count().reset_index()
    class_type_df.columns = class_type_level_vars + ['size']
    
    teachers = remove_duplicates(class_type_df['teacher'].values)
    
    # Calculate mean and merge it back into class-level data
    temp = data.groupby(class_type_level_vars)['residual'].mean().reset_index()
    temp.columns = class_type_level_vars + ['mean score']
    class_type_df = pd.merge(class_type_df, temp)
    temp = data.groupby(class_type_level_vars)['residual'].var().reset_index()
    temp.columns = class_type_level_vars + ['var']
    class_type_df = pd.merge(class_type_df, temp)

    assert len(class_type_df.index) > 0
    
    var_epsilon_hat = moments.get('var epsilon', [estimate_var_epsilon(class_type_df[class_type_df['type']==i]) for i in [0,1]])
    assert np.array(var_epsilon_hat).shape == (2,)
    
    # Estimate TVA variances and covariances
    var_mu_hat, cov_mu_hat, cov_theta_01 = [moments['var mu'], moments['cov mu'], moments['cov theta']] \
          if 'var_mu' in moments and 'cov_mu' in moments  and 'cov theta' in moments \
          else estimate_mu_covariances(class_type_df)
    
    assert np.array(var_mu_hat).shape == (2,)
    assert np.array(cov_mu_hat).shape == ()
    
    corr_mu_hat = cov_mu_hat/(var_mu_hat[0]*var_mu_hat[1])**(.5)
    if not (corr_mu_hat > -1 and corr_mu_hat < 1):
        warnings.warn('Calculated corr_mu_hat is ' + str(corr_mu_hat) + '; it should be between 0 and 1. Your data may be too small.')
        print('cov_mu_hat')
        print(cov_mu_hat)
        print('var mu hat')
        print(var_mu_hat)
    
    var_theta_hat = [ssr[i] - var_mu_hat[i] - var_epsilon_hat[i]
                     for i in [0, 1]]

    n_obs = len(data['residual'].notnull())
    
    return {'var mu':var_mu_hat, 'cov mu':cov_mu_hat, 'corr mu':corr_mu_hat,\
            'var epsilon':var_epsilon_hat, 'var theta':var_theta_hat, 'cov theta':cov_theta_01}, \
            n_obs, class_type_df, teachers
