## This file should be run as the first step of va_alg_two_groups.py
from va_functions import *
import time
import warnings

# Data should be at class level
# Returns covariance matrix of teacher effects and covariance between girl and boy class shocks
# Ben thinks Liz is the bestest ever
def estimate_mu_covariances(data, teachers, teacher_class_map):
    cov_mu_00, cov_mu_11, cov_mu_01 = 0, 0, 0
    n_obs_used_00, n_obs_used_11, n_obs_used_01 = 0, 0, 0
    cov_theta_01 = 0
    n_obs_used_theta = 0

    for teacher in teachers:
        df = data[(data['teacher'] == teacher) & (data['mean score'].notnull())]
        df_0 = df[df['type'] == 0]
        df_1 = df[df['type'] == 1]
        
        classes_0 = df_0['class id'].values
        classes_1 = df_1['class id'].values
        
        if not np.array_equal(classes_0, classes_1):
            continue
            
        scores_0 = df_0['mean score'].values
        scores_1 = df_1['mean score'].values
        sizes_0 = df_0['size'].values
        sizes_1 = df_1['size'].values
        
        assert len(scores_0) == len(sizes_0)
        assert len(scores_1) == len(sizes_1)
        
        for i in range(len(scores_0)):
            cov_theta_01 += scores_0[i] * scores_1[i] * (sizes_0[i] + sizes_1[i])
            n_obs_used_theta += sizes_0[i] + sizes_1[i]
            for j in range(i+1, len(scores_0)):
                cov_mu_00 += scores_0[i] * scores_0[j] * (sizes_0[i] + sizes_0[j])
                n_obs_used_00 += sizes_0[i] + sizes_0[j]
                cov_mu_11 += scores_1[i] * scores_1[j] * (sizes_1[i] + sizes_1[j])
                n_obs_used_11 += sizes_1[i] + sizes_1[j]
                
                cov_mu_01 += scores_0[i] * scores_1[j] * (sizes_0[i] + sizes_1[j]) \
                             + scores_1[i] * scores_0[j] * (sizes_1[i] + sizes_0[j])
                n_obs_used_01 += sizes_0[i] + sizes_1[j] + sizes_1[i] + sizes_0[j]

        assert n_obs_used_theta > 0
                
    try:
        var_mu_hat = [cov_mu_00 / n_obs_used_00, cov_mu_11 / n_obs_used_11]
    except ZeroDivisionError:
        print(data)
        raise Exception('Not enough teachers teach classes with students of both types')

    cov_mu_hat = cov_mu_01 / n_obs_used_01
    cov_theta_01 = cov_theta_01 / n_obs_used_theta - cov_mu_hat
    
    assert var_mu_hat[0] > 0 
    assert var_mu_hat[1] > 0
    
    return var_mu_hat, cov_mu_hat, cov_theta_01

## Returns VA's and important moments
## a residual can be specified
## Covariates is a list like ['prev score', 'free lunch']
## Column names can specify 'class id', 'student id', and 'type'
def calculate_covariances(data, covariates, residual = None, moments = None, column_names = None, class_type_level_vars = []):
    # Fix column names
    if column_names is not None:
        start = time.time()
        data.rename(columns={column_names[k]: k for k in column_names}, inplace=True)
        if 'type' in column_names and column_names['type'] in covariates:
            covariates[covariates.index(column_names['type'])] = 'type'
        timer_print('Time to rename columns ' + str(time.time() - start))

    for var in ['score', 'student id', 'teacher', 'class id', 'year', 'type']:
        try:
            assert var in data.columns
        except AssertionError:
            raise Exception(var + ' must be in column names')

    if moments is None:
        moments = {}

    # If a residual was not included, residualize scores
    if residual is None:
        start = time.time()
        data.loc[:, 'residual'], _ = residualize(data, 'score', covariates, 'teacher')
        timer_print('Time to residualize ' + str(time.time() - start))
    else:
        data.rename(columns={residual: 'residual'}, inplace=True)
    
    start = time.time()
    data = data[data['residual'].notnull()] # Drop students with missing scores
    timer_print('Time to drop students with missing residuals ' + str(time.time() - start))
    
    start = time.time()
    ssr = [np.var(data[data['type'] == i]['residual'].values) for i in [0,1]] # sum of squared residuals
    timer_print('Time to calculae SSR ' + str(time.time() - start))
    
    # Reduce data to class level
    # Count number of students in class
    start = time.time()
    class_type_level_vars = ['teacher', 'class id', 'type', 'year'] + class_type_level_vars
    class_type_df = data.groupby(class_type_level_vars)['student id'].count().reset_index()
    class_type_df.columns = class_type_level_vars + ['size']
    
    teachers = remove_duplicates(class_type_df['teacher'].values)
    teacher_class_map = get_teacher_class_map(class_type_df, teachers)
    
    # Calculate mean and merge it back into class-level data
    temp = data.groupby(class_type_level_vars)['residual'].mean().reset_index()
    temp.columns = class_type_level_vars + ['mean score']
    class_type_df = pd.merge(class_type_df, temp)
    temp = data.groupby(class_type_level_vars)['residual'].var().reset_index()
    temp.columns = class_type_level_vars + ['var']
    class_type_df = pd.merge(class_type_df, temp)
    timer_print('Time to collapse to class level ' + str(time.time() - start))

    assert len(class_type_df.index) > 0
    
    start = time.time()
    var_epsilon_hat = moments.get('var epsilon', [estimate_var_epsilon(class_type_df[class_type_df['type']==i]) for i in [0,1]])
    timer_print('Time to calculate var epsilon ' + str(time.time() - start))
    assert np.array(var_epsilon_hat).shape == (2,)
    
    # Estimate TVA variances and covariances
    start = time.time()
    var_mu_hat, cov_mu_hat, cov_theta_01 = [moments['var mu'], moments['cov mu'], moments['cov theta']] \
              if 'var_mu' in moments and 'cov_mu' in moments  and 'cov theta' in moments \
              else estimate_mu_covariances(class_type_df, teachers,
                                           teacher_class_map)
    timer_print('Time to calculate mu covariance ' + str(time.time() - start))
    
    assert np.array(var_mu_hat).shape == (2,)
    assert np.array(cov_mu_hat).shape == ()
    
    corr_mu_hat = cov_mu_hat/(var_mu_hat[0]*var_mu_hat[1])**(.5)
    if not (corr_mu_hat > -1 and corr_mu_hat < 1):
        warnings.warn('Calculated corr_mu_hat is ' + str(corr_mu_hat) + '; it should be between 0 and 1. Your data may be too small.')
    
    var_theta_hat = [ssr[i] - var_mu_hat[i] - var_epsilon_hat[i]
                     for i in [0, 1]]

    n_obs = len(data['residual'].notnull())
    
    return {'var mu':var_mu_hat, 'cov mu':cov_mu_hat, 'corr mu':corr_mu_hat,\
            'var epsilon':var_epsilon_hat, 'var theta':var_theta_hat, 'cov theta':cov_theta_01}, \
            n_obs, class_type_df, teachers
