from multiprocessing import Pool, cpu_count
from va_functions import *
import random
import warnings

def estimate_mu_covariances(data):
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

def get_each_va(df, var_theta_hat, var_epsilon_hat, var_mu_hat, jackknife):
    def f(df): # df columns are type, size, mean score
        type_ = int(df['type'].values[0])
        return get_va(df, var_theta_hat[type_], var_epsilon_hat[type_], var_mu_hat[type_], jackknife)
        
    if jackknife:
        results = df.groupby(('teacher', 'type'))[['size', 'mean score', 'type']].apply(f)
        df.loc[:, 'va'] = np.hstack(results)
    else:
        results = pd.DataFrame(df.groupby(('teacher', 'type'))[['size', 'mean score', 'type']].apply(f)).reset_index()
        results.columns = ['teacher', 'type', 'va']
        df = pd.merge(df, results)

    return df
    

def get_phi_coefficients(scores, sizes, var_mu, var_theta, var_epsilon, cov_mu, cov_theta):
    s_00, s_11 = [var_mu[i] + np.dot(sizes[i], sizes[i])*var_theta[i]/np.sum(sizes[i])**2 \
                  + var_epsilon[i]/np.sum(sizes[i]) for i in [0,1]]
    s_01 = cov_mu + np.dot(sizes[0], sizes[1])*cov_theta/(np.sum(sizes[0])*np.sum(sizes[1]))

    phi_0 = np.linalg.solve([[s_00, s_01], [s_01, s_11]], [var_mu[0], cov_mu])
    phi_1 = np.linalg.solve([[s_00, s_01], [s_01, s_11]], [cov_mu, var_mu[1]])
    
    return [phi_0, phi_1]
    
# TODO: This really sucks
# Other way of calculating VA, uses data from both genders
def get_va_2(df, var_theta_hat, var_epsilon_hat, var_mu_hat, cov_theta_hat, cov_mu_hat):
    years = remove_duplicates(df['year'].values)
    # Jackknife estimator requires multiple years
    if len(years) < 2:
        return df
        
    # Identify classes when both girls and boys were taught
    classes = set(df[df['type'] ==0]['class id'].values) & set(df[df['type'] == 1]['class id'].values)
    # Keep only classes in which both girls and boys were taught
    df_good_classes = df[[id_ in classes for id_ in df['class id'].values]]

    for year in years:
        # Remove current year for jackknife estimation
        df_type_otheryears = [df_good_classes[(df_good_classes['type'] == i) & (df_good_classes['year'] != year)] for i in [0,1]]
        if (not df_type_otheryears[0].empty) and (not df_type_otheryears[1].empty):
            sizes = np.array([df_type_otheryears[i]['size'].values for i in [0,1]])
            assert len(sizes[0]) > 0
            assert len(sizes[1]) > 0
            scores = np.array([df_type_otheryears[i]['mean score'].values for i in [0,1]])

            phi = get_phi_coefficients(scores, sizes, var_mu_hat, var_theta_hat, var_epsilon_hat, cov_mu_hat, cov_theta_hat)

            va = [phi[i][0]*np.dot(sizes[0], scores[0])/np.sum(sizes[0]) 
                + phi[i][1]*np.dot(sizes[1], scores[1])/np.sum(sizes[1]) for i in [0,1]]

            indices = pd.Series(df['year'] == year)
            df.loc[(indices & (df['type'] == 0)), 'va'] = va[0]
            df.loc[(indices & (df['type'] == 1)), 'va other'] = va[0]
            df.loc[(indices & (df['type'] == 1)), 'va'] = va[1]
            df.loc[(indices & (df['type'] == 0)), 'va other'] = va[1]
            
    return df
    
## Returns VA's and important moments
## a residual can be specified
## Covariates is a list like ['prev score', 'free lunch']
## Column names can specify 'class id', 'student id', and 'type'
def calculate_va(data, covariates, jackknife, class_type_level_vars = [], residual = None, moments = {}, column_names = None, method = 1):
    assert method in [1,2]
    # Fix column names
    # TODO: do the reverse ting at the end
    if column_names is not None:
        data.rename(columns={column_names[k]: k for k in column_names}, inplace=True)
        if 'type' in column_names and column_names['type'] in covariates:
            covariates[covariates.index(column_names['type'])] = 'type'

    for var in ['score', 'student id', 'teacher', 'class id', 'year', 'type']:
        try:
            assert var in data.columns
        except AssertionError:
            raise Exception(var + ' must be in column names')

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
    
    if 'var epsilon' in moments:
        var_epsilon_hat = moments['var epsilon']
    else:
        var_epsilon_hat = [estimate_var_epsilon(class_type_df[class_type_df['type']==i]) for i in [0,1]]
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
    
    var_theta_hat = [ssr[i] - var_mu_hat[i] - var_epsilon_hat[i]
                     for i in [0, 1]]

    n_obs = len(data['residual'].notnull())
    
    if method == 1:
        class_type_df = get_each_va(class_type_df, var_theta_hat, var_epsilon_hat, var_mu_hat, jackknife)
    else:
        for i in range(len(teachers)):
            results[i] = get_va_2(class_type_df[class_type_df['teacher'] == teachers[i]], \
                                 var_theta_hat, var_epsilon_hat, var_mu_hat, cov_theta_hat, cov_mu_hat)
        class_type_df = pd.concat(results)
    
    return [class_type_df, var_mu_hat, corr_mu_hat, var_theta_hat, var_epsilon_hat]
