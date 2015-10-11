import multiprocessing
from multiprocessing import Pool as ThreadPool
from va_functions import *
import pandas as pd

# TODO: always use a leave-year-out estimator, not leave-class-out
def collapse(data):
    class_level_vars = ['teacher', 'class id', 'year', 'true va'] \
        if 'true va' in data.columns \
        else ['teacher', 'class id', 'year',]
    class_df = data.groupby(class_level_vars)['student id'].count().reset_index()
    class_df.columns = class_level_vars + ['size']
    
    # Calculate mean and merge it back into class-level data
    temp = data.groupby(class_level_vars)[['residual', 'continuous var']].mean().reset_index()
    temp.columns = class_level_vars + ['mean score', 'mean continuous var']
    class_df = pd.merge(class_df, temp)
    temp = data.groupby(class_level_vars)['residual', 'continuous var'].var().reset_index()
    temp.columns = class_level_vars + ['variance', 'var continuous var']
    return pd.merge(class_df, temp)

    
# Estimates the variance of some variable that is constant but noisily measured within a teacher
# It must have mean zero
def estimate_variance(data, teachers, variable):
    data = data[data[variable].notnull()]
    assert len(data) > 0
    variance_sum = 0
    n_obs_used = 0

    for teacher in teachers:
        df = data[data['teacher'] == teacher]
        classes = remove_duplicates(df['class id'].values)
        class_pairs = [(classes[i], classes[j]) for j in range(len(classes)) for i in range(j+1, len(classes))]
        
        for pair in class_pairs:
            measurements  = [df[df['class id'] == pair[i]][variable].values[0] for i in [0,1]]
            sizes         = [df[df['class id'] == pair[i]]['size'].values[0] for i in [0,1]]
            variance_sum += measurements[0]*measurements[1]*(sizes[0]+sizes[1])
            n_obs_used   += sizes[0]+sizes[1]

    return variance_sum / n_obs_used
    
def get_mc1(class_level_df):
    z = class_level_df['continuous var'].values
    scores = class_level_df['residual'].values   
    return np.cov(z, scores, bias = 1)[0,1] / np.var(z)
    
def get_mc0(teacher_level_df):
    return teacher_level_df['mean score'].values - teacher_level_df['m_c1'].values * teacher_level_df['mean continuous var'].values

def get_mc1_precision(class_level_df, var_epsilon):
    size = class_level_df['size'].values[0]
    ind_var_variance = class_level_df['var continuous var'].values[0]
    return size * ind_var_variance / var_epsilon

# E[(m_{c1} - mu_{c1})^2]
def get_mc1_squared_error(continuous_var_vector, var_epsilon):
    return var_epsilon / (len(continuous_var_vector) * np.var(continuous_var_vector))

# TODO: test this    
def get_mc0_precision(var_theta, var_epsilon, continuous_var_vector):
    third_term = np.mean(continuous_var_vector)**2 \
        * get_mc1_squared_error(continuous_var_vector, var_epsilon)
    return 1/(var_theta + third_term + var_epsilon / len(continuous_var_vector))

def estimate_var_epsilon_one_class(class_level_df, mu_1):
    residual = class_level_df['residual'] - np.mean(class_level_df['residual']) - mu_1 * (class_level_df['continuous var'] - np.mean(class_level_df['continuous var']))
    return np.mean(residual**2)

def estimate_var_epsilon_continuous_model(class_df, individual_df, teachers):
    var_epsilon_estimates = 0
    class_sizes = 0
    
    for teacher in teachers:
        for class_ in set(class_df[class_df['teacher'] == teacher]['class id'].values):
            this_class_indices = pd.Series((class_df['teacher'] == teacher) \
                                         & (class_df['class id'] == class_))
            m_1 = class_df[this_class_indices]['m_c1'].values[0]
            
            var_epsilon_estimate = estimate_var_epsilon_one_class(individual_df[(individual_df['teacher'] == teacher) & (individual_df['class id'] == class_)], m_1)
            size = class_df[this_class_indices]['size'].values[0]
            
            var_epsilon_estimates += var_epsilon_estimate * size
            class_sizes += size

    return var_epsilon_estimates / class_sizes

def mse_minimizing_estimator(values, precisions, variance):
    return np.dot(values, precisions)/(np.sum(precisions) + 1/variance)
   
def jackknife_mse_minimizing_estimator(estimates, precisions, years, variance):
    n_classes = len(estimates)
    result = np.zeros(n_classes)
    
    for year in remove_duplicates(years):
        to_use = [y != year for y in years]
        use_estimates = np.where(to_use, estimates, np.zeros(n_classes))
        use_precisions = np.where(to_use, precisions, np.zeros(n_classes))
        
        answer =  np.dot(use_estimates, use_precisions) / (np.sum(use_precisions) + 1/variance)
        
        result = np.where(to_use, answer, result)
        
    return result
    
# columns should contain 'class id', 'continuous var', and 'student id'
# moments can contain 'var epsilon'
def calculate_va_continuous(data, covariates, columns = {}, residual = None, moments = None):
    # Get variables for column names
    data.rename(columns = {columns[k]:k for k in columns})
            
    # Drop teachers who only teach one class
    start = time.time()    
    data = drop_one_class_teachers(data)
    end = time.time()
    
    ## If a residual was not included, residualize scores
    if residual == None:
        data['residual'] = residualize(data, 'score', covariates)
    else:
        data = data.rename(columns = {residual: 'residual'})
        
    data = data[data['residual'].notnull()]     # Drop students with missing scores
    data['continuous var'] = normalize(data['continuous var'].values)
    class_df = collapse(data)    
    
    # Reduce data to class level
    teachers = remove_duplicates(class_df['teacher'].values)
    
    for teacher in teachers:
        for class_ in set(class_df[class_df['teacher'] == teacher]['class id'].values):
            m_c1 = get_mc1(data[(data['teacher'] == teacher) & (data['class id'] == class_)])
            class_df.loc[(class_df['teacher'] == teacher) & (class_df['class id'] == class_), 'm_c1'] = m_c1
            
        class_df.loc[class_df['teacher'] == teacher, 'm_c0'] = get_mc0(class_df[class_df['teacher'] == teacher])
    
    var_mu_0 = estimate_variance(class_df, teachers, 'm_c0')
    var_mu_1 = estimate_variance(class_df, teachers, 'm_c1')
    var_epsilon = estimate_var_epsilon_continuous_model(class_df, data, teachers)
    var_theta = np.var(data['residual']) - var_mu_0  - var_mu_1 - var_epsilon
    
    for teacher in teachers:
        for class_ in set(class_df[class_df['teacher'] == teacher]['class id'].values):
            indices = pd.Series((class_df['teacher'] == teacher) & (class_df['class id'] == class_))
            # Set m_c1 precisions
            class_df.loc[indices, 'precision m_c1'] = \
                get_mc1_precision(class_df[indices], var_epsilon)
            # Set m_c0 precisions
            continuous_var_vector = data[(data['teacher'] == teacher) & (data['class id'] == class_)]['continuous var'].values
            class_df.loc[indices, 'precision m_c0'] = get_mc0_precision(var_theta, var_epsilon, continuous_var_vector)
    
    # Estimate mu_0 and mu_1
    for teacher in teachers:
        df = class_df[class_df['teacher'] == teacher]
        class_df.loc[class_df['teacher'] == teacher, 'mu_0'] = jackknife_mse_minimizing_estimator(df['m_c0'].values, df['precision m_c0'].values, df['year'].values, var_mu_0)
        class_df.loc[class_df['teacher'] == teacher, 'mu_1'] = jackknife_mse_minimizing_estimator(df['m_c1'].values, df['precision m_c1'].values, df['year'].values, var_mu_1)
    
    return class_df, var_mu_0, var_mu_1, var_theta, var_epsilon
