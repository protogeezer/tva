from multiprocessing import cpu_count
from multiprocessing import Pool as ThreadPool
from va_functions import *
import pandas as pd
import time
import copy

def timer_print(string):
    pass
#    print(string)

# VA for one teacher, for each class
def get_va(df, teacher, classes, var_theta_hat, var_epsilon_hat, var_mu_hat, jackknife):
    df = copy.copy(df)
    classes = df['class id'].values
    assert len(classes) == len(set(classes))

    precisions = np.zeros(len(classes))
    numerators = np.zeros(len(classes))
    
    for k in range(len(classes)):
        class_size = df['size'].values[k]
        precisions[k] = 1 / (var_theta_hat + var_epsilon_hat / class_size)
        numerators[k] = precisions[k] * df['mean score'].values[k]
        
    precision_sum = np.sum(precisions)
    if jackknife:
        num_sum = np.sum(numerators)
        va = np.array([(num_sum - n) / (precision_sum - p + 1 / var_mu_hat)
                           for n, p in zip(numerators, precisions)])
        variance = np.array([var_mu_hat / (1 + var_mu_hat * (precision_sum - p))
                                 for p in precisions])
        return np.concat((np.ones(len(va)) * teacher, va, variance))
    else:
        va = np.sum(numerators) / (precision_sum + 1 / var_mu_hat)
        variance =  1 / (precision_sum + 1 / var_mu_hat)
        return [teacher, va, variance]

def get_va_from_tuple(input_tuple):
    df, t, classes, var_theta_hat, var_epsilon_hat, var_mu_hat, jackknife = input_tuple
    return get_va(df[df['teacher'] == t], teacher, classes, var_theta_hat, var_epsilon_hat, var_mu_hat, jackknife)

def estimate_mu_variance(data, teachers, teacher_class_map):
    variance = 0
    n_obs_used = 0

    for teacher in teachers:
        data_this_teacher = data[(data['teacher'] == teacher) & (data['mean score'].notnull())]
        classes = teacher_class_map[teacher]
        
        def calculate_var_and_n(scores, sizes):
            assert len(scores) == len(sizes)
            var = 0
            n = 0
            for i in range(len(scores)):
                score_i, size_i = scores[i], sizes[i]
                for j in range(i+1, len(scores)):
                    score_j, size_j = scores[j], sizes[j]
                    
                    var += score_i * score_j * (size_i + size_j)
                    n += size_i + size_j
            return var, n
        
        scores = data_this_teacher['mean score'].values
        sizes = data_this_teacher['size'].values
        var, n = calculate_var_and_n(scores, sizes)     
        variance += var 
        n_obs_used += n   

    return variance / n_obs_used


# Returns VA's and important moments
# a residual can be specified
# Covariates is a list like ['prev score', 'free lunch']
# moments contains 'var epsilon', 'var mu', 'cov mu', 'var theta', 'cov theta'
# Column names can specify 'class id', 'student id', and 'teacher'
# class_level_vars can contain any variables are constant at the class level and will stay in the final data set
def calculate_va(data, covariates, jackknife, residual=None, moments=None, column_names=None, parallel=False, class_level_vars=['teacher', 'class id'], categorical_controls = None, num_cores = None, moments_only = False):
    ## First, a bunch of data processing
    if moments is None:
        moments = {}

    # Fix column names
    start = time.time()
    if column_names is not None:
        data.rename(columns=column_names, inplace=True)
    timer_print('Rename columns time ' + str(time.time() - start))

    # If a residual was not included, residualize scores
    start = time.time()
    if residual is None:
        data.loc[:, 'residual'], _ = residualize(data, 'score', covariates, 'teacher', categorical_controls)
    else:
        data.rename(columns={residual: 'residual'}, inplace=True)
    timer_print('Residualize time ' + str(time.time() - start))

    data = data[data['residual'].notnull()]  # Drop students with missing scores
    assert len(data) > 0
        
    start = time.time()
    ssr = np.var(data['residual'].values)  # Calculate sum of squared residuals
    timer_print('SSR time ' + str(time.time() - start))

    # Collapse data to class level
    # Count number of students in class
    start = time.time()
    class_df = data.groupby(class_level_vars).size().reset_index()
    class_df.columns = class_level_vars + ['size']

    # Calculate mean and merge it back into class-level data
    class_df.loc[:, 'mean score'] = data.groupby(class_level_vars)['residual'].mean().values
    class_df.loc[:, 'var'] = data.groupby(class_level_vars)['residual'].var().values
    assert len(class_df) > 0
        
    timer_print('Time to collapse to class level ' + str(time.time() - start))
    
    if jackknife: # Drop teachers teaching only one class
        start = time.time()
        class_df = drop_one_class_teachers(class_df, get_teacher_class_map(class_df, remove_duplicates(class_df['teacher'].values)))
        timer_print('Time to drop one class teachers ' + str(time.time() - start))
    teachers = remove_duplicates(class_df['teacher'].values)
    teacher_class_map = get_teacher_class_map(class_df, teachers)
    assert teachers[0] in class_df['teacher'].values
    ## Second, calculate a bunch of moments
    # Calculate variance of epsilon
    start = time.time()
    if 'var epsilon' in moments:
        var_epsilon_hat = moments['var epsilon']
    else:
        var_epsilon_hat = estimate_var_epsilon(class_df)
    timer_print('Time to estimate var epsilon' + str(time.time() - start))

    # Estimate TVA variances and covariances
    start = time.time()
    if 'var mu' in moments:
        var_mu_hat = moments['var mu']
    else:
        var_mu_hat = estimate_mu_variance(class_df, teachers, teacher_class_map)

    timer_print('Time to estimate var mu hat ' + str(time.time() - start))

    # Estimate variance of class-level shocks
    start = time.time()
    var_theta_hat = moments.get('var theta', ssr - var_mu_hat - var_epsilon_hat)
    timer_print('Time to estimate var theta hat ' + str(time.time() - start))

    if moments_only:
        return var_mu_hat, var_theta_hat, var_epsilon_hat
    else:
        ## Finally, compute value-added
        if parallel:
            if num_cores is None:
                num_cores = cpu_count()
            pool = ThreadPool(num_cores)
            params = [(class_df, t, var_theta_hat, var_epsilon_hat, var_mu_hat, jackknife) for t in teachers]
            results = pool.map(get_va_from_tuple, params)
            
            pool.close()
            pool.join()
            results = np.vstack(results)
        else:
            assert teachers[0] in class_df['teacher'].values
            start = time.time()
            results = [get_va(class_df[class_df['teacher'] == t], t, teacher_class_map[t], var_theta_hat, var_epsilon_hat, var_mu_hat, jackknife)
                       for t in teachers]
            timer_print('Time to compute VA ' + str(time.time() - start))

        results = pd.DataFrame(results)
        results.columns = ['teacher', 'va', 'variance']

        if column_names is not None:
            results.rename(columns={column_names[key]:key for key in column_names}, inplace=True)
                       
        return results, var_mu_hat, var_theta_hat, var_epsilon_hat, len(teachers)
