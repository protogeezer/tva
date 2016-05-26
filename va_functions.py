import numpy as np
import numpy.linalg as linalg
import pandas as pd
import scipy.sparse as sps
from functools import reduce


def estimate_var_epsilon(data):
    data = data[data['var'].notnull()]
    var_epsilon_hat = np.dot(data['var'].values, data['size'].values)/np.sum(data['size'])
    assert var_epsilon_hat > 0
    return var_epsilon_hat
    
def fe_demean(df, variables, group):
    f = lambda df: df - np.mean(df.values, axis = 0)
    return df.groupby(group)[variables].apply(f)

# Calculates beta using y_name = x_names * beta + group_name (dummy) + dummy_control_name
# Returns               y_name - x_names * beta - dummy_control_name
def residualize(df, y_name, x_names, first_group, second_groups = None):
    y = df[y_name].values
    
    if len(x_names) == 0 and second_groups is None: # don't do anything
        return y - np.mean(y), [np.mean(y)]
    else:   
        # If only one set of FE's, use demeaning transformation
        if second_groups is None: 
            x_demeaned = fe_demean(df, x_names, first_group)
            beta = linalg.lstsq(x_demeaned, y)[0]
            resid = y - np.dot(df[x_names], beta)
            
        else: # Create dummies if there is are fixed effects
            def get_dummies(group): # Drops first dummy
                return pd.get_dummies(df, columns=[group]).iloc[:, len(df.columns):]
            
            dummy_df = reduce(lambda x, y: x.join(y), \
                              (get_dummies(g) for g in second_groups))
                
            x_df = dummy_df.join(df[x_names])
            first_group_df = pd.get_dummies(df, columns = first_group).iloc[:, len(df.columns)-1:]
            n = len(first_group_df.columns)
            rhs = (first_group_df.join(x_df)).values
            beta = sparse.
            
#            teacher_df = pd.DataFrame(df[first_group])
#            x_demeaned = fe_demean(x_df.join(teacher_df), x_df.columns, first_group).values
#       
#            # remove columns that are close to zero
#            n, k = x_demeaned.shape
#            not_all_zero = np.array([not np.allclose(x_demeaned[:, i], np.zeros(n), 
#                                     atol = 1/n) for i in range(k)])
#            print('%d columns being dropped because of collinearity' % (len(not_all_zero) - sum(not_all_zero)))
#            x_demeaned = x_demeaned[:, not_all_zero]
#            x = x_df.values[:, not_all_zero]
        
            beta = linalg.lstsq(x_demeaned, y)[0]
            resid = y - np.dot(x, beta)
            
        return resid - np.mean(resid), beta


# function by some internet person, not me
def remove_duplicates(seq): 
    # order preserving
    seen = {}
    result = []
    for item in seq:
        marker = item
        if marker in seen:
            continue
        seen[marker] = 1
        result.append(item)
    return result


def drop_one_class_teachers(class_df):
    try:
        grouped = class_df.groupby('teacher')['class id']
    except KeyError:
        print(class_df.head())
        assert False
    df = pd.DataFrame(grouped.apply(lambda x: len(x) > 1).reset_index())
    df.columns = ['teacher', 'keep']
    class_df = pd.merge(df, class_df)
    return class_df[class_df['keep']].drop('keep', axis=1)


def binscatter(x, y, nbins):
    assert len(x) == len(y)
    # sort according to x
    indices = np.argsort(x)
    x = [x[i] for i in indices]
    y = [y[i] for i in indices]
    assert x == sorted(x)
    
    bins = np.zeros(nbins)
    y_means = np.zeros(nbins)
    y_medians = np.zeros(nbins)
    
    for i in xrange(0, nbins):
        start = len(x) * i / nbins
        end = len(x) * (i+1) / nbins
        bins[i] = np.mean(x[int(start):int(end)])
        y_means[i] = np.mean(y[int(start):int(end)])
        y_medians[i] = np.median(y[int(start):int(end)])
    return bins, y_means, y_medians


def check_calibration(errors, precisions):         
    mean_error = np.mean(errors)
    se = (np.var(errors) / len(errors))**.5

    standardized_errors = errors**2 * precisions
    mean_standardized_error = np.mean(standardized_errors)
    standardized_error_se = (np.var(standardized_errors) \
                              / len(standardized_errors))**.5
    assert mean_error > -3 * se
    assert mean_error < 3 * se
    assert mean_standardized_error > 1 - 2 * standardized_error_se 
    assert mean_standardized_error < 1 + 2 * standardized_error_se 

# Now returns [mu-hat, variance of mu-hat]
def get_va(df, var_theta_hat, var_epsilon_hat, var_mu_hat, jackknife):
    array = df.values
    precisions = np.array([1 / (var_theta_hat + var_epsilon_hat / class_size) 
                          for class_size in array[:, 0]])
    numerators = precisions * array[:, 1]

    precision_sum = np.sum(precisions)
    num_sum = np.sum(numerators)
    # TODO: also return unshrunk va and variance
    if jackknife:
        denominators = np.array([precision_sum - p for p in precisions]) \
                 + 1 / var_mu_hat
        return [[(num_sum - n) / d, 1 / d]
                          for n, d in zip(numerators, denominators)]
    else:
        denominator = precision_sum + 1 / var_mu_hat
        return [num_sum / denominator, 1 / denominator]
        
def get_bootstrap_sample(myList):
    indices = np.random.choice(range(len(myList)), len(myList))
    return myList[indices]
