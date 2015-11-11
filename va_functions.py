import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2 as chi2c
import numpy.linalg as linalg
import pandas as pd

def timer_print(string):
    print(string)

def estimate_var_epsilon(data):
    data = data[data['var'].notnull()]
    var_epsilon_hat = np.dot(data['var'].values, data['size'].values)/np.sum(data['size'])
    assert var_epsilon_hat > 0
    return var_epsilon_hat
    
def get_average(df, var_name, group_name):
    grouped = df.groupby(group_name)[var_name]
    return grouped.apply(lambda group: group.mean() + 0 * group)
    
# Demeaning with one or two fixed effects 
def demean(df, var_name, first_group, second_group = None):
    if second_group is None:
        return df[var_name] - get_average(df, var_name, first_group)  
    return df[var_name] - get_average(df, var_name, first_group) \
           - get_average(df, var_name, second_group) + np.mean(df[var_name])


# Calculates beta using y_name = x_names * beta + group_name (dummy) + dummy_control_name
# Returns               y_name - x_names * beta - dummy_control_name
def residualize(df, y_name, x_names, first_group, second_group = None):
    df['constant'] = 1
    x_names.append('constant')
    # Drop missing values to calculate beta
    df_no_null = df[[y_name, first_group] + x_names].dropna() if second_group is None \
                 else df[[y_name, first_group, second_group] + x_names].dropna()

    Y = np.array(demean(df_no_null, y_name, first_group, second_group))
    X = np.transpose([demean(df_no_null, x, first_group, second_group)
                  for x in x_names])
    try:
        beta = linalg.lstsq(X, Y)[0]
    except ValueError:
        raise Exception('Your covariates may be too large relative to the length of your data. This may happen after dropping many null values.')
    
    if second_group is None:
        return df[y_name] - np.dot(df[x_names], beta), beta 
    else:
        x_demeaned = np.transpose([demean(df, x, second_group) + np.mean(df[x]) for x in x_names])
        return demean(df, y_name, second_group) + np.mean(df[y_name]) - np.dot(x_demeaned, beta), beta


def get_teacher_class_map(data, teachers):
    return {teacher: data[data['teacher'] == teacher]['class id'].values for teacher in teachers}


# Mean 0, variance 1
def normalize(vector):
    vector = vector - np.mean(vector)
    return vector / np.std(vector)


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


def drop_one_class_teachers(class_df, teacher_class_map):
    keep = [len(teacher_class_map[t]) >1 for t in class_df['teacher'].values]
    return class_df[keep]


def binscatter(x, y, nbins):
    assert len(x) == len(y)
    # sort according to x
    indices = np.argsort(x)
    x = [x[i] for i in indices]
    y = [y[i] for i in indices]
    assert x == sorted(x)
    
    bins = np.zeros(nbins)
    y_means = np.zeros(nbins)
    
    for i in range(0, nbins):
        start = len(x) * i / nbins
        end = len(x) * (i+1) / nbins
        bins[i] = np.mean(x[int(start):int(end)])
        y_means[i] = np.mean(y[int(start):int(end)])
    return bins, y_means 


# p-value of chi-squared statistic
def do_chi2_test(measurements, true_values, errors):
    chi2_stat = np.sum([((m - v) / e)**2 for m, v, e in zip(measurements, true_values, errors)])
    return chi2_stat, chi2.cdf(chi2_stat, len(measurements))


def check_calibration(errors, precisions):         
    mean_error = np.mean(errors)
    se = (np.var(errors) / len(errors))**.5

    standardized_errors = errors**2 * precisions
    mean_standardized_error = np.mean(standardized_errors)
    standardized_error_se = (np.var(standardized_errors) / len(standardized_errors))**.5
    assert mean_error > -3 * se
    assert mean_error < 3 * se
    assert mean_standardized_error > 1 - 2 * standardized_error_se 
    assert mean_standardized_error < 1 + 2 * standardized_error_se 
