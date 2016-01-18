import numpy as np
import numpy.linalg as linalg
import pandas as pd
import scipy.sparse as sparse

class Groupby:
    def __init__(self, keys):
        self.unique_keys = frozenset(keys)
        self.set_indices(keys)
        
    def set_indices(self, keys):
        self.indices = {k:[] for k in self.unique_keys}
        for i, k in enumerate(keys):
            self.indices[k].append(i)
            
    def apply(self, values, function):
        result = np.zeros(len(values))
        for k in self.unique_keys:
            result[self.indices[k]] = function(values[self.indices[k]])
        return result


def estimate_var_epsilon(data):
    data = data[data['var'].notnull()]
    var_epsilon_hat = np.dot(data['var'].values, data['size'].values)/np.sum(data['size'])
    assert var_epsilon_hat > 0
    return var_epsilon_hat

def residualize(df, y_var, x_vars, first_group, other_groups):
    y = df[y_var].values
    z = df[x_vars].values
    categorical_data = df[[first_group] + other_groups].values
    beta, fes = estimate_coefficients(y, z, categorical_data)
    residual = y - z @ beta - np.sum(fes[:, 1:], axis=1)
    # need to demean residual because FE's are only identified up to a constant
    return residual - np.mean(residual), beta


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


def drop_one_class_teachers(class_df):
    grouped = class_df.groupby('teacher')['class id']
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
    
    for i in range(0, nbins):
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
    standardized_error_se = (np.var(standardized_errors) / len(standardized_errors))**.5
    assert mean_error > -3 * se
    assert mean_error < 3 * se
    assert mean_standardized_error > 1 - 2 * standardized_error_se 
    assert mean_standardized_error < 1 + 2 * standardized_error_se 
    
def get_va(df, var_theta_hat, var_epsilon_hat, var_mu_hat, jackknife):
    array = df.values
    precisions = np.array([1 / (var_theta_hat + var_epsilon_hat / class_size) 
                          for class_size in array[:, 0]])
    numerators = precisions * array[:, 1]

    precision_sum = np.sum(precisions)
    num_sum = np.sum(numerators)
    # TODO: also return unshrunk va and variance
    if jackknife:
        return [(num_sum - n) / (precision_sum - p + 1 / var_mu_hat)
                for n, p in zip(numerators, precisions)]
    else:
        return num_sum / (precision_sum + 1 / var_mu_hat)
        
def get_bootstrap_sample(myList):
    indices = np.random.choice(range(len(myList)), len(myList))
    return myList[indices]

## Functions for high-dimensional fixed effects
def get_beta(y, z_projection, fixed_effects):
    residual = y - np.sum(fixed_effects, axis=1)
    return z_projection @ residual 
   
def get_fes(y, fixed_effects, index, grouped):
    use_fes = list(range(0, index)) + list(range(index + 1, fixed_effects.shape[1]))
    residual =  y - np.sum(fixed_effects[:, use_fes], axis=1)

    return grouped.apply(residual, lambda x: np.mean(x))
    
def estimate_coefficients(y, z, categorical_data):
    z_projection = np.linalg.inv(z.T @ z) @ z.T
    n, num_fes = categorical_data.shape
    
    # set up data structures
    fixed_effects = np.zeros((n, num_fes))
    grouped = [Groupby(categorical_data[:, i]) for i in range(num_fes)]

    # initialize fixed effects
    for j in range(num_fes):
        fixed_effects[:, j] = get_fes(y, fixed_effects, j, grouped[j])  
    # initialize beta
    beta = get_beta(y, z_projection, fixed_effects)
    
    # needed for loop
    beta_resid = y - z @ beta
    ssr_initial = np.sum((beta_resid - np.sum(fixed_effects, axis=1))**2)
    current_ssr = ssr_initial
    last_ssr = ssr_initial * 10
    
    while (last_ssr - current_ssr) / ssr_initial > 10**(-6):
        # first update fixed effects
        for j in range(num_fes):
            fixed_effects[:, j] = get_fes(beta_resid, fixed_effects, j, grouped[j])          
        # then update beta
        beta = get_beta(y, z_projection, fixed_effects)
        
        beta_resid = y - z @ beta
        last_ssr = current_ssr
        current_ssr = np.sum((beta_resid - np.sum(fixed_effects, axis=1))**2)

    return beta, fixed_effects
