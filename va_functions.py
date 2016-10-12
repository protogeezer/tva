import numpy as np
import numpy.linalg as linalg
import pandas as pd
import scipy.sparse as sparse


class Groupby:
    def __init__(self, keys, already_dense = False):
        if already_dense:
            self.keys_as_int = keys
        else:
            _, self.keys_as_int = np.unique(keys, return_inverse = True)
        self.n_keys = max(self.keys_as_int)
        self.set_indices()

    def set_indices(self):
        self.indices = [[] for i in range(self.n_keys + 1)]
        for i, k in enumerate(self.keys_as_int):
            self.indices[k].append(i)
        self.indices = [np.array(elt) for elt in self.indices]

    def apply(self, function, vector):
        result = np.zeros(len(vector))
        for k in range(self.n_keys):
            result[self.indices[k]] = function(vector[self.indices[k]])
        return result

def find_collinear(matrix, tol):
    _, r = np.linalg.qr(matrix)
    # find collinear columns by looking at diagonals of r
    n, k = matrix.shape
    collinear_cols = [False for i in range(k)]
    row_idx = 0
    col_idx = 0
    for col_idx in range(k):
        collinear = abs(r[row_idx, col_idx]) < tol
        if collinear:
            collinear_cols[col_idx] = True
        else:
            row_idx = row_idx + 1

    return collinear_cols

# create lags
def make_lags(df, n_lags_back, n_lags_forward, outcomes, groupby, fill_zeros=True):
    lags = list(range(-1 * n_lags_forward, 0)) + list(range(1, n_lags_back+1))
    # First sort
    grouped = Groupby(groupby)

    for out in outcomes:
        for lag in lags:
            df[out + '_lag_' + str(lag)] = grouped.apply(lambda x: x.shift(lag), df[out].values)

    lag_vars = {out: [out + '_lag_' + str(lag) for lag in lags]
                for out in outcomes}

    if fill_zeros:
        for out in outcomes:
            for lag_var in lag_vars[out]:
                missing = pd.isnull(df[lag_var])
                df[lag_var + '_mi'] = missing.astype(int)
                df.loc[missing, lag_var] = 0
            lag_vars[out] = lag_vars[out] + [out + '_lag_' + str(lag) + '_mi']    

    return df, lag_vars


def estimate_var_epsilon(data):
    data = data[data['var'].notnull()]
    var_epsilon_hat = np.dot(data['var'].values, data['size'].values)/np.sum(data['size'])
    assert var_epsilon_hat > 0
    return var_epsilon_hat
    
# Demeaning with one fixed effect
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
        x = df[x_names].values
        x_demeaned = fe_demean(df, x_names, first_group)
        
        if second_groups is not None: # If extra FE, create dummies
            dummy_df = pd.get_dummies(df, columns=second_groups)
            dummy_cols = []
            for col in second_groups:
                dummy_cols += [elt for elt in dummy_df.columns if col in elt]
                
            dummies = dummy_df[dummy_cols].values
                                
            if len(x_names) > 0:
                x = np.hstack((x, dummies))
                x_demeaned = np.hstack((x_demeaned, dummies))
            else:
                x = dummies
                x_demeaned = dummies

        beta = linalg.lstsq(x_demeaned, y)[0]
        resid = y - np.dot(x, beta)
        return resid - np.mean(resid), beta
                
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
    y_5 = np.zeros(nbins)
    y_95 = np.zeros(nbins)
    
    for i in range(0, nbins):
        start = len(x) * i / nbins
        end = len(x) * (i+1) / nbins
        bins[i] = np.mean(x[int(start):int(end)])
        y_means[i] = np.mean(y[int(start):int(end)])
        y_medians[i] = np.median(y[int(start):int(end)])
        y_5[i] = np.percentile(y[int(start):int(end)], 5)
        y_95[i] = np.percentile(y[int(start):int(end)], 95)

    return bins, y_means, y_medians, y_5, y_95


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
