from va_functions import remove_duplicates, estimate_var_epsilon#, remove_collinear_cols
from functools import reduce
import pandas as pd
import warnings
import numpy as np
import scipy.sparse as sps
import scipy.linalg
import sys
sys.path += ['/n/home09/esantorella/hdfe/']
from hdfe import Groupby, estimate
from variance_ls_numopt import get_g_and_tau


# A should be a vector, rest are matrices
# assume symmetric and positive definite: C = B;
# This works, yay
def invert_block_matrix(A, B, D):
    A_inv = 1 / A
    A_inv_B = (A_inv * B.T).T

    cho = scipy.linalg.cho_factor(D - B.T.dot(A_inv_B))
    tmp = scipy.linalg.cho_solve(cho, np.eye(D.shape[0]))

    first = np.diag(A_inv) + A_inv_B.dot(tmp).dot(A_inv_B.T)
    second = -1 * A_inv_B.dot(tmp)
    return np.vstack((np.hstack((first, second)),
                      np.hstack((second.T, tmp))))


def estimate_mu_variance(data, teacher):
    def f(vector):
        val = 0
        for i in range(1, len(vector)):
            val += np.dot(vector[i:], vector[:-i])

        return np.array([val, len(vector) * (len(vector) - 1) /2])

    # First column is sum of all products, by teacher; second is number of products, by teacher
    mu_estimates = Groupby(data[teacher].values)\
                    .apply(f, data['mean score'].values, width=2)
    return np.sum(mu_estimates[:, 0]) / np.sum(mu_estimates[:, 1])


def get_each_va(df, var_theta_hat, var_epsilon_hat, var_mu_hat, jackknife):
    # Get unshrunk VA
    f = lambda data: get_unshrunk_va(data, var_theta_hat, var_epsilon_hat 
                                   , jackknife)
    df['unshrunk va']  = df.groupby('teacher')[['size', 'mean score']]\
                           .apply(f).values
    if var_mu_hat > 0:
        f = lambda data: get_va(data, var_theta_hat, var_epsilon_hat, var_mu_hat
                              , jackknife)
        results = df.groupby('teacher')[['size', 'mean score']].apply(f).values

        if not jackknife: # collapse to teacher leel
            df = df.groupby('teacher').size().reset_index()

        df['va'], df['variance'] = zip(*results)

    return df



def fk_alg(data, outcome, teacher, dense_controls, 
        categorical_controls, jackknife, moments_only):
    """
    This file implements a value-added estimator inspired by
    Fessler and Kasy 2016, "How to Use Economic Theory to Improve Estimators." Documentation available via pdf.
    I believe something similar has been used in a 
    different value-added paper.
    TODO: Teacher-level controls
    """
    if not moments_only:
        raise NotImplementedError('Please set moments_only to True')

    n_teachers = max(data[teacher]) + 1
    n = len(data)

    cat = [teacher] if categorical_controls is None\
            else [teacher] + categorical_controls
    b, _, _, V = estimate(data, data[outcome].values, dense_controls, cat, estimate_variance=True)

    mu_preliminary = b[:n_teachers]
    mu_preliminary -= np.mean(mu_preliminary)

    g_hat = b[n_teachers:]
    assert len(mu_preliminary) == n_teachers

    sigma_mu_squared, gamma = get_g_and_tau(mu_preliminary, g_hat, V, 
                                            starting_guess = 0)
    return sigma_mu_squared, gamma
    

def moment_matching_alg(data, outcome, teacher, dense_controls, class_level_vars,
        categorical_controls, jackknife, moments_only, method, add_constant):

    n = len(data)

    def demean(mat):
        return mat - np.mean(mat, 0)

        # If method is 'ks', just ignore teachers when residualizing
    if method == 'ks':
        beta, x, residual = estimate(data, data[outcome].values, dense_controls, 
                                  categorical_controls, get_residual=True)
    # Residualize with fixed effects
    else:
        n_teachers = len(set(data[teacher]))
        cat = [teacher] if categorical_controls is None\
                else [teacher] + categorical_controls
        beta, x = estimate(data, data[outcome].values, dense_controls, cat)
        # add teacher fixed effects back in
        residual = data[outcome].values - x.A[:, n_teachers:].dot(beta[n_teachers:])
        residual -= np.mean(residual)
        
    assert np.all(np.isfinite(residual))
    assert len(residual) == len(data)
    data['residual'] = residual
    ssr = np.var(residual)

    # Collapse data to class level
    # Count number of students in class

    class_df = data.groupby(class_level_vars).size().reset_index()
    class_df.columns = class_level_vars + ['size']

    # Calculate mean and merge it back into class-level data
    class_df.loc[:, 'mean score'] = \
                        data.groupby(class_level_vars)['residual'].mean().values
    class_df.loc[:, 'var'] = \
                         data.groupby(class_level_vars)['residual'].var().values
    assert len(class_df) > 0
    
    if jackknife: # Drop teachers teaching only one class
        keeps = Groupby(class_df[teacher]).apply(lambda x: len(x) > 1, class_df[teacher]).astype(bool)
        class_df = class_df.loc[keeps, :].reset_index(drop=True)

    # Second, calculate a bunch of moments
    var_epsilon_hat = estimate_var_epsilon(class_df)
    var_mu_hat = estimate_mu_variance(class_df, teacher)


    # Estimate variance of class-level shocks
    var_theta_hat = ssr - var_mu_hat - var_epsilon_hat
    if var_theta_hat < 0:
        warnings.warn('Var theta hat is negative. Measured to be ' +\
                       str(var_theta_hat))
        var_theta_hat = 0
        
    if var_mu_hat <= 0:
        warnings.warn('Var mu hat is negative. Measured to be '+ str(var_mu_hat))
    if moments_only:
        return var_mu_hat, var_theta_hat, var_epsilon_hat

    results = get_each_va(class_df, var_theta_hat, var_epsilon_hat
                        , var_mu_hat, jackknife)

    return results, var_mu_hat, var_theta_hat, var_epsilon_hat


def calculate_va(data, outcome, teacher, covariates, class_level_vars,
                categorical_controls=None, jackknife=False, moments_only=True, 
                method='ks', add_constant=False):
    """
    :param data: Pandas DataFrame
    :param outcome: string with name of outcome column
    :param teacher: string with name of teacher column
    :param covariates: List of strings with names of covariate columns
    :param class_level_vars: List of string with names of class-level columns.
        For example, a class may be identified by a combination of a teacher
        and time period, or classroom id and time period.
    :param categorical_controls: Controls that must be expanded into dummy variables.
    :param jackknife: Whether to use leave-out estimator
    :param method: 'ks' for method from Kane & Staiger (2008)
                   'cfr' to residualize in the presence of fixed effects, as in
                        Chetty, Friedman, and Rockoff (2014)
                    'fk' to use an estimator derived from Fessler and Kasy (2016)
    :param add_constant: Boolean. The only time a regression is run without fixed effects
                is when method='ks' and categorical_controls is None, so that is the only
                time when add_constant is relevent.
    """

    # Input checks
    assert outcome in data.columns
    assert teacher in data.columns
    if covariates is not None:
        assert set(covariates).issubset(set(data.columns))
    if class_level_vars is None:
        if method != 'fk':
            raise TypeError('class_level_vars must not be none with method ', method)
    else:
        assert set(class_level_vars).issubset(set(data.columns))

    # Preprocessing
    use_cols = [outcome, teacher]
    if covariates is not None:
        use_cols += covariates
    if class_level_vars is not None:
        use_cols += class_level_vars

    if categorical_controls is not None:
        use_cols += categorical_controls
        assert set(categorical_controls).issubset(set(data.columns))
            

    not_null = (pd.notnull(data[x]) for x in remove_duplicates(use_cols))
    not_null = reduce(lambda x, y: x & y, not_null)
    assert not_null.any()
    if not not_null.all():
        print('Dropping ', len(not_null) - sum(not_null), ' observations due to missing data')
        data = data[not_null]
        data = data[remove_duplicates(use_cols)]

    # Recode categorical variables using consecutive integers
    to_recode = [teacher] if categorical_controls is None \
                else [teacher] + categorical_controls
    for col in to_recode:
        if len(set(data[col])) <= max(data[col]):
            _, data[col] = np.unique(data[col], return_inverse=True)

        
    dense_controls = None if covariates is None else data[covariates].values

    # Recode categorical variables
    # Recode teachers as contiguous integers
    key_to_recover_teachers, data.loc[:, teacher] = \
            np.unique(data[teacher], return_inverse=True)

    if method in ['ks', 'cfr']:
        return moment_matching_alg(data, outcome, teacher, dense_controls, class_level_vars,
                categorical_controls, jackknife, moments_only, method, add_constant)
    elif method == 'fk':
        return fk_alg(data, outcome, teacher, dense_controls, categorical_controls,
                      jackknife, moments_only)
    else:
        raise NotImplementedError('Only the methods ks, cfr, and fk are currently implmented.')

