from va_functions import *#, remove_collinear_cols
from functools import reduce
import pandas as pd
import warnings
import numpy as np
import scipy.sparse as sps
import scipy.linalg
import sys
from config import *
sys.path += [hdfe_dir]
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


def get_each_va(df, var_theta_hat, var_epsilon_hat, var_mu_hat, jackknife, teacher):
    grouped = Groupby(df[teacher].values)
    # Get unshrunk VA
    f = lambda data: get_unshrunk_va(data, var_theta_hat, var_epsilon_hat 
                                   , jackknife)

    tmp = grouped.apply(f, df[['size', 'mean score']].values, broadcast=True, width=1)
    df['unshrunk va'] = tmp 
    #df['unshrunk va']  = df.groupby(teacher)[['size', 'mean score']]\
    #                       .apply(f).values
    if var_mu_hat > 0:
        f = lambda data: get_va(data, var_theta_hat, var_epsilon_hat, var_mu_hat
                              , jackknife)
        results = df.groupby(teacher)[['size', 'mean score']].apply(f).values

        if not jackknife: # collapse to teacher leel
            df = df.groupby(teacher).size().reset_index()

        df['va'], df['variance'] = zip(*results)

    return df


def fk_alg(data, outcome, teacher, dense_controls, class_level_vars,
        categorical_controls, jackknife, moments_only, teacher_controls):
    """
    This file implements a value-added estimator inspired by
    Fessler and Kasy 2016, "How to Use Economic Theory to Improve Estimators." Documentation available via pdf.
    I believe something similar has been used in a 
    different value-added paper.
    TODO: Teacher-level controls
    """
    n_teachers = max(data[teacher]) + 1
    n = len(data)
    ones = np.ones((n_teachers, 1))
    if teacher_controls is not None:
        teacher_controls = np.hstack((ones, data[teacher_controls].values))
    else:
        teacher_controls = ones

    cat = [teacher] if categorical_controls is None\
            else [teacher] + categorical_controls
    b, _, _, V = estimate(data, data[outcome].values, dense_controls, cat, 
                          estimate_variance=True, check_rank=True, cluster=class_level_vars)

    mu_preliminary = b[:n_teachers]
    #mu_preliminary -= np.mean(mu_preliminary)
    b_hat = b[n_teachers:]

    sigma_mu_squared, beta = get_g_and_tau(mu_preliminary, b_hat, V, teacher_controls,
                                            starting_guess = 0)
    if moments_only:
        return sigma_mu_squared, beta
    # this may have already been computed in 'estimate'; fix if time-consuming
    inv_V = np.linalg.inv(V)
    m = np.mean(mu_preliminary)

    epsilon = np.linalg.lstsq(inv_V[:n_teachers, :n_teachers], V[:n_teachers, n_teachers:])[0].dot(b_hat - beta)
    tmp_1 = inv_V[:n_teachers, :n_teachers] + np.eye(n_teachers) / sigma_mu_squared
    tmp_2 = inv_V[:n_teachers, :n_teachers] * (mu_preliminary - epsilon) + m / sigma_mu_squared
    ans = np.linalg.lstsq(tmp_1, tmp_2)[0]

    alternate = m + np.linalg.lstsq(np.eye(n_teachers) + inv_V[:n_teachers, :n_teachers], mu_preliminary - m)
    return alternate, sigma_mu_squared


# TODO: this is the old version
def mle(data, outcome, teacher, dense_controls, categorical_controls, jackknife,
        moments_only):
    def demean(mat):
        return mat - np.mean(mat, 0)
    def mean(mat):
        return np.mean(mat, 0)

    grouped = Groupby(data[teacher])
    # since using within estimator, just make everything dense
    if categorical_controls is None:
        x = dense_controls
    else:
        x = np.hstack((dense_controls, 
                sps.hstack((make_dummies(data[elt]) for elt in categorical_controls)).A))
    x_means = grouped.apply(mean, x, width = x.shape[1], broadcast=False)
    y_means = grouped.apply(mean, data[outcome].values, broadcast=False)
    beta_plus_lambda = np.linalg.lstsq(x_means, y_means)[0]
    errors_2 = y_means - x_means.dot(beta_plus_lambda)
    ssr_2 = errors_2.dot(errors_2)
    x_demeaned = grouped.apply(demean, x, broadcast=True)
    y_demeaned = grouped.apply(demean, data[outcome].values, broadcast=True)
    beta = np.linalg.lstsq(x_demeaned, y_demeaned)[0]
    errors_1 = y_demeaned - x_demeaned.dot(beta)
    ssr_1 = errors_1.dot(errors_1)
    n_teachers = len(set(data[teacher]))
    assert n_teachers == len(y_means)
    # sigma_mu_^2 + sigma_e^2/K
    tmp = ssr_2 / n_teachers
    sigma_epsilon_sq = ssr_1 / (len(data) - n_teachers)
    # assume number of students per class is constant
    sigma_mu_sq = tmp - sigma_epsilon_sq / (len(data) / n_teachers)
    # mu ~ N(x_means * lambda, sigma_mu_sq)
    lambda_ = beta_plus_lambda - beta
    total_variance = np.var(x_means.dot(lambda_)) + sigma_mu_sq
    return total_variance, sigma_epsilon_sq
    

def moment_matching_alg(data, outcome, teacher, dense_controls, class_level_vars,
        categorical_controls, jackknife, moments_only, method, add_constant):

    n = len(data)

    def demean(mat):
        return mat - np.mean(mat, 0)

    # If method is 'ks', just ignore teachers when residualizing
    if method == 'ks':
        beta, x, residual = estimate(data, data[outcome].values, dense_controls, 
                                  categorical_controls, get_residual=True,
                                  check_rank=True)
    # Residualize with fixed effects
    else:
        n_teachers = len(set(data[teacher]))
        cat = [teacher] if categorical_controls is None\
                else [teacher] + categorical_controls
        beta, x = estimate(data, data[outcome].values, dense_controls, cat,
                           check_rank=True)
        # add teacher fixed effects back in
        try:
            x = x.A
        except AttributeError:
            pass
        residual = data[outcome].values - x[:, n_teachers:].dot(beta[n_teachers:])
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
                        , var_mu_hat, jackknife, teacher)

    return results, var_mu_hat, var_theta_hat, var_epsilon_hat


def calculate_va(data, outcome, teacher, covariates, class_level_vars,
                categorical_controls=None, jackknife=False, moments_only=True, 
                method='ks', add_constant=False, teacher_controls=None):
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
    if method != 'fk':
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
    if add_constant:
        assert method == 'ks'
        if dense_controls is None:
            dense_controls = np.ones((len(data), 1))
        else:
            dense_controls = np.hstack((dense_controls, np.ones((len(data), 1))))

    # Recode categorical variables
    # Recode teachers as contiguous integers
    key_to_recover_teachers, data.loc[:, teacher] = \
            np.unique(data[teacher], return_inverse=True)

    if method in ['ks', 'cfr']:
        return moment_matching_alg(data, outcome, teacher, dense_controls, class_level_vars,
                categorical_controls, jackknife, moments_only, method, add_constant)
    elif method == 'fk':
        return fk_alg(data, outcome, teacher, dense_controls,
                      class_level_vars, categorical_controls,
                      jackknife, moments_only, teacher_controls)
    elif method == 'mle':
        return mle(data, outcome, teacher, dense_controls, categorical_controls,
                   jackknife, moments_only)
    else:
        raise NotImplementedError('Only the methods ks, cfr, and fk are currently implmented.')

