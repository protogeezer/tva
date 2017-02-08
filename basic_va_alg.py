from va_functions import remove_duplicates, estimate_var_epsilon
from functools import reduce
import pandas as pd
import warnings
import numpy as np
import scipy.sparse as sps
import sys
sys.path += ['/Users/lizs/hdfe/']
from hdfe import Groupby
from variance_ls_numopt import get_ll, get_grad, get_hessian
from scipy.optimize import minimize

# TODO: make sure things work correctly with no covariates


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


def make_dummies(elt, drop_col):
    dummies = sps.csc_matrix((np.ones(len(elt)), (range(len(elt)), elt)))
    if drop_col:
        return dummies[:, 1:]
    else:
        return dummies


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

    teacher_dummies = make_dummies(data[teacher], drop_col=False)
    if categorical_controls is None:
        dummies = teacher_dummies
    elif len(categorical_controls) == 1:
        dummies = sps.hstack((teacher_dummies, make_dummies(data[categorical_controls[0]], True)))
    else:
        dummies = sps.hstack((teacher_dummies,
            sps.hstack((make_dummies(data[elt], True) for elt in categorical_controls))))

    x = sps.hstack((dummies, dense_controls))
    b = sps.linalg.lsqr(x, data[outcome].values)[0]
    errors = data[outcome] - x.dot(b)
    # x'x as vector rather than diagonal matrix
    # Works because x is a matrix of dummy variables -- only one nonzero element per row,
    # and all nonzero elements are ones
    x_prime_x = teacher_dummies.T.dot(np.ones(dummies.shape[0]))
    print('Min and max of x_prime_x ', min(x_prime_x), max(x_prime_x))
    assert len(x_prime_x) == n_teachers
    V = errors.dot(errors) / (x_prime_x * (n - x.shape[1]))
    assert V.ndim == 1
    assert len(V) == n_teachers
    print('Min and max of V ', min(V), max(V))
    mu_preliminary = b[:n_teachers]
    assert len(mu_preliminary) == n_teachers
    assert len(mu_preliminary) == len(V)

    objfun = lambda x: get_ll(x, mu_preliminary, V)
    g = lambda x: get_grad(x, mu_preliminary, V)
    h = lambda x: get_hessian(x, mu_preliminary, V)

    print('About to optimization')
    sigma_mu_squared = minimize(objfun, np.min(V), method='Newton-CG', jac=g, hess=h).x
    return sigma_mu_squared
    

def moment_matching_alg(data, outcome, teacher, dense_controls, class_level_vars,
        categorical_controls, jackknife, moments_only, method, add_constant):

    n = len(data)

    def demean(mat):
        return mat - np.mean(mat, 0)

    # Use within estimator if method='ks' and len(categorical_controls) = 1
    # Use within estimator if method='cfr' and len(categorical_controls) = 0
    # Otherwise, don't use within estimator

    # If method is 'ks', just ignore teachers
    if method == 'ks':
        # Within estimator
        if len(categorical_controls) == 1:
            grouped = Groupby(data[categorical_controls[0]].values)
            x_demeaned = grouped.apply(demean, dense_controls)
            if x_demeaned.ndim == 1:
                x_demeaned.shape = (n, 1)
            beta_tmp = np.linalg.lstsq(x_demeaned, data[outcome].values)[0]
            print(beta_tmp)
            resid_tmp = data[outcome] - dense_controls.dot(beta_tmp)
            residual = grouped.apply(demean, resid_tmp)
            assert(len(residual) == len(data))
        elif len(categorical_controls) == 0:
            if add_constant:
                dense_controls = np.hstack((np.ones(n, 1), dense_controls))

            beta = np.linalg.lstsq(dense_controls, data[outcome].values)[0]
            residual = data[outcome] - dense_controls.dot(beta)
        else:
            dummies = (make_dummies(data[elt], drop_col=True) for elt in categorical_controls)
            controls = sps.hstack((np.ones(n, 1), sps.hstack(dummies), dense_controls))
            beta = sps.linalg.lsqr(controls, data[outcome].values)
            residual = data[outcome] - controls.dot(beta)
    # Residualize with fixed effects
    else:
        if categorical_controls is None:
            grouped = Groupby(data[teacher].values)
            x_demeaned = grouped.apply(demean, dense_controls)
            beta = np.linalg.lstsq(x_demeaned, data[outcome].values)
            residual = data[outcome] - x_demeaned.dot(beta)
        else:
            teacher_dummies = make_dummies(data[teacher], drop_col=False)
            n_teachers = len(set(data[teacher]))

            if len(categorical_controls) == 1:
                non_teacher_dummies = make_dummies(data[categorical_controls[0]], drop_col=True)
            else:
                non_teacher_dummies = sps.hstack((make_dummies(data[elt], True) 
                                                  for elt in categorical_controls))

            other_controls = sps.hstack((non_teacher_dummies, dense_controls))

            controls = sps.hstack((teacher_dummies, other_controls))
            beta = sps.linalg.lsqr(controls, data[outcome].values)[0]
            print(beta)
            residual = data[outcome].values - other_controls.dot(beta[n_teachers:])

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
        keeps = Groupby(class_df[teacher].values).apply(lambda x: len(x) > 1, class_df[teacher])
        class_df = class_df.loc[keeps, :]
        class_df.reset_index(drop=True, inplace=True)

    ## Second, calculate a bunch of moments
    # Calculate variance of epsilon
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
    if column_names is not None:
        results.rename(columns={column_names[key]:key 
                                for key in column_names}, inplace=True)
    
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
    if covariates is None:
        covariates = []
    else:
        assert set(covariates).issubset(set(data.columns))
    assert set(class_level_vars).issubset(set(data.columns))
    if categorical_controls is None:
        categorical_controls = []
    else:
        assert set(categorical_controls).issubset(set(data.columns))

    # Preprocessing
    use_cols = remove_duplicates([outcome, teacher] + covariates + \
                                  class_level_vars + categorical_controls)
    not_null = (pd.notnull(data[x]) for x in  use_cols)
    not_null = reduce(lambda x, y: x & y, not_null)
    assert not_null.any()
    print('Dropping ', sum(not_null), ' observations due to missing data')
    data = data[not_null]

    dense_controls = data[covariates].values

    # Recode categorical variables
    # Recode teachers as contiguous integers
    key_to_recover_teachers, data[teacher] = np.unique(data[teacher], return_inverse=True)
    for elt in categorical_controls:
        _, data[elt] = np.unique(data[elt], return_inverse=True)

    if method in ['ks', 'cfr']:
        return moment_matching_alg(data, outcome, teacher, dense_controls, class_level_vars,
                categorical_controls, jackknife, moments_only, method, add_constant)
    elif method == 'fk':
        return fk_alg(data, outcome, teacher, dense_controls, categorical_controls,
                      jackknife, moments_only)
    else:
        raise NotImplementedError('Only the methods ks, cfr, and fk are currently implmented.')

