from va_functions import remove_duplicates, estimate_var_epsilon, remove_collinear_cols
from functools import reduce
import pandas as pd
import warnings
import numpy as np
import scipy.sparse as sps
import scipy.linalg
import sys
#sys.path += ['/n/home09/esantorella/Bureaucrat-Value-Added/']
#from groupby import Groupby
sys.path += ['/n/home09/esantorella/hdfe/']
from hdfe import Groupby
from variance_ls_numopt import get_g_and_tau


# A should be a vector, rest are matrices
# assume symmetric and positive definite: C = B;
# This works, yay
def invert_block_matrix(A, B, D):
    import ipdb; ipdb.set_trace()
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


def make_dummies(elt, drop_col):
    if np.max(elt) >= len(set(elt)):
        _, elt = np.unique(elt, return_inverse=True)

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
        # Within estimator
        grouped = Groupby(data[teacher].values)
        x_demeaned = grouped.apply(demean, dense_controls)
        b = np.linalg.lstsq(x_demeaned, data[outcome].values)[0]
        residual = data[outcome].values - dense_controls.dot(b)
        teacher_effects = grouped.apply(np.mean, residuals)
        b = np.concatenate((teacher_effects, b))
        x = sps.hstack((techer_dummies, dense_controls))
    else:
        if len(categorical_controls) == 1:
            non_teacher_dummies = make_dummies(data[categorical_controls[0]], True)
        else:
            non_teacher_dummies = sps.hstack([make_dummies(data[elt].values, True)
                                              for elt in categorical_controls])
        if dense_controls is None:
            other_controls = non_teacher_dummies
        else:
            other_controls = sps.hstack((non_teacher_dummies, dense_controls))
        x = sps.hstack((teacher_dummies, other_controls))
        rank = np.linalg.matrix_rank(x.todense())
        print('Rank of x before ', rank)
        print('Shape of x before ', x.shape)
        if rank < x.shape[1]:
            warnings.warn('x is rank deficient')
            x = remove_collinear_cols(x)
#            assert np.linalg.matrix_rank(x.todense()) == x.shape[1]

        b = sps.linalg.lsqr(x, data[outcome].values)[0]

    assert np.all(np.isfinite(b))

    errors = data[outcome].values - x.dot(b)

    # Get variance matrix: first method
    q, r = np.linalg.qr(x.todense())
    # will fail if x is rank deficient
    inv_r = scipy.linalg.solve_triangular(r, np.eye(r.shape[0]))
    inv_x_prime_x = inv_r.dot(inv_r.T)
    V = inv_x_prime_x * errors.dot(errors) / (n - x.shape[1])

    mu_preliminary = b[:n_teachers]
    mu_preliminary -= np.mean(mu_preliminary)

    g_hat = b[n_teachers:]
    assert len(mu_preliminary) == n_teachers

    sigma_mu_squared, gamma = get_g_and_tau(mu_preliminary, g_hat, V, starting_guess = .01 * np.var(data[outcome]))
    return sigma_mu_squared, gamma
    

def moment_matching_alg(data, outcome, teacher, dense_controls, class_level_vars,
        categorical_controls, jackknife, moments_only, method, add_constant):

    n = len(data)

    def demean(mat):
        return mat - np.mean(mat, 0)

    # Use within estimator if method='ks' and len(categorical_controls) = 1
    # Use within estimator if method='cfr' and len(categorical_controls) = 0
    # Otherwise, don't use within estimator

    # If method is 'ks', just ignore teachers when residualizing
    if method == 'ks':
        # Within estimator
        if categorical_controls is None:
            if add_constant:
                if dense_controls is None:
                    dense_controls = np.ones((n, 1))
                else:
                    dense_controls = np.hstack((np.ones((n, 1)), dense_controls))

            beta = np.linalg.lstsq(dense_controls, data[outcome].values)[0]
            assert np.all(np.isfinite(beta))
            residual = data[outcome] - dense_controls.dot(beta)

        elif len(categorical_controls) == 1:
            grouped = Groupby(data[categorical_controls[0]].values)
            if dense_controls is None:
                residual = grouped.apply(demean, data[outcome].values)
            else:
                x_demeaned = grouped.apply(demean, dense_controls)
                if x_demeaned.ndim == 1:
                    x_demeaned = np.expand_dims(x_demeaned, 1)
                beta_tmp = np.linalg.lstsq(x_demeaned, data[outcome].values)[0]
                assert np.all(np.isfinite(beta_tmp))
                resid_tmp = data[outcome].values - dense_controls.dot(beta_tmp)
                assert np.all(np.isfinite(resid_tmp))
                residual = grouped.apply(demean, resid_tmp)
        else:
            dummies = [make_dummies(data[elt], drop_col=True) for elt in categorical_controls]
            if dense_controls is None:
                controls = sps.hstack((np.ones((n, 1)), sps.hstack(dummies)))
            else:
                controls = sps.hstack((np.ones((n, 1)), sps.hstack(dummies), dense_controls))
            # Or is this supposed to have [0] at the end?
            beta = sps.linalg.lsqr(controls, data[outcome].values)[0]
            assert np.all(np.isfinite(beta))
            residual = data[outcome] - controls.dot(beta)
    # Residualize with fixed effects
    else:
        if categorical_controls is None:
            grouped = Groupby(data[teacher].values)
            x_demeaned = grouped.apply(demean, dense_controls)
            beta = np.linalg.lstsq(x_demeaned, data[outcome].values)[0]
            residual = data[outcome].values - dense_controls.dot(beta)
            residual -= np.mean(residual)
            
        else:
            teacher_dummies = make_dummies(data[teacher], drop_col=False)
            n_teachers = len(set(data[teacher]))

            if len(categorical_controls) == 1:
                non_teacher_dummies = make_dummies(data[categorical_controls[0]], drop_col=True)
            else:
                non_teacher_dummies = sps.hstack([make_dummies(data[elt], True) 
                                                  for elt in categorical_controls])

            if dense_controls is None:
                other_controls = non_teacher_dummies
            else:
                other_controls = sps.hstack((non_teacher_dummies, dense_controls))
            controls = sps.hstack((teacher_dummies, other_controls))
            beta = sps.linalg.lsqr(controls, data[outcome].values)[0]
            residual = data[outcome].values - other_controls.dot(beta[n_teachers:])

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
        for elt in categorical_controls:
            _, data[elt] = np.unique(data[elt], return_inverse=True)
        # make sure each category is coded as dense

    not_null = (pd.notnull(data[x]) for x in remove_duplicates(use_cols))
    not_null = reduce(lambda x, y: x & y, not_null)
    assert not_null.any()
    print('Dropping ', len(not_null) - sum(not_null), ' observations due to missing data')
    data = data[not_null]
    data = data[remove_duplicates(use_cols)]

    for col in ['person'] + categorical_controls:
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

