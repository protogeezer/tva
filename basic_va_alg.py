from va_functions import *
from functools import reduce
import pandas as pd
import warnings
import numpy as np
import scipy.linalg
from variance_ls_numopt import get_g_and_tau
from mle import MLE
from config_tva import hdfe_dir
import sys
sys.path += [hdfe_dir]
from hdfe import Groupby, estimate


def invert_block_matrix(A: np.ndarray, B: np.ndarray, D: np.ndarray):
    """
    Inverts block matrix [[A, B], [B.T, D]]
    :param A: One dimensional, representing diagonal matrix
    :param B: matrix
    :param D: matrix
    :return:
    """
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
            val += vector[i:].T.dot(vector[:-i])

        return np.array([val, len(vector) * (len(vector) - 1) / 2])

    # First column is sum of all products, by teacher; second is number of products, by teacher
    mu_estimates = Groupby(data[teacher].values).apply(f, data['mean score'].values,
                                                       width=2)
    return np.sum(mu_estimates[:, 0]) / np.sum(mu_estimates[:, 1])


def get_each_va(df, var_theta_hat, var_epsilon_hat, var_mu_hat, jackknife, teacher):
    grouped = Groupby(df[teacher].values)

    # Get unshrunk VA
    def f(data): return get_unshrunk_va(data, jackknife)

    df['unshrunk va'] = grouped.apply(f, df[['size', 'mean score']].values, broadcast=True, width=1)
    if var_mu_hat > 0:

        def f(data): return get_va(data, var_theta_hat, var_epsilon_hat,
                                   var_mu_hat, jackknife)

        results = df.groupby(teacher)[['size', 'mean score']].apply(f).values

        if not jackknife:  # collapse to teacher level
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
    if not moments_only and jackknife:
        raise NotImplementedError('jackknife must be False')

    # TODO: Do teachers need to be coded as dense?
    # n_teachers = max(data[teacher]) + 1
    n_teachers = len(set(data[teacher]))
    if teacher_controls is not None:
        warnings.warn('Can\'t handle teacher-level controls')
    else:
        teacher_controls = np.ones((n_teachers, 1))

    cat = [teacher] if categorical_controls is None \
        else [teacher] + categorical_controls
    b, _, _, V = estimate(data, data[outcome].values, dense_controls, cat, 
                          estimate_variance=True, check_rank=True, cluster=class_level_vars)

    mu_preliminary = b[:n_teachers]
    # mu_preliminary -= np.mean(mu_preliminary)
    b_hat = b[n_teachers:]

    try:
        sigma_mu_squared, beta, gamma = get_g_and_tau(mu_preliminary, b_hat, V,
                                                      teacher_controls,
                                                      starting_guess=0)
    except np.linalg.LinAlgError:
        return {'sigma mu squared': 0, 'beta': None, 'gamma': None}
    if moments_only:
        return {'sigma mu squared': sigma_mu_squared, 'beta': beta, 
                'gamma': gamma}
    # TODO: this may have already been computed in 'estimate'; fix if time-consuming
    inv_V = np.linalg.inv(V)

    # epsilon = -1 * np.linalg.lstsq(inv_V[:n_teachers, :n_teachers], V[:n_teachers, n_teachers:])[0].dot(b_hat - beta)
    # tmp_1 = inv_V[:n_teachers, :n_teachers] + np.eye(n_teachers) / sigma_mu_squared
    # tmp_2 = inv_V[:n_teachers, :n_teachers].dot(mu_preliminary - epsilon) \
    #     + teacher_controls.dot(gamma) / sigma_mu_squared
    # ans = np.linalg.lstsq(tmp_1, tmp_2)[0]
    m = np.mean(mu_preliminary)
    alternate = m + np.linalg.lstsq(np.eye(n_teachers) + inv_V[:n_teachers, :n_teachers],
                                    mu_preliminary - m)[0]
    print('shape', alternate.shape)
    assert len(alternate) == n_teachers
    return {'individual effects': alternate, 'sigma mu squared': sigma_mu_squared,
            'beta': beta, 'gamma': gamma}


def moment_matching_alg(data, outcome, teacher, dense_controls, class_level_vars,
                        categorical_controls, jackknife, moments_only, method):

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
    
    if jackknife:  # Drop teachers teaching only one class
        keeps = Groupby(class_df[teacher]).apply(lambda elt: len(elt) > 1,
                                                 class_df[teacher]).astype(bool)
        class_df = class_df.loc[keeps, :].reset_index(drop=True)

    # Second, calculate a bunch of moments
    var_epsilon_hat = estimate_var_epsilon(class_df)
    var_mu_hat = estimate_mu_variance(class_df, teacher)

    # Estimate variance of class-level shocks
    var_theta_hat = ssr - var_mu_hat - var_epsilon_hat
    if var_theta_hat < 0:
        warnings.warn('Var theta hat is negative. Measured to be ' +
                      str(var_theta_hat))
        var_theta_hat = 0
        
    if var_mu_hat <= 0:
        warnings.warn('Var mu hat is negative. Measured to be ' + str(var_mu_hat))
    if moments_only:
        return {'sigma mu squared': var_mu_hat, 
                'sigma theta squared': var_theta_hat, 
                'sigma epsilon squared': var_epsilon_hat}

    results = get_each_va(class_df, var_theta_hat, var_epsilon_hat,
                          var_mu_hat, jackknife, teacher)

    return {'individual effects': results, 'sigma mu squared': var_mu_hat, 
            'sigma theta squared': var_theta_hat, 
            'sigma epsilon squared': var_epsilon_hat}


def calculate_va(data: pd.DataFrame, outcome: str, teacher: str, covariates: list,
                 class_level_vars: list, categorical_controls=None, jackknife=False,
                 moments_only=True, method='ks', add_constant=False, teacher_controls=None):
    """

    :param data: Pandas DataFrame
    :param outcome: string with name of outcome column
    :param teacher: string with name of teacher column
    :param covariates: List of strings with names of covariate columns
    :param class_level_vars: List of string with names of class-level columns.
        For example, a class may be identified by a combination of a teacher
        and time period, or classroom id and time period.
    :param categorical_controls: List of strings: Names of columns containing
        categorical data that will be expanded into dummy variables
    :param jackknife: Whether to use leave-out estimator for individual effects
    :param moments_only: Whether to get individual effects
    :param method: 'ks' for method from Kane & Staiger (2008)
                   'cfr' to residualize in the presence of fixed effects, as in
                        Chetty, Friedman, and Rockoff (2014)
                   'fk' to use an estimator derived from Fessler and Kasy (2016)
                   'mle' for maximum likelihood estimation
    :param add_constant: Boolean. The only time a regression is run without fixed effects
                is when method='ks' and categorical_controls is None, so that is the only
                time when add_constant is relevent.
    :param teacher_controls:
    :return:
    """

    # Input checks
    assert outcome in data.columns
    assert teacher in data.columns
    if covariates is not None:
        if not set(covariates).issubset(set(data.columns)):
            missing = set(covariates) - set(covariates) & set(data.columns)
            raise ValueError('The following columns are in covariates but not in data.columns:'
                             + str(missing))
    if method != 'fk':
        if not set(class_level_vars).issubset(set(data.columns)):
            print('class level vars ', class_level_vars)
            print('are not in data.columns')

    # Preprocessing
    use_cols = [outcome, teacher]
    if covariates is not None:
        use_cols += covariates
    if class_level_vars is not None:
        use_cols += class_level_vars

    if categorical_controls is not None:
        use_cols += categorical_controls
        assert set(categorical_controls).issubset(set(data.columns))
            
    not_null = (pd.notnull(data[col]) for col in remove_duplicates(use_cols))
    not_null = reduce(lambda z, y: z & y, not_null)
    assert not_null.any()
    if not not_null.all():
        print('Dropping ', len(not_null) - sum(not_null), ' observations due to missing data')
        data = data[not_null]
        data = data[remove_duplicates(use_cols)]

    # Recode categorical variables using consecutive integers
    to_recode = [teacher] if categorical_controls is None \
        else [teacher] + categorical_controls
    for col in to_recode:
        if np.issubdtype(data[col].dtype, np.number) and len(set(data[col])) <= max(data[col]):
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
    key_to_recover_teachers, data.loc[:, teacher] = np.unique(data[teacher],
                                                              return_inverse=True)

    if method in ['ks', 'cfr']:
        # TODO: figure out if this is still necessary
        if teacher not in class_level_vars:
            class_level_vars.append(teacher)
        assert teacher in data.columns
        return moment_matching_alg(data, outcome, teacher, dense_controls,
                                   class_level_vars, categorical_controls,
                                   jackknife, moments_only, method)
    elif method == 'fk':
        return fk_alg(data, outcome, teacher, dense_controls,
                      class_level_vars, categorical_controls,
                      jackknife, moments_only, teacher_controls)
    elif method == 'mle':
        estimator = MLE(data, outcome, teacher, dense_controls, categorical_controls,
                        jackknife, class_level_vars, moments_only)
        estimator.fit()
        teacher_idx = estimator.teacher_grouped.first_occurrences
        return {'sigma mu squared': estimator.sigma_mu_squared,
                'sigma theta squared': estimator.sigma_theta_squared,
                'sigma epsilon squared': estimator.sigma_epsilon_squared, 'beta': estimator.beta,
                'lambda': estimator.lambda_, 'alpha': estimator.alpha,
                'predictable var': estimator.predictable_var, 'hessian': estimator.hessian,
                'total var': estimator.total_var, 'asymp var': estimator.asymp_var,
                'bias correction': estimator.bias_correction,
                'x bar bar': estimator.x_bar_bar_long[teacher_idx],
                'total var se': estimator.total_var_se,
                'sigma mu squared se': estimator.sigma_mu_squared_se,
                'individual effects': estimator.individual_scores}
    else:
        raise NotImplementedError('Only the methods ks, cfr, fk, and mle are currently implmented.')
