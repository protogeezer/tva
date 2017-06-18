from va_functions import *
from functools import reduce
import pandas as pd
import warnings
import numpy as np
import scipy.sparse as sps
import scipy.linalg
import sys
from config_tva import hdfe_dir
sys.path += [hdfe_dir]
from hdfe import Groupby, estimate, make_dummies
from multicollinearity import find_collinear_cols
from variance_ls_numopt import get_g_and_tau
from scipy.optimize import minimize
from variance_ls_numopt import newton
import ad
from ad.admath import *


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
            val += vector[i:].T.dot(vector[:-i])

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

    df['unshrunk va'] = grouped.apply(f, df[['size', 'mean score']].values, broadcast=True, width=1)
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

    sigma_mu_squared, beta, gamma = get_g_and_tau(mu_preliminary, b_hat, V, teacher_controls,
                                            starting_guess = 0)
    if moments_only:
        return sigma_mu_squared, beta, gamma
    ## TODO: this may have already been computed in 'estimate'; fix if time-consuming
    inv_V = np.linalg.inv(V)

    epsilon = -1 * np.linalg.lstsq(inv_V[:n_teachers, :n_teachers], V[:n_teachers, n_teachers:])[0].dot(b_hat - beta)
    tmp_1 = inv_V[:n_teachers, :n_teachers] + np.eye(n_teachers) / sigma_mu_squared
    tmp_2 = inv_V[:n_teachers, :n_teachers].dot(mu_preliminary - epsilon) + teacher_controls.dot(gamma) / sigma_mu_squared
    ans = np.linalg.lstsq(tmp_1, tmp_2)[0]
    return ans, sigma_mu_squared


def mle(data, outcome, teacher, dense_controls, categorical_controls,
        jackknife, class_level_vars, moments_only):
    #TODO: make sure there is not a constant in dense_controls
    if categorical_controls is None:
        x = dense_controls
    elif len(categorical_controls) == 1:
        x = np.hstack((dense_controls,
                       make_dummies(data[categorical_controls[0]], True).A))
    else:
        x = sps.hstack([make_dummies(data[elt], True) 
                        for elt in categorical_controls]).A
        if dense_controls is not None:
            x = np.hstack((dense_controls, x))

    assert len(class_level_vars) == 1
    class_grouped = Groupby(data[class_level_vars].values)
    assert class_grouped.already_sorted

    # Make sure everything a varies within teacher, so beta is identified
    x_with_teacher_dummies = np.hstack((make_dummies(data[teacher], False).A, 
                                        np.array(x)))
    collinear_cols, not_collinear_cols = find_collinear_cols(x_with_teacher_dummies)
    if len(collinear_cols) > 0:
        print('Found', len(collinear_cols), 'collinear columns in x.')
        x = x_with_teacher_dummies[:, not_collinear_cols][:, len(set(data[teacher])):]
    else:
        print('No collinear columns in x')
        
    y = data[outcome].values
    n_students_per_class = class_grouped.apply(len, y, broadcast=False)
    n_classes = len(class_grouped.first_occurrences)
    n_students = len(data)

    y_tilde = class_grouped.apply(lambda x: x - np.mean(x), y)
    x_tilde = class_grouped.apply(lambda x: x - np.mean(x, 0), x, 
                                  width = x.shape[1])

    x_bar = class_grouped.apply(lambda x: np.mean(x, 0), x, broadcast=False, 
                                width=x.shape[1])

    y_bar = class_grouped.apply(np.mean, y, broadcast=False)
    
    teachers = data[teacher].values[class_grouped.first_occurrences]
    # Should only be applied to objects that have been created with
    # class_grouped.apply
    teacher_grouped = Groupby(teachers)
    assert teacher_grouped.already_sorted
    n_teachers = len(teacher_grouped.first_occurrences)

    x_bar_bar_tmp = teacher_grouped.apply(lambda x: np.mean(x, 0), x_bar,
                                          width=x_bar.shape[1], broadcast=False)
    x_bar_bar_tmp = np.hstack((np.ones((n_teachers, 1)), x_bar_bar_tmp))
    # x bar may be rank deficient. If so, we will set all components of lambda
    # corresponding to not_collin_x_bar_bar to zero.
    collin_x_bar_bar, not_collin_x_bar_bar = find_collinear_cols(sps.csc_matrix(x_bar_bar_tmp))
    if len(collin_x_bar_bar) > 0:
        print('Found', len(collin_x_bar_bar), 'collinear columns in x bar bar')
        x_tilde = x_tilde[:, not_collin_x_bar_bar[1:] - 1]
        x_bar = x_bar[:, not_collin_x_bar_bar[1:] - 1]

    else:
        print('No collinear columns in x bar bar')

    xx_tilde = x_tilde.T.dot(x_tilde)
    xy_tilde = x_tilde.T.dot(y_tilde)[:, 0]
    assert xy_tilde.ndim == 1

    # Vectorize and encapsulate variances
    ll_vec_func = get_ll_vec_func(n_students_per_class, n_classes, n_students, 
                                  y_tilde, x_tilde, x_bar, y_bar, 
                                  teacher_grouped)

    def update_variances(beta, lambda_, alpha, sigma_mu_squared, 
                         sigma_theta_squared, sigma_epsilon_squared):

        def get_ll_helper(params):
            params = np.concatenate((params, beta, lambda_, [alpha]))
            return ll_vec_func(params)
             
        def get_grad_helper(params):
            sigma_mu_squared, sigma_theta_squared, sigma_epsilon_squared = \
                     params
            grad = get_grad_variances(sigma_mu_squared, sigma_theta_squared, 
                                      sigma_epsilon_squared, beta, 
                                      lambda_, alpha, 1e-8, ll_vec_func)
            return grad


        # bad overshooting -> bad hessian approximation -> lower maxcor needed
        # but why does the gradient get so huge?
        # Probably fixed with upper bound.
        result = minimize(get_ll_helper, [sigma_mu_squared, sigma_theta_squared,
                                           sigma_epsilon_squared], 
                          jac=get_grad_helper, method='L-BFGS-B',
                          bounds=[(1e-7, np.var(y)), (1e-7, np.var(y)), (1e-7, np.var(y))],
                          options={'disp': False, 'ftol': 1e-14})
                    #constraints={'type': 'ineq', 'fun': lambda x: x})
        sigma_mu_squared, sigma_theta_squared, sigma_epsilon_squared = result['x']

        return sigma_mu_squared, sigma_theta_squared, sigma_epsilon_squared


    def update_coefficients(sigma_mu_squared, sigma_theta_squared, 
                            sigma_epsilon_squared):
        h = 1 / (sigma_theta_squared + sigma_epsilon_squared / n_students_per_class)
        h_sum_long = teacher_grouped.apply(np.sum, h)[:, 0]
        # For beta
        precision_weights = h / h_sum_long
        y_bar_bar_long = teacher_grouped.apply(np.sum, 
                                              precision_weights * y_bar)[:, 0]
        
        x_bar_bar_long = teacher_grouped.apply(lambda x: np.sum(x, 0), 
                                               precision_weights[:, None] * x_bar,
                                               width=x_bar.shape[1])

        y_bar_tilde = y_bar - y_bar_bar_long
        x_bar_tilde = x_bar - x_bar_bar_long

        x_mat = xx_tilde / sigma_epsilon_squared + x_bar_tilde.T.dot(x_bar_tilde * h[:, None])
        y_mat = xy_tilde / sigma_epsilon_squared + x_bar_tilde.T.dot(y_bar_tilde * h)
        beta = np.linalg.solve(x_mat, y_mat)

        # Now do beta + lambda
        y_bar_bar = y_bar_bar_long[teacher_grouped.first_occurrences]

        teacher_precision_sums = h_sum_long[teacher_grouped.first_occurrences]
        # Weights aren't actually square-rooted, but do this to distribute them
        weights = 1 / np.sqrt(1 / teacher_precision_sums + sigma_mu_squared)
        assert np.all(np.isfinite(weights))

        x_bar_bar = x_bar_bar_long[teacher_grouped.first_occurrences, :]
        assert x_bar_bar.shape[1] == x_bar.shape[1]
        assert x_bar_bar.shape[1] == x_tilde.shape[1]
        y_w = (y_bar_bar - x_bar_bar.dot(beta)) * weights
        assert y_w.ndim == 1
        stacked = np.hstack((np.ones((n_teachers, 1)), x_bar_bar))
        x_w = stacked * weights[:, None]
        lambda_tmp, _, rank, _ = np.linalg.lstsq(x_w, y_w)

        if rank != x_w.shape[1]:
            warnings.warn('x_w is rank deficient')

        alpha = lambda_tmp[0]
        lambda_ = lambda_tmp[1:]

        # Check for orthogonality
        testing = False
        if testing:
            error = (y_bar_bar - x_bar_bar.dot(beta + lambda_) - alpha) * weights
            print('Mean error', np.mean(error))
            print('Error times x', np.max(np.abs(x_w.T.dot(error))))

        return beta, lambda_, alpha, x_bar_bar_long

    sigma_mu_squared, sigma_theta_squared, sigma_epsilon_squared = \
                np.ones(3) * np.var(y) / 6

    beta_old = np.zeros(x_bar.shape[1])
    lambda_old = beta_old.copy()
    sigma_mu_squared_old, sigma_theta_squared_old, sigma_epsilon_squared_old = 10, 10, 10
    max_diff = 10

    i = 0

    while abs(max_diff) > .001 and i < 15:
        print('\n')
        beta, lambda_, alpha, x_bar_bar_long =\
                update_coefficients(sigma_mu_squared, sigma_theta_squared,
                                    sigma_epsilon_squared)
                       
        sigma_mu_squared, sigma_theta_squared, sigma_epsilon_squared\
            = update_variances(beta, lambda_, alpha, sigma_mu_squared, 
                                 sigma_theta_squared, sigma_epsilon_squared)

        assert np.all(beta.shape == beta_old.shape)
        assert np.all(lambda_.shape == lambda_old.shape)

        max_diff = max(np.max(np.abs(beta - beta_old)), np.max(np.abs(lambda_ - lambda_old)),
                       abs(sigma_mu_squared - sigma_mu_squared_old),
                       abs(sigma_theta_squared - sigma_theta_squared_old),
                       abs(sigma_epsilon_squared - sigma_epsilon_squared_old))
        if max_diff > 10**4:
            print('beta\n', np.max(np.abs(beta - beta_old)), '\n', np.argmax(beta - beta_old))
            print('lambda\n', np.max(np.abs(lambda_ - lambda_old)),
                                    '\n', np.argmax(lambda_ - lambda_old))
            print('sigma mu squared\n', sigma_mu_squared - sigma_mu_squared_old)
            print('sigma theta squared\n', sigma_theta_squared - sigma_theta_squared_old)
            print('sigma epsilon squared\n', sigma_epsilon_squared - sigma_epsilon_squared_old)

        beta_old = beta.copy()
        lambda_old = lambda_.copy()
        sigma_mu_squared_old = sigma_mu_squared
        sigma_theta_squared_old = sigma_theta_squared
        sigma_epsilon_squared_old = sigma_epsilon_squared
        i += 1
        assert np.isfinite(sigma_mu_squared)

    beta, lambda_, alpha, x_bar_bar_long =\
                update_coefficients(sigma_mu_squared, sigma_theta_squared,
                                    sigma_epsilon_squared)

    x_bar_bar = x_bar_bar_long[teacher_grouped.first_occurrences, :]
    predictable_var = np.var(x_bar_bar.dot(lambda_))
    grad = get_grad_all(sigma_mu_squared, sigma_theta_squared,
                        sigma_epsilon_squared, beta, lambda_, alpha, 
                        10**(-9), ll_vec_func)

    hessian = get_hess_2(sigma_mu_squared, sigma_theta_squared,
                         sigma_epsilon_squared, beta, lambda_, alpha, 
                         10**(-9), ll_vec_func)

    try:
        asymp_var = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        print('Hessian was not invertible')
        asymp_var = float('nan')

    if not np.isfinite(asymp_var).all():
        bias_correction = float('nan')
        total_var = float('nan')
        total_var_se = float('nan')
    else:
        lambda_idx = slice(-1 - len(beta), -1)
        var_lambda = asymp_var[lambda_idx, lambda_idx]

        #print('var lambda', var_lambda)
        tmp = x_bar_bar - np.mean(x_bar_bar, 0)
        try:
            bias_correction_2 = np.sum(x_bar_bar.dot(np.linalg.cholesky(var_lambda))**2) / n_teachers
            bias_correction = np.sum(tmp.dot(np.linalg.cholesky(var_lambda))**2) / n_teachers
        except np.linalg.LinAlgError:
            print('Hessian was not positive definite')
            bias_correction_2 = np.mean([row.T.dot(var_lambda).dot(row) 
                                       for row in x_bar_bar])
            bias_correction = np.mean([row.T.dot(var_lambda).dot(row)
                                         for row in tmp])

        # print('bias correction', bias_correction)
        total_var = sigma_mu_squared + predictable_var - bias_correction
        print('total with bias correction 1', total_var)
        print('with 2', sigma_mu_squared + predictable_var - bias_correction_2)
        # Delta method: lambda variance to predictable var variance
        grad = np.zeros(asymp_var.shape[0])
        grad[0] = 1
        grad[lambda_idx] = 2 * np.mean(x_bar_bar.dot(lambda_)[:, None] * (x_bar_bar - np.mean(x_bar_bar, 0)), 0)
        total_var_se = np.sqrt(grad.T.dot(asymp_var).dot(grad))

    assert np.isfinite(sigma_mu_squared)

    return {'sigma mu squared': sigma_mu_squared, 
            'sigma theta squared': sigma_theta_squared, 
            'sigma epsilon squared': sigma_epsilon_squared, 'beta': beta, 
            'lambda':lambda_, 'alpha': alpha, 'predictable var':predictable_var,
            'gradient': grad, 'hessian': np.array(hessian),
            'total var': total_var, 'asymp var': asymp_var,
            'bias correction': bias_correction, 'x bar bar': x_bar_bar,
            'total var se': total_var_se}
    

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
        if np.issubdtype(data[col].dtype, np.number) and len(set(data[col])) <= max(data[col]):
            _, data[col] = np.unique(data[col], return_inverse=True)
        
    dense_controls = None if covariates is None else data[covariates].values
    if add_constant:
        #assert method == 'ks'
        if dense_controls is None:
            dense_controls = np.ones((len(data), 1))
        else:
            dense_controls = np.hstack((dense_controls, np.ones((len(data), 1))))

    # Recode categorical variables
    # Recode teachers as contiguous integers
    key_to_recover_teachers, data.loc[:, teacher] = \
            np.unique(data[teacher], return_inverse=True)

    if method in ['ks', 'cfr']:
        if teacher not in class_level_vars:
            class_level_vars.append(teacher)
        #assert 'teacher' in data.columns
        return moment_matching_alg(data, outcome, teacher, dense_controls, class_level_vars,
                categorical_controls, jackknife, moments_only, method, add_constant)
    elif method == 'fk':
        return fk_alg(data, outcome, teacher, dense_controls,
                      class_level_vars, categorical_controls,
                      jackknife, moments_only, teacher_controls)
    elif method == 'mle':
        return mle(data, outcome, teacher, dense_controls, categorical_controls,
                   jackknife, class_level_vars, moments_only)
    else:
        raise NotImplementedError('Only the methods ks, cfr, and fk are currently implmented.')

