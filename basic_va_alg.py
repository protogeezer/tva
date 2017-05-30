from va_functions import *#, remove_collinear_cols
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
from variance_ls_numopt import get_g_and_tau
from scipy.optimize import minimize
from variance_ls_numopt import newton
import ad
from ad.admath import *

def profile(f):
    return f

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
    y = data[outcome].values
    n_students_per_class = class_grouped.apply(len, y, broadcast=False)
    n_classes = len(class_grouped.first_occurrences)
    n_students = len(data)

    y_tilde = class_grouped.apply(lambda x: x - np.mean(x), y)
    x_tilde = class_grouped.apply(lambda x: x - np.mean(x, 0), x, 
                                  width = x.shape[1])
    xx_tilde = x_tilde.T.dot(x_tilde)
    xy_tilde = x_tilde.T.dot(y_tilde)[:, 0]
    assert xy_tilde.ndim == 1

    x_bar = class_grouped.apply(lambda x: np.mean(x, 0), x, broadcast=False, 
                                width=x.shape[1])
    y_bar = class_grouped.apply(np.mean, y, broadcast=False)
    
    teachers = data[teacher].values[class_grouped.first_occurrences]
    # Should only be applied to objects that have been created with
    # class_grouped.apply
    teacher_grouped = Groupby(teachers)
    assert teacher_grouped.already_sorted
    n_teachers = len(teacher_grouped.first_occurrences)

    def get_precisions(sigma_theta_squared, sigma_epsilon_squared):
        return 1 / (sigma_theta_squared + sigma_epsilon_squared / n_students_per_class)
    
    def update_variances(beta, beta_plus_lambda, alpha, sigma_mu_squared, 
                         sigma_theta_squared, sigma_epsilon_squared,
                         y_bar_bar, x_bar_bar, y_bar_tilde, x_bar_tilde):
 
        def get_ll(params):
            sigma_mu_squared, sigma_theta_squared, sigma_epsilon_squared = params
            h = get_precisions(sigma_theta_squared, sigma_epsilon_squared)
            h_sum = teacher_grouped.apply(np.sum, h, broadcast=False)

            ll = (n_classes - n_students) * np.log(sigma_epsilon_squared)\
                 + np.sum(np.log(h)) - np.sum(np.log(h_sum))\
                 - np.sum(np.log(sigma_mu_squared + 1 / h_sum))\
                 - np.sum((y_tilde[:, 0] - x_tilde.dot(beta))**2) / sigma_epsilon_squared\
                 - h.dot((y_bar_tilde - x_bar_tilde.dot(beta))**2)\
                 - np.dot((y_bar_bar - x_bar_bar.dot(beta_plus_lambda) - alpha)**2,
                          1 / (sigma_mu_squared + 1 / h_sum))
            return -1 * ll


        auto_diff = False

        if auto_diff:

            teacher_dummies = make_dummies(teachers, False).A

            def get_ll(sigma_mu_squared, sigma_theta_squared, sigma_epsilon_squared):
                h = 1 / ((n_students_per_class / sigma_epsilon_squared)**(-1) + sigma_theta_squared)
                h_sum = teacher_grouped.apply(np.sum, h, broadcast=False)
                print('Old h sum', h_sum[0])
                print(h[0])
                h_sum = teacher_dummies.T.dot(h)
                print(h_sum[0])

                ll = (n_classes - n_students) * log(sigma_epsilon_squared)
                ll += sum(log(h)) - sum(log(h_sum))
                ll += -sum(h_sum**(-1) + log(sigma_mu_squared))
                ll += -sum((y_tilde[:, 0] - x_tilde.dot(beta))**2) / sigma_epsilon_squared
                ll += -h.dot((y_bar_tilde - x_bar_tilde.dot(beta))**2)
                ll += -np.dot((y_bar_bar - x_bar_bar.dot(beta_plus_lambda) - alpha)**2,
                              1 / (h_sum**(-1) + sigma_mu_squared))
                return -1 * ll

            def get_ll_helper(params, get_grad=False, get_hess=False):
                print('Params', params)
                sigma_mu_squared, sigma_theta_squared, sigma_epsilon_squared = ad.adnumber(params)
                ll = get_ll(sigma_mu_squared, sigma_theta_squared, sigma_epsilon_squared)
                print('ll', ll)
                if get_grad:
                    grad = np.array(ll.gradient([sigma_mu_squared, sigma_theta_squared, sigma_epsilon_squared]))
                    assert grad.shape[0] == 3
                    if get_hess:
                        hess = np.array(ll.hessian([sigma_mu_squared, sigma_theta_squared, sigma_epsilon_squared]))
                        assert hess.shape[0] == 3
                        assert hess.shape[1] == 3
                        return ll, grad, hess
                    return ll, grad
                return ll

            print('\n\n\n')
            result = newton(get_ll_helper, 
                            ad.adnumber(np.array([sigma_mu_squared, sigma_theta_squared, sigma_epsilon_squared])))
            sigma_mu_squared, sigma_theta_squared, sigma_epsilon_squared = [float(x) for x in result[0]]
        else:
            result = minimize(get_ll, 
                        (sigma_mu_squared, sigma_theta_squared, sigma_epsilon_squared), 
                        method='COBYLA',
                        constraints={'type': 'ineq', 'fun': lambda x: x})
            sigma_mu_squared, sigma_theta_squared, sigma_epsilon_squared = result['x']

        return sigma_mu_squared, sigma_theta_squared, sigma_epsilon_squared


    @profile
    def update_coefficients(sigma_mu_squared, sigma_theta_squared, 
                            sigma_epsilon_squared):
        h = get_precisions(sigma_theta_squared, sigma_epsilon_squared)
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
        x_bar_bar = x_bar_bar_long[teacher_grouped.first_occurrences]
        teacher_precision_sums = h_sum_long[teacher_grouped.first_occurrences]
        # Weights aren't actually square-rooted, but do this to distribute them
        weights = 1 / np.sqrt(1 / teacher_precision_sums + sigma_mu_squared)
        assert np.all(np.isfinite(weights))

        y_w = y_bar_bar * weights
        x_w = np.hstack((np.ones((n_teachers, 1)), x_bar_bar)) * weights[:, None]
        beta_plus_lambda = np.linalg.lstsq(x_w, y_w)[0]
        alpha, beta_plus_lambda = beta_plus_lambda[0], beta_plus_lambda[1:]

        return beta, beta_plus_lambda, alpha, y_bar_bar, x_bar_bar, y_bar_tilde, x_bar_tilde

    beta = np.linalg.lstsq(y_tilde, x_tilde)[0]
    beta_plus_lambda = np.zeros(x.shape[1])
    sigma_mu_squared = np.var(y) / 3
    sigma_theta_squared = sigma_mu_squared
    sigma_epsilon_squared = sigma_mu_squared

    for i in range(10):
        beta, beta_plus_lambda, alpha, y_bar_bar, x_bar_bar, y_bar_tilde, x_bar_tilde =\
                update_coefficients(sigma_mu_squared, sigma_theta_squared,
                                    sigma_epsilon_squared)

        sigma_mu_squared, sigma_theta_squared, sigma_epsilon_squared\
            = update_variances(beta, beta_plus_lambda, alpha, sigma_mu_squared, 
                                 sigma_theta_squared, sigma_epsilon_squared,
                                 y_bar_bar, x_bar_bar, y_bar_tilde, x_bar_tilde)

    lambda_ = beta_plus_lambda - beta
    predictable_var = np.var(x_bar_bar.dot(lambda_))
    # Get asymptotic distribution of lambda

    return {'sigma mu squared': sigma_mu_squared, 'sigma theta squared': sigma_theta_squared, 
            'sigma epsilon squared': sigma_epsilon_squared, 'beta': beta, 'lambda': lambda_, 
            'alpha': alpha, 'predictable var': predictable_var}
    

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

