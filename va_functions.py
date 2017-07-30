import numpy as np
import numpy.linalg as linalg
import pandas as pd
import scipy.sparse as sps
from functools import reduce
import sys
from config_tva import *
sys.path += [hdfe_dir]
from hdfe import Groupby
import warnings

def get_ll_grad(log_sigma_mu_squared, log_sigma_theta_squared, log_sigma_epsilon_squared,
           n_students_per_class, n_classes, n_students, y_tilde, x_tilde,
           x_bar, y_bar, beta, lambda_, alpha, teacher_grouped, get_grad=False,
           return_means=False, h=None, h_sum=None, y_bar_tilde=None, 
           x_bar_tilde=None, y_bar_bar=None, x_bar_bar=None, start=0, variances_only=False):
    assert np.isscalar(log_sigma_mu_squared)
    assert np.isscalar(log_sigma_theta_squared)
    assert np.isscalar(log_sigma_epsilon_squared)
    assert np.isscalar(alpha)
    assert np.isscalar(n_students)
    assert n_students_per_class.shape[0] == n_classes
    assert y_tilde.shape[0] == n_students

    sigma_mu_squared = np.exp(log_sigma_mu_squared)
    sigma_theta_squared = np.exp(log_sigma_theta_squared)
    sigma_epsilon_squared = np.exp(log_sigma_epsilon_squared)


    if h is None:
        h = 1 / (sigma_theta_squared + sigma_epsilon_squared / n_students_per_class)
        assert np.all(h > 0)
    else:
        assert np.all(h > 0)
    if y_bar_tilde is None or x_bar_tilde is None or h_sum is None:
        h_sum_long = teacher_grouped.apply(np.sum, h)[:, 0]
        assert np.min(h_sum_long) >= np.min(h)
        precision_weights = h / h_sum_long
        y_bar_bar_long = teacher_grouped.apply(np.sum, 
                                              precision_weights * y_bar)[:, 0]
        
        x_bar_bar_long = teacher_grouped.apply(lambda x: np.sum(x, 0), 
                                               precision_weights[:, None] * x_bar,
                                               width=x_bar.shape[1])

        y_bar_tilde = y_bar - y_bar_bar_long
        x_bar_tilde = x_bar - x_bar_bar_long

        h_sum = h_sum_long[teacher_grouped.first_occurrences]
        assert np.min(h_sum) == np.min(h_sum_long)
    
    assert np.all(h_sum > 0)

    if y_bar_bar is None or x_bar_bar is None:
        y_bar_bar = y_bar_bar_long[teacher_grouped.first_occurrences]
        x_bar_bar = x_bar_bar_long[teacher_grouped.first_occurrences, :]

    one_over_h_sum = 1 / h_sum
    bar_bar_err = y_bar_bar - x_bar_bar.dot(beta + lambda_) - alpha

    ll = (n_classes - n_students) * log_sigma_epsilon_squared\
         + np.sum(np.log(h)) - np.sum(np.log(h_sum))\
         - np.sum(np.log(sigma_mu_squared + one_over_h_sum))\
         - np.sum((y_tilde[:, 0] - x_tilde.dot(beta))**2) / sigma_epsilon_squared\
         - h.dot((y_bar_tilde - x_bar_tilde.dot(beta))**2)\
         - np.dot(bar_bar_err**2, 1 / (sigma_mu_squared + one_over_h_sum))
    ll /= -2
    assert np.isfinite(ll)
    if not get_grad:
        if return_means:
            return ll, h, h_sum, y_tilde, x_tilde, y_bar_tilde, x_bar_tilde,\
                    y_bar_bar, x_bar_bar
        return ll

    gradient = [None, None, None]
    if start == 0:
        tmp = sigma_mu_squared + 1 / h_sum
        grad_s_mu = np.dot(bar_bar_err**2, 1 / tmp**2) - np.sum(1 / tmp)
        grad_s_mu *= -sigma_mu_squared / 2
        gradient[0] = grad_s_mu

    if start <= 2:
        first = 1 / h - (y_bar_tilde - x_bar_tilde.dot(beta))**2
        second = -one_over_h_sum + one_over_h_sum**2 / (sigma_mu_squared + one_over_h_sum)\
                - (one_over_h_sum / (sigma_mu_squared + one_over_h_sum))**2 * bar_bar_err**2

        d_h_d_log_e = -h**2 * sigma_epsilon_squared / n_students_per_class
        d_e_sum = teacher_grouped.apply(np.sum, d_h_d_log_e, broadcast=False)

        grad_s_eps = n_classes - n_students + d_h_d_log_e.dot(first)
        grad_s_eps += d_e_sum.dot(second)
        grad_s_eps += np.sum((y_tilde[:, 0] - x_tilde.dot(beta))**2) / sigma_epsilon_squared

        gradient[2] = grad_s_eps / -2
 
    # Get gradient for log sigma theta squared
    if start <= 1:
        d_h_d_log_t = -h**2 * sigma_theta_squared
        d_h_sum = teacher_grouped.apply(np.sum, d_h_d_log_t, broadcast=False)

        grad_s_theta = d_h_d_log_t.dot(first)
        grad_s_theta += d_h_sum.dot(second)
        grad_s_theta /= -2
        gradient[1] = grad_s_theta
       
    if variances_only:
        return ll, np.array(gradient)
    grad_beta = -x_tilde.T.dot(y_tilde[:, 0] - x_tilde.dot(beta)) / np.exp(log_sigma_epsilon_squared)\
                - (h[:, None] * x_bar_tilde).T.dot(y_bar_tilde - x_bar_tilde.dot(beta))

    w = 1 / (np.exp(log_sigma_mu_squared) + 1 / h_sum[:, None])
    grad_lambda = -(w * x_bar_bar).T.dot(bar_bar_err)
    grad_alpha = -bar_bar_err.dot(w)

    gradient = np.concatenate((gradient, grad_beta, grad_lambda, grad_alpha))

    return ll, gradient


def get_ll_vec_func(n_students_per_class, n_classes, n_students, y_tilde,
                    x_tilde, x_bar, y_bar, teacher_grouped):
    """
    Returns a function of a vector which contains variances, 
    beta, lambda_, and alpha
    """
    k = x_bar.shape[1] 
    def f(vec, get_grad=False, return_means=False, h=None, h_sum=None, 
          x_bar_tilde=None, y_bar_tilde=None, y_bar_bar=None, x_bar_bar=None,
          start=0, variances_only=False):
        return get_ll_grad(vec[0], vec[1], vec[2], n_students_per_class, 
                           n_classes, n_students, y_tilde, x_tilde, x_bar, 
                           y_bar, vec[3:3 + k], vec[3+k: 3 + 2 * k], vec[-1], 
                           teacher_grouped, get_grad, return_means, h, h_sum, 
                           y_bar_tilde=y_bar_tilde, x_bar_tilde=x_bar_tilde, 
                           y_bar_bar=y_bar_bar, x_bar_bar=x_bar_bar, 
                           start=start, variances_only=variances_only)
    return f


def get_hess(log_sigma_mu_squared, log_sigma_theta_squared, log_sigma_epsilon_squared, 
             beta, lambda_, alpha, epsilon, ll_vec_func, n_students_per_class):
    for elt in [log_sigma_mu_squared, log_sigma_theta_squared, log_sigma_epsilon_squared, 
                alpha, epsilon]:
        assert np.isscalar(elt)

    vec = np.concatenate(([log_sigma_mu_squared, log_sigma_theta_squared, 
                           log_sigma_epsilon_squared], 
                          beta, lambda_, [alpha]))
    eye = np.eye(len(vec))
    hessian = np.zeros((len(vec), len(vec)))
     
    ans = ll_vec_func(vec, return_means=True)
    ll, h, h_sum, y_tilde, x_tilde, y_bar_tilde, x_bar_tilde, y_bar_bar, x_bar_bar = ans
    assert np.all(h > 0)
    assert np.all(h_sum > 0)
    assert np.max(h) <= np.max(h_sum)

    for i in range(len(vec)):
        # when updating variances, need to recalculate h, h_sum, etc.
        if i < 3:
            ll_up, upper = ll_vec_func(vec + eye[:, i] * epsilon, start=i, get_grad=True)
            ll_lo, lower = ll_vec_func(vec - eye[:, i] * epsilon, start=i, get_grad=True)
            # Since we should be at a minimum, the gradient should be positive
            # to the right and negative to the left
            if ll_up > ll or ll_lo > ll:
                warnings.warn('Found a lower ll; you may not be at an optimum')
            if upper[i] < 0 or lower[i] > 0 or (upper[i] - lower[i] < 0):
                warnings.warn('Gradient indicates you may not be at an optimum')
        # When updating other parameters, don't need to recalculate stuff
        else:
            _, upper = ll_vec_func(vec + eye[:,i] * epsilon, start=i, get_grad=True,
                                 h=h, h_sum=h_sum, y_bar_tilde=y_bar_tilde, 
                                 x_bar_tilde=x_bar_tilde, y_bar_bar=y_bar_bar, 
                                 x_bar_bar=x_bar_bar)
            _, lower = ll_vec_func(vec - eye[:,i] * epsilon, start=i, get_grad=True,
                                 h=h, h_sum=h_sum, y_bar_tilde=y_bar_tilde, 
                                 x_bar_tilde=x_bar_tilde, y_bar_bar=y_bar_bar, 
                                 x_bar_bar=x_bar_bar)
        hess = (upper[i:] - lower[i:]) / (2 * epsilon)
        hessian[i, i:] = hess
        hessian[i:, i] = hess

    return hessian


toString = lambda *x: '\n'.join((str(elt) for elt in x))
# For printing stuff
# sample use
# text = Test('teaher number', teacher_number)
# text.append('observation number', observation_number, other_number)
# print(text)
class Text(object): 
    def __init__(self, *string):
        self.text = toString(*string)
    def append(self, *string):
        self.text = self.text + '\n' + toString(*string)
    def __str__(self):
        return '\n\n\n' + self.text


def make_reg_table(reg_obj, var_names, categorical_controls):
    def format(param, t_stat, se):
        if abs(t_stat) > 3.291:
            stars = '***'
        elif abs(t_stat) > 2.576:
            stars = '**'
        elif abs(t_stat) > 1.96:
            stars = '*'
        else: stars = '*'
        return (str(int(round(param * 1000)) / 1000) + stars
              , '(' + str(int(round(se * 1000)) / 1000) + ')')


    coef_col = reduce(lambda x, y: x + y
                    , (format(p, t, se) 
                       for p, t, se 
                       in zip(reg_obj.params, reg_obj.tvalues, reg_obj.bse)))
    tuples = [(name, type) for name in var_names for type in ('beta', 'se')]
    coef_col = pd.Series(coef_col, index=pd.MultiIndex.from_tuples(tuples))

    # Just keep the ones that are not categorical
    keeps = [all([cat not in t[0] for cat in categorical_controls]) 
                                  for t in tuples]
    coef_col = coef_col[keeps]
    for cat in categorical_controls:
        coef_col[(cat, 'F')] = 'X'
        #vars = [cat in v for v in var_names]
        #B = np.zeros((sum(vars), len(var_names)))
        #for i, idx in enumerate(np.where(np.array(vars))[0]):
        #    B[i, idx] = 1
        #f_results = reg_obj.f_test(B).__dict__
        #coef_col[(cat, 'F')] = f_results['fvalue'][0][0]
        #coef_col[(cat, 'p')] = '(' + str(int(round(f_results['pvalue'] * 1000)) / 1000) + ')'

    coef_col[('N', '')] = reg_obj.df_resid + len(var_names) + 1
    coef_col[('R-squared', '')] = reg_obj.rsquared
    return coef_col


# List of regression objects; list of lists of controls
def make_table(regs, controls, categorical_controls):
    tab =  pd.concat((make_reg_table(x, var_names, categorical_controls) 
                      for x, var_names in zip(regs, controls)), axis=1)
    # order columns better
    constant = tab.select(lambda x: x[0] == 'constant')
    beta_parts = tab.select(lambda x: x[1] in ('beta', 'se') 
                                      and x[0] != 'constant')

    beta_not_null = beta_parts[pd.notnull(beta_parts.iloc[:, 0])]
    beta_null = beta_parts[pd.isnull(beta_parts.iloc[:, 0])]
    F_parts = tab.select(lambda x: x[1] in ('F', 'p'))
    rest = tab.select(lambda x: x[1] == '')

    tab = pd.concat((constant, beta_not_null, beta_null, F_parts, rest))
    
    for col in tab.columns:
        tab.loc[pd.isnull(tab.loc[:, col]), col] = ''

    tab = tab.reset_index()
    tab.iloc[1:-2:2, 0] = ''

    return tab.drop('level_1', axis=1)


def estimate_var_epsilon(data):
    data = data[data['var'].notnull()]
    var_epsilon_hat = np.dot(data['var'].values, data['size'].values)\
                      /np.sum(data['size'])
    assert var_epsilon_hat > 0
    return var_epsilon_hat
    

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
    return np.array(result)


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
        start = int(len(x) * i / nbins)
        end = int(len(x) * (i+1) / nbins)
        bins[i] = np.mean(x[start:end])
        y_means[i] = np.mean(y[start:end])
        y_medians[i] = np.median(y[start:end])
        y_5[i] = np.percentile(y[start:end], 5)
        y_95[i] = np.percentile(y[start:end], 95)

    return bins, y_means, y_medians, y_5, y_95


def check_calibration(errors, precisions):         
    mean_error = np.mean(errors)
    se = (np.var(errors) / len(errors))**.5

    standardized_errors = errors**2 * precisions
    mean_standardized_error = np.mean(standardized_errors)
    standardized_error_se = (np.var(standardized_errors) / \
                             len(standardized_errors))**.5
    assert mean_error > -3 * se
    assert mean_error < 3 * se
    assert mean_standardized_error > 1 - 2 * standardized_error_se 
    assert mean_standardized_error < 1 + 2 * standardized_error_se 


# df should actually be an array
def get_unshrunk_va(array, var_theta_hat, var_epsilon_hat, jackknife):
    if jackknife:
        unshrunk = np.array(np.sum(array[:, 1]) - array[:, 1]) / (len(array)-1)
    else:
        unshrunk = np.mean(array[:, 1])

    return unshrunk


def get_va(df, var_theta_hat, var_epsilon_hat, var_mu_hat, jackknife):
    assert(var_mu_hat > 0)
    array = df.values
    precisions = np.array([1 / (var_theta_hat + var_epsilon_hat / class_size) 
                          for class_size in array[:, 0]])
    numerators = precisions * array[:, 1]
    precision_sum = np.sum(precisions)
    num_sum = np.sum(numerators)
    if jackknife:
        denominators = np.array([precision_sum - p for p in precisions]) \
                 + 1 / var_mu_hat
        return_val =[[(num_sum - n) / d, 1 / d]
                         for n, d in zip(numerators, denominators)]
    else:
        denominator = precision_sum + 1 / var_mu_hat
        return_val = [[num_sum / denominator, 1 / denominator]]

    return pd.DataFrame(data=np.array(return_val))
