import numpy as np
from config_tva import hdfe_dir
from scipy import sparse as sps
from scipy.optimize import minimize
from va_functions import get_ll_grad
import sys
sys.path += [hdfe_dir]
from hdfe import Groupby, make_dummies
from multicollinearity import find_collinear_cols

class MLE():
    def __init__(self, data, outcome, teacher, dense_controls,
                categorical_controls, jackknife, class_level_vars, moments_only):
        if not moments_only and not jackknife:
            raise NotImplementedError('jackknife must be false')
        assert len(class_level_vars) == 1
        # Set x
        if categorical_controls is None:
            x = dense_controls
        elif len(categorical_controls) == 1:
            x = np.hstack((dense_controls, make_dummies(data[categorical_controls[0]], True).A))
        else:
            x = sps.hstack([make_dummies(data[elt], True) for elt in categorical_controls]).A
            if dense_controls is not None:
                x = np.hstack((dense_controls, x))
        # Make sure everything varies within teacher, so beta is identified
        x_with_teacher_dummies = np.hstack((make_dummies(data[teacher], False).A, x))
        collinear_cols, not_collinear_cols = find_collinear_cols(x_with_teacher_dummies, .01)
        if len(collinear_cols) > 0:
            print('Found', len(collinear_cols), 'collinear columns in x')
            x = x_with_teacher_dummies[:, not_collinear_cols][:, len(set(data[teacher])):]

        y = data['outcome'].values

        # Make groupby objects
        class_grouped = Groupby(data[class_level_vars].values)
        assert class_grouped.already_sorted
        self.n_students_per_class = class_grouped.apply(len, y, broadcast=False)
        self.n_students = len(data)
        self.y_tilde = class_grouped.apply(lambda x: x - np.mean(x), y)
        self.x_tilde = class_grouped.apply(lambda x: x - np.mean(x, 0), x, width=x.shape[1])
        self.x_bar = class_grouped.apply(lambda x: np.mean(x, 0), x, broadcast=False,
                                              width=x.shape[1])
        self.y_bar = class_grouped.apply(np.mean, y, broadcast=False)
        teachers = data[teacher].values[class_grouped.first_occurrences]
        self.teacher_grouped = Groupby(teachers)
        self.n_teachers = len(self.teacher_grouped.first_occurrences)
        x_bar_bar = self.teacher_grouped.apply(lambda x: np.mean(x, 0), self.x_bar,
                                               width=self.x_bar.shape[1], broadcast=False)
        x_bar_bar = np.hstack((np.ones((self.n_teachers, 1)), x_bar_bar))
        collin_x_bar_bar, not_collin_x_bar_bar = find_collinear_cols(x_bar_bar)
        if len(collin_x_bar_bar) > 0:
            print('Found', len(collin_x_bar_bar), 'collinear columns in x bar bar')
            self.x_tilde = self.x_tilde[:, not_collin_x_bar_bar[1:] - 1]
            self.x_bar = self.x_bar[:, not_collin_x_bar_bar[1:] - 1]
        self.xx_tilde = self.x_tilde.T.dot(self.x_tilde)
        self.xy_tilde = self.x_tilde.T.dot(self.y_tilde)[:, 0]
        assert self.xy_tilde.ndim == 1
        self.n_classes = len(class_grouped.first_occurrences)
        self.n_students = len(data)

        self.sigma_mu_squared, self.sigma_theta_squared, self.sigma_epsilon_squared = None, None, None
        self.beta = np.zeros(self.x_bar.shape[1])
        self.lambda_ = np.zeros(self.x_bar.shape[1])
        self.alpha = 0
        self.h = np.ones(self.n_classes)
        self.h_sum = self.teacher_grouped.apply(lambda x: np.sum(x, 0), self.h)
        self.y_bar_tilde = np.zeros(self.y_bar.shape)
        self.x_bar_tilde = np.zeros(self.x_bar.shape)

#     def set_h(self):
#         assert np.all(self.h > 0)
#         assert np.all(self.h_sum > 0)
#         return

    def update_variances(self):
        def get_ll_helper(params):
            log_sigma_mu_squared, log_sigma_theta_squared, log_sigma_epsilon_squared = params
            return get_ll_grad(log_sigma_mu_squared, log_sigma_theta_squared, log_sigma_epsilon_squared,
                               self.n_students_per_class, self.n_classes, self.n_students,
                               self.y_tilde, self.x_tilde, self.x_bar, self.y_bar, self.beta,
                               self.lambda_, self.alpha, self.teacher_grouped, get_grad=False)

        def get_grad_helper(params):
            log_sigma_mu_squared, log_sigma_theta_squared, log_sigma_epsilon_squared = params
            _, grad = get_ll_grad(log_sigma_mu_squared, log_sigma_theta_squared,
                                  log_sigma_epsilon_squared, self.n_students_per_class,
                                  self.n_classes, self.n_students, self.y_tilde, self.x_tilde,
                                  self.x_bar, self.y_bar, self.beta, self.lambda_, self.alpha,
                                  self.teacher_grouped, get_grad=True, return_means=False,
                                  variances_only=True)
            return grad

        bounds = np.var(self.y_tilde) * 1e-7, np.var(self.y_tilde)

        result = minimize(get_ll_helper,
                          np.log(np.array([self.sigma_mu_squared, self.sigma_theta_squared,
                                           self.sigma_epsilon_squared])),
                          jac=get_grad_helper, method='L-BFGS-B', bounds=[bounds, bounds, bounds],
                          options={'ftol': 1e-14, 'gtol':1e-7})
       #  = result['x']
        self.sigma_mu_squared, self.sigma_theta_squared, self.sigma_epsilon_squared = np.exp(result['x'])
        return

#     def get_ll_grad(self, get_grad=False, start=0, variances_only=False):
#         one_over_h_sum = 1 / self.h_sum
#         bar_bar_err = self.y_bar_bar - self.x_bar_bar.dot(self.beta + self.lambda_) - self.alpha
#
#         ll = (self.n_classes - self.n_students) * np.log(self.sigma_epsilon_squared)\
#             + np.sum(np.log(self.h)) - np.sum(np.log(self.h_sum))\
#             - np.sum(np.log(self.sigma_mu_squared + one_over_h_sum))\
#             - np.sum((self.y_tilde[:, 0] - self.x_tilde.dot(self.beta))**2) / self.sigma_epsilon_squared\
#             - self.h.dot((self.y_bar_tilde - self.x_bar_tilde.dot(self.beta))**2)\
#             - np.dot(bar_bar_err**2, 1 / (self.sigma_mu_squared + one_over_h_sum))
#         ll /= -2
#         assert np.isfinite(ll)
#         if not get_grad:
#             return ll
#         gradient = [None, None, None]
#         if start == 0:
#             # TODO: Fix this for not working in logs
#             tmp = self.sigma_mu_squared + one_over_h_sum
#             grad_s_mu = np.dot(bar_bar_err ** 2, 1 / tmp ** 2) - np.sum(1 / tmp)
#             grad_s_mu /= -2
#             gradient[0] = grad_s_mu
#         if start <= 2:
#             first = one_over_h_sum - (self.y_bar_tilde - self.x_bar_tilde.dot(beta))**2
#             second = -one_over_h_sum + one_over_h_sum**2 / (self.sigma_mu_squared + one_over_h_sum)\
#                 -(one_over_h_sum / (self.sigma_mu_squared + one_over_h_sum))**2 * bar_bar_err**2
#             d_h_d_log_e = -h**2 * self.sigma_epsilon_squared / self.n_students_per_class
#             d_e_sum = self.teacher_grouped.apply(np.sum, d_h_d_log_e, broadcast=False)
#             grad_s_eps = self.n_classes - self.n_students + d_h_d_log_e.dot(first)
#             grad_s_eps += d_e_sum.dot(second)
#             grad_s_eps += np.sum((self.y_tilde[:, 0] - self.x_tilde.dot(self.beta)**2) / self.sigma_epsilon_squared)
#             gradient[2] = grad_s_eps / -2
#
#             # theta
#             d_h_d_log_t = - self.h**2
#             d_h_sum = self.teacher_grouped.apply(np.sum, d_h_d_log_t, broadcast=False)
#             grad_s_theta = d_h_d_log_t.dot(first)
#             grad_s_theta += d_h_sum.dot(second)
#             gradient[1] = grad_s_theta
#
#         if variances_only:
#             return ll, np.array(gradient)
#
#         grad_beta = -self.x_tilde.dot(self.y_tilde[:, 0] - self.x_tilde.dot(self.beta)) / self.sigma_epsilon_squared
#         grad_beta -= (self.h[:, None] * self.x_bar_tilde).T.dot(self.y_bar_tilde - self.x_bar_tilde.dot(self.beta))
#         w = 1 / (self.sigma_mu_squared + one_over_h_sum)
#         grad_lambda = -(w * self.x_bar_bar).T.dot(bar_bar_err)
#         grad_alpha = -bar_bar_err.dot(w)
#         gradient = np.concatenate((gradient, grad_beta, grad_lambda, grad_alpha))
#         return ll, gradient


    def fit(self):
        pass
