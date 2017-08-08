import numpy as np
from config_tva import hdfe_dir
from scipy import sparse as sps
from scipy.optimize import minimize
import warnings
import pandas as pd
import sys
sys.path += [hdfe_dir]
from hdfe import Groupby, make_dummies
from multicollinearity import find_collinear_cols


class MLE:
    def __init__(self, data: pd.DataFrame, outcome: str, teacher: str,
                 dense_controls: np.ndarray, categorical_controls: list,
                 jackknife: bool, class_level_vars: list, moments_only: bool):
        """

        :param data:
        :param outcome:
        :param teacher:
        :param dense_controls:
        :param categorical_controls:
        :param jackknife:
        :param class_level_vars:
        :param moments_only:
        """
        if not moments_only and jackknife:
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
        self.moments_only = moments_only

        y = data[outcome].values

        # Make Groupby objects
        class_grouped = Groupby(data[class_level_vars].values)
        assert class_grouped.already_sorted
        self.n_students_per_class = class_grouped.apply(len, y, broadcast=False)
        self.n_students = len(y)
        self.y_tilde = class_grouped.apply(lambda vec: vec - np.mean(vec), y)
        self.x_tilde = class_grouped.apply(lambda arr: arr - np.mean(arr, 0), x, width=x.shape[1])
        self.x_bar = class_grouped.apply(lambda arr: np.mean(arr, 0), x, broadcast=False,
                                         width=x.shape[1])
        del x
        self.y_bar = class_grouped.apply(np.mean, y, broadcast=False)

        teachers = data[teacher].values[class_grouped.first_occurrences]
        self.sigma_mu_squared, self.sigma_theta_squared, self.sigma_epsilon_squared = \
            np.ones(3) * np.var(data[outcome]) / 6
        del data
        self.teacher_grouped = Groupby(teachers)
        assert self.teacher_grouped.already_sorted
        self.n_teachers = len(self.teacher_grouped.first_occurrences)
        x_bar_bar = self.teacher_grouped.apply(lambda arr: np.mean(arr, 0), self.x_bar,
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
        self.n_students = self.x_tilde.shape[0]

        self.beta = np.zeros(self.x_bar.shape[1])
        self.lambda_ = np.zeros(self.x_bar.shape[1])
        self.alpha = 0
        self.h, self.h_sum = None, None
        self.y_bar_tilde, self.x_bar_tilde = None, None
        self.x_bar_bar_long, self.y_bar_bar_long = None, None
        self.predictable_var, self.total_var = None, None
        self.individual_scores = None
        n_params = 2 * self.x_bar.shape[1] + 4
        self.asymp_var = np.full((n_params, n_params), np.nan)
        self.hessian = self.asymp_var.copy()
        self.bias_correction, self.total_var, self.total_var_se = np.nan, np.nan, np.nan
        self.sigma_mu_squared_se = np.nan

    def get_ll_grad(self, log_sigma_mu_squared: float = None, log_sigma_theta_squared=None,
                    log_sigma_epsilon_squared=None, beta=None, lambda_=None, alpha=None,
                    get_grad=False, variances_only=False):

        if beta is None:
            beta = self.beta
        if lambda_ is None:
            lambda_ = self.lambda_
        if alpha is None:
            alpha = self.alpha

        change_variances = log_sigma_mu_squared is not None or log_sigma_theta_squared is not None\
            or log_sigma_epsilon_squared is not None
        if log_sigma_mu_squared is None:
            log_sigma_mu_squared = np.log(self.sigma_mu_squared)
            sigma_mu_squared = self.sigma_mu_squared
        else:
            sigma_mu_squared = np.exp(log_sigma_mu_squared)

        if log_sigma_theta_squared is None:
            sigma_theta_squared = self.sigma_theta_squared
        else:
            sigma_theta_squared = np.exp(log_sigma_theta_squared)

        if log_sigma_epsilon_squared is None:
            log_sigma_epsilon_squared = np.log(self.sigma_epsilon_squared)
            sigma_epsilon_squared = self.sigma_epsilon_squared
        else:
            sigma_epsilon_squared = np.exp(log_sigma_epsilon_squared)

        if change_variances:
            h = 1 / (sigma_theta_squared + sigma_epsilon_squared / self.n_students_per_class)
        else:
            h = self.h
        assert isinstance(h, np.ndarray)
        assert np.all(h > 0)
        if change_variances:
            h_sum_long = self.teacher_grouped.apply(np.sum, h)[:, 0]
            assert np.min(h_sum_long) >= np.min(h)
            precision_weights = h / h_sum_long
            y_bar_bar_long = self.teacher_grouped.apply(np.sum,
                                                        precision_weights * self.y_bar)[:, 0]

            x_bar_bar_long = self.teacher_grouped.apply(lambda x: np.sum(x, 0),
                                                        precision_weights[:, None] * self.x_bar,
                                                        width=self.x_bar.shape[1])

            y_bar_tilde = self.y_bar - y_bar_bar_long
            x_bar_tilde = self.x_bar - x_bar_bar_long

            h_sum = h_sum_long[self.teacher_grouped.first_occurrences]
            assert np.min(h_sum) == np.min(h_sum_long)
        else:
            y_bar_tilde = self.y_bar_tilde
            x_bar_tilde = self.x_bar_tilde
            y_bar_bar_long = self.y_bar_bar_long
            x_bar_bar_long = self.x_bar_bar_long
            h_sum = self.h_sum

        assert isinstance(h_sum, np.ndarray)
        assert np.all(h_sum > 0)

        y_bar_bar = y_bar_bar_long[self.teacher_grouped.first_occurrences]
        x_bar_bar = x_bar_bar_long[self.teacher_grouped.first_occurrences, :]

        # Done with setup
        one_over_h_sum = 1 / h_sum
        bar_bar_err = y_bar_bar - x_bar_bar.dot(beta + lambda_) - alpha

        ll = (self.n_classes - self.n_students) * log_sigma_epsilon_squared \
            + np.sum(np.log(h)) - np.sum(np.log(h_sum)) \
            - np.sum(np.log(sigma_mu_squared + one_over_h_sum)) \
            - np.sum((self.y_tilde[:, 0] - self.x_tilde.dot(beta))**2) / sigma_epsilon_squared \
            - h.dot((y_bar_tilde - x_bar_tilde.dot(beta))**2) \
            - np.dot(bar_bar_err**2, 1 / (sigma_mu_squared + one_over_h_sum))
        ll /= -2
        assert np.isfinite(ll)
        if not get_grad:
            return ll

        gradient = [None, None, None]
        # Gradient for log sigma mu squared
        tmp = sigma_mu_squared + 1 / h_sum
        grad_s_mu = np.dot(bar_bar_err**2, 1 / tmp**2) - np.sum(1 / tmp)
        grad_s_mu *= -sigma_mu_squared / 2
        gradient[0] = grad_s_mu

        # Get gradient for log sigma theta squared
        first = 1 / h - (y_bar_tilde - x_bar_tilde.dot(beta))**2
        second = -one_over_h_sum + one_over_h_sum**2 / (sigma_mu_squared + one_over_h_sum) \
                 - (one_over_h_sum / (sigma_mu_squared + one_over_h_sum))**2 * bar_bar_err**2

        d_h_d_log_t = -h**2 * sigma_theta_squared
        d_h_sum = self.teacher_grouped.apply(np.sum, d_h_d_log_t, broadcast=False)

        grad_s_theta = d_h_d_log_t.dot(first)
        grad_s_theta += d_h_sum.dot(second)
        grad_s_theta /= -2
        gradient[1] = grad_s_theta

        # Get gradient for log sigma epsilon squared
        d_h_d_log_e = -h**2 * sigma_epsilon_squared / self.n_students_per_class
        d_e_sum = self.teacher_grouped.apply(np.sum, d_h_d_log_e, broadcast=False)

        grad_s_eps = self.n_classes - self.n_students + d_h_d_log_e.dot(first)
        grad_s_eps += d_e_sum.dot(second)
        grad_s_eps += np.sum((self.y_tilde[:, 0] - self.x_tilde.dot(beta))**2) / sigma_epsilon_squared

        gradient[2] = grad_s_eps / -2

        if variances_only:
            return ll, np.array(gradient)
        grad_beta = -self.x_tilde.T.dot(self.y_tilde[:, 0] - self.x_tilde.dot(beta)) / sigma_epsilon_squared \
                    - (h[:, None] * x_bar_tilde).T.dot(y_bar_tilde - x_bar_tilde.dot(beta))

        w = 1 / (np.exp(log_sigma_mu_squared) + 1 / h_sum[:, None])
        assert isinstance(w, np.ndarray)
        assert isinstance(bar_bar_err, np.ndarray)
        grad_lambda = -(w * x_bar_bar).T.dot(bar_bar_err)
        grad_alpha = -bar_bar_err.dot(w)

        gradient = np.concatenate((gradient, grad_beta, grad_lambda, grad_alpha))

        return ll, gradient

    def update_variances(self):
        def get_ll_helper(params: np.ndarray):
            return self.get_ll_grad(*params, get_grad=False)

        def get_grad_helper(params: np.ndarray):
            _, grad = self.get_ll_grad(*params, get_grad=True, variances_only=True)
            return grad

        bounds = [-np.inf, np.log(np.var(self.y_tilde))]
        # bounds = np.var(self.y_tilde) * 1e-7, np.var(self.y_tilde)

        result = minimize(get_ll_helper,
                          np.log(np.array([self.sigma_mu_squared, self.sigma_theta_squared,
                                           self.sigma_epsilon_squared])),
                          jac=get_grad_helper, method='L-BFGS-B', bounds=[bounds, bounds, bounds],
                          options={'disp': False, 'ftol': 1e-14, 'gtol': 1e-7})
        self.sigma_mu_squared, self.sigma_theta_squared, self.sigma_epsilon_squared = np.exp(result['x'])
        return

    def update_coefficients(self):
        """
        Resets beta, lambda, alpha, precisions, and weighted means; keeps variances constant.
        :return:
        """
        self.h = 1 / (self.sigma_theta_squared + self.sigma_epsilon_squared / self.n_students_per_class)
        h_sum_long = self.teacher_grouped.apply(np.sum, self.h)[:, 0]
        self.h_sum = h_sum_long[self.teacher_grouped.first_occurrences]
        # For beta
        precision_weights = self.h / h_sum_long
        self.y_bar_bar_long = self.teacher_grouped.apply(np.sum, precision_weights * self.y_bar)[:, 0]
        self.x_bar_bar_long = self.teacher_grouped.apply(lambda x: np.sum(x, 0),
                                                         precision_weights[:, None] * self.x_bar,
                                                         width=self.x_bar.shape[1])
        self.y_bar_tilde = self.y_bar - self.y_bar_bar_long
        self.x_bar_tilde = self.x_bar - self.x_bar_bar_long
        x_mat = self.xx_tilde / self.sigma_epsilon_squared + self.x_bar_tilde.T.dot(self.x_bar_tilde * self.h[:, None])
        y_mat = self.xy_tilde / self.sigma_epsilon_squared + self.x_bar_tilde.T.dot(self.y_bar_tilde * self.h)
        self.beta = np.linalg.solve(x_mat, y_mat)
        # Now get beta + lambda
        y_bar_bar = self.y_bar_bar_long[self.teacher_grouped.first_occurrences]
        teacher_precision_sums = h_sum_long[self.teacher_grouped.first_occurrences]
        sqrt_weights = 1 / np.sqrt(1 / teacher_precision_sums + self.sigma_mu_squared)
        assert np.all(np.isfinite(sqrt_weights))
        x_bar_bar = self.x_bar_bar_long[self.teacher_grouped.first_occurrences, :]
        y_w = (y_bar_bar - x_bar_bar.dot(self.beta)) * sqrt_weights
        stacked = np.hstack((np.ones((self.n_teachers, 1)), x_bar_bar))
        x_w = stacked * sqrt_weights[:, None]
        lambda_, _, rank, _ = np.linalg.lstsq(x_w, y_w)
        if rank != x_w.shape[1]:
            warnings.warn('x_w is rank deficient')
        self.alpha = lambda_[0]
        self.lambda_ = lambda_[1:]
        return

    def get_hess(self, epsilon: float):
        k = len(self.beta)
        hessian = np.zeros((2 * k + 4, 2 * k + 4))

        ll_up, upper = self.get_ll_grad(np.log(self.sigma_mu_squared) + epsilon, get_grad=True)
        ll_lo, lower = self.get_ll_grad(np.log(self.sigma_mu_squared) - epsilon, get_grad=True)
        hessian[0, :] = (upper - lower) / (2 * epsilon)

        ll_up, upper = self.get_ll_grad(log_sigma_theta_squared=np.log(self.sigma_theta_squared) + epsilon,
                                        get_grad=True)
        ll_lo, lower = self.get_ll_grad(log_sigma_theta_squared=np.log(self.sigma_theta_squared) - epsilon,
                                        get_grad=True)
        hessian[1, :] = (upper - lower) / (2 * epsilon)

        ll_up, upper = self.get_ll_grad(log_sigma_epsilon_squared=np.log(self.sigma_epsilon_squared) + epsilon,
                                        get_grad=True)
        ll_lo, lower = self.get_ll_grad(log_sigma_epsilon_squared=np.log(self.sigma_epsilon_squared) - epsilon,
                                        get_grad=True)
        hessian[2, :] = (upper - lower) / (2 * epsilon)

        eye = np.eye(len(self.beta))

        for i in range(k):
            ll_up, upper = self.get_ll_grad(beta=self.beta + epsilon * eye[i], get_grad=True)
            ll_lo, lower = self.get_ll_grad(beta=self.beta - epsilon * eye[i], get_grad=True)
            hessian[2 + i, :] = (upper - lower) / (2 * epsilon)

        for i in range(k):
            ll_up, upper = self.get_ll_grad(lambda_=self.lambda_ + epsilon * eye[i], get_grad=True)
            ll_lo, lower = self.get_ll_grad(lambda_=self.lambda_ - epsilon * eye[i], get_grad=True)
            hessian[2 + k + i, :] = (upper - lower) / (2 * epsilon)

        # alpha
        ll_up, upper = self.get_ll_grad(alpha=self.alpha + epsilon, get_grad=True)
        ll_lo, lower = self.get_ll_grad(alpha=self.alpha - epsilon, get_grad=True)
        hessian[-1, :] = (upper - lower) / (2 * epsilon)

        hessian += hessian.T
        hessian /= 2
        return hessian

    def fit(self):
        max_diff = 10
        i = 0
        while abs(max_diff) > 1e-7 and i < 30:
            beta_old = self.beta.copy()
            lambda_old = self.lambda_.copy()
            sigma_mu_squared_old = self.sigma_mu_squared
            sigma_theta_squared_old = self.sigma_theta_squared
            sigma_epsilon_squared_old = self.sigma_epsilon_squared
            self.update_coefficients()
            self.update_variances()
            differences = np.array([np.max(np.abs(self.beta - beta_old)),
                                    np.max(np.abs(self.lambda_ - lambda_old)),
                                    abs(self.sigma_mu_squared - sigma_mu_squared_old),
                                    abs(self.sigma_theta_squared - sigma_theta_squared_old),
                                    abs(self.sigma_epsilon_squared - sigma_epsilon_squared_old)])
            max_diff = np.max(differences)
            i += 1
        print('Number of tries', i)
        print('variances', self.sigma_mu_squared, self.sigma_theta_squared, self.sigma_epsilon_squared)
        self.update_coefficients()
        self.hessian = self.get_hess(1e-6)
        print('Number of zeros in hessian', np.sum(self.hessian == 0))
        is_hessian_invertible = True
        try:
            self.asymp_var = np.linalg.inv(self.hessian)
            if np.any(np.diag(self.asymp_var) <= 0):
                warnings.warn('Some variables will have negative or zero variance')
        except np.linalg.LinAlgError:
            warnings.warn('Hessian was not invertible')
            is_hessian_invertible = False

        lambda_idx = slice(-1 - len(self.beta), -1)
        x_bar_bar = self.x_bar_bar_long[self.teacher_grouped.first_occurrences]
        self.predictable_var = np.var(x_bar_bar.dot(self.lambda_))
        x_bar_bar_demeaned = x_bar_bar - np.mean(x_bar_bar, 0)

        if is_hessian_invertible:
            var_lambda = self.asymp_var[lambda_idx, lambda_idx]
            try:
                self.bias_correction = np.sum(x_bar_bar_demeaned.dot(np.linalg.cholesky(var_lambda))**2)\
                    / (self.n_teachers - 1)
                print('bias correction', self.bias_correction)
            except np.linalg.LinAlgError:
                warnings.warn('Hessian was not positive definite')
                self.bias_correction = np.sum([row.T.dot(var_lambda).dot(row)
                                               for row in x_bar_bar_demeaned]) / (self.n_teachers - 1)

        if is_hessian_invertible:
            self.total_var = self.predictable_var - self.bias_correction + self.sigma_mu_squared
            # Delta method
            grad = np.zeros(self.asymp_var.shape[0])
            grad[0] = self.sigma_mu_squared
            grad[lambda_idx] = 2 * np.mean(x_bar_bar.dot(self.lambda_)[:, None] * x_bar_bar_demeaned, 0)
            self.total_var_se = np.sqrt(grad.T.dot(self.asymp_var).dot(grad))
            self.sigma_mu_squared_se = np.sqrt(self.asymp_var[0, 0]) * self.sigma_mu_squared

        if not self.moments_only:
            rho = self.sigma_mu_squared / (self.sigma_mu_squared + 1 / self.h_sum)
            residual = self.y_bar_bar_long[self.teacher_grouped.first_occurrences] - \
                x_bar_bar.dot(self.beta) - self.alpha
            predicted = x_bar_bar.dot(self.lambda_)
            self.individual_scores = (1 - rho) * residual + rho * predicted
        return self
