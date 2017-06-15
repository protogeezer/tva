"""
Functions to solve maximum likelihood problem
(mu_hat, gamma_hat) ~N ((0, gamma), V + ((tau^2 I, 0), (0, 0)))
Optimize over tau^2 and gamma.

As in Fessler and Kasy (2016)
"""
import numpy as np
from scipy.optimize import check_grad, minimize_scalar

# Scalar for now
# Better than Scipy's implementation because it has a (backtracking) line search
# And because it allows for objective function, grad, and hessian to come from same function
# This is a *minimization* problem
def newton(f, x):
    max_iter = 4
    abs_grad = 1
    i = 0
    # do at least 2 iterations, just in case gradient is scaled poorly
    while i < 3 or (abs_grad > 10**(-6) and i < max_iter):
        print('iter = ', i)
        print(x)
        
        obj_fun_old, grad, hess = f(x, True, True)
        if grad.shape == hess.shape:
            step = -1 * grad / hess
        else:
            step = -1 * np.linalg.lstsq(hess, grad)[0]
        obj_fun = f(x + step)
        # Line search
        n_tries = 0
        while obj_fun > obj_fun_old and n_tries < 20:
            step /= 2
            obj_fun = f(x + step)
            n_tries += 1
#            print('Obj fun val after ', n_tries, 'steps: ', obj_fun)
        if n_tries == 20:
            break

        x += step
        abs_grad = np.max(np.abs(grad))
        assert np.isscalar(abs_grad)
        i += 1
        
    return x, obj_fun


# concentrating out beta and gamma, ll only as a function of sigma_sq
def get_ll(mu_p, beta_p, v, teacher_controls, diag_mat, sigma_sq):
    Sigma = v + diag_mat * sigma_sq
    b = np.concatenate((mu_p, beta_p))
    n_teachers, k = teacher_controls.shape

    R = np.vstack((
            np.hstack((teacher_controls, np.zeros((n_teachers, len(beta_p))))),
            np.hstack((np.zeros((len(beta_p), k)), np.eye(len(beta_p))))
            ))
    
    Sigma_inv = np.linalg.inv(Sigma)
    P_R = R.dot(np.linalg.lstsq(R.T.dot(Sigma_inv).dot(R), R.T.dot(Sigma_inv))[0])
    assert P_R.shape[0] == P_R.shape[1]
    assert P_R.shape[0] == len(b)
    tmp = (np.eye(len(b)) - P_R).dot(b)
#    ll = np.log(np.linalg.det(Sigma)) + tmp.dot(Sigma_inv.dot(tmp.T))
    ll = np.linalg.slogdet(Sigma)[1] + tmp.dot(Sigma_inv.dot(tmp.T))
    return ll, Sigma_inv, b, R


def get_g_and_tau(mu_p, beta_p, v, teacher_controls, starting_guess=1):
    m, b = len(mu_p), len(beta_p)
    diag_mat = np.vstack((
                    np.hstack((np.eye(m), np.zeros((m, b)))),
                    np.hstack((np.zeros((b, m)), np.zeros((b, b))))
                    ))

    def f(sigma_sq):
        return get_ll(mu_p, beta_p, v, teacher_controls, diag_mat, sigma_sq)[0]

    sigma_mu_squared = minimize_scalar(f, bounds = [0, np.inf])['x']
    _, Sigma_inv, b, R = get_ll(mu_p, beta_p, v, teacher_controls, diag_mat, sigma_mu_squared)

    # recover other parameters

    b_hat = np.linalg.lstsq(R.T.dot(Sigma_inv.dot(R)), R.T.dot(Sigma_inv.dot(b)))[0]

    return sigma_mu_squared, b_hat[:-1*len(beta_p)], b_hat[-1*len(beta_p):]

# Check numerically that this works
# Cool it works
if __name__ == '__main__':
    # Generate Monte Carlo data
    # First, hyperparameters
    np.random.seed(4717)
    n_teacher_covariates = 5
    n_teachers = 100
    n_covariates = 4

    sigma_squared = 2
    beta = np.random.normal(0, 1, n_covariates)
    z = np.random.normal(0, 1, (n_teachers, n_teacher_covariates))
    gamma = np.random.normal(0, 1, n_teacher_covariates)
    tmp = np.random.normal(0, 1, (n_teachers + n_covariates, 2 * (n_teachers + n_covariates)))
    V = tmp.dot(tmp.T)
    mu = np.random.normal(z.dot(gamma), sigma_squared)
    b_hat = np.random.multivariate_normal(np.concatenate((mu, beta)), V)
    ans = get_g_and_tau(b_hat[:n_teachers], b_hat[n_teachers:], V, z)

    #sigma_squared_answer = []
    #gamma_answer = []
    #for i in range(100):
    #    gamma_hat = gamma
    #    mu = np.random.normal(0, sigma_squared**.5, (J, 1))
    #    prelim = np.random.multivariate_normal(np.vstack((mu, gamma))[:,0], V)
    #    sigma_squared_hat, gamma_est = get_g_and_tau(prelim[:J], prelim[J:], V)
    #    sigma_squared_answer.append(sigma_squared_hat)
    #    gamma_answer.append(gamma_est)

    #print(np.mean(sigma_squared_answer))
    #print(np.var(sigma_squared_answer))
    #print(np.mean(np.array(gamma_answer), 0))
