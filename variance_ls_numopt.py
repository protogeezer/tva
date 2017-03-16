"""
Functions to solve maximum likelihood problem
(mu_hat, gamma_hat) ~N ((0, gamma), V + ((tau^2 I, 0), (0, 0)))
Optimize over tau^2 and gamma.

As in Fessler and Kasy (2016)
"""
import numpy as np
from scipy.optimize import check_grad#, newton

# Scalar for now
# Better than Scipy's implementation because it has a (backtracking) line search
# And because it allows for objective function, grad, and hessian to come from same function
# This is a *minimization* problem
def newton(f, x):
    max_iter = 20
    abs_grad = 1
    i = 0
    # do at least 2 iterations, just in case gradient is scaled poorly
    while i < 3 or (abs_grad > 10**(-6) and i < max_iter):
        print('iter = ', i)
        print(x)
        
        obj_fun_old, grad, hess = f(x, True, True)
        step = -1 * grad / hess
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
        abs_grad = abs(grad)
        i += 1
        
    return x, obj_fun
        

def get_g_and_tau(mu_hat, g_hat, v, starting_guess=1):
    print('Size of v ', v.shape)
    
    if g_hat is None:
        e_values = np.linalg.eigvalsh(v)
        v_ = v

    else:
        J = v.shape[0] - len(g_hat)
        # Precompute stuff
        v_11 = v[:J, :J]
        v_12 = v[:J, J:]
        v_22 = v[J:, J:]
        
        schur = v_11 - v_12.dot(np.linalg.solve(v_22, v_12.T))
        e_values = np.linalg.eigvalsh(schur)
        v_ = v_11

    assert not np.any(e_values < 0)

    def get_ll_grad_hess(tau_squared, get_grad=False, get_hess=False):
        v_plus_tau_sq = v_ + np.eye(J) * tau_squared
        e_plus_tau_sq = e_values + tau_squared
        assert np.all(e_plus_tau_sq > 0)

        tmp = np.linalg.solve(v_plus_tau_sq, mu_hat)

        ll = np.sum(np.log(e_plus_tau_sq)) + mu_hat.T.dot(tmp)
        if get_grad or get_hess:
            grad = np.sum(1 / e_plus_tau_sq) - tmp.dot(tmp)
            if get_hess:
                hess = -1 * np.sum(1 / e_plus_tau_sq**2) \
                      + 2 * tmp.dot(np.linalg.solve(v_plus_tau_sq, tmp))

                return ll, grad, hess
            else:
                return ll, grad
        else:
            return ll

    tau_sq, ll = newton(get_ll_grad_hess, starting_guess)
    # check for corner solution
    #ll_0 = get_ll_grad_hess(0)
    #if ll_0 < ll:
    #    tau_sq = 0

    if g_hat is not None:
        g = g_hat - v_12.T.dot(np.linalg.solve(v_11 + tau_sq * np.eye(J), mu_hat))
        return tau_sq, g
    else:
        return tau_sq


# Check numerically that this works
# Cool it works
if __name__ == '__main__':
    # Generate Monte Carlo data
    sigma_squared = 2
    gamma = np.array([1, 2, 3])
    gamma.shape = (3, 1)
    k = len(gamma)
    J = 100
    x = np.random.normal(0, .01, (10,  J + k))
    V = np.eye(J + k) * np.exp(np.random.normal(0, 1, J+k))\
            + x.T.dot(x)

    sigma_squared_answer = []
    gamma_answer = []
    for i in range(100):
        gamma_hat = gamma
        mu = np.random.normal(0, sigma_squared**.5, (J, 1))
        prelim = np.random.multivariate_normal(np.vstack((mu, gamma))[:,0], V)
        sigma_squared_hat, gamma_est = get_g_and_tau(prelim[:J], prelim[J:], V)
        sigma_squared_answer.append(sigma_squared_hat)
        gamma_answer.append(gamma_est)

    print(np.mean(sigma_squared_answer))
    print(np.var(sigma_squared_answer))
    print(np.mean(np.array(gamma_answer), 0))
