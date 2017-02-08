"""
Functions to solve maximum likelihood problem
(mu_hat, gamma_hat) ~N ((0, gamma), V + ((tau^2 I, 0), (0, 0)))
Optimize over tau^2 and gamma.

As in Fessler and Kasy (2016)
"""
import numpy as np
from scipy.optimize import check_grad, newton

def get_g_and_tau(mu_hat, g_hat, v):
    J = v.shape[0] - len(g_hat)
    # Precompute stuff
    v_11 = v[:J, :J]
    v_12 = v[:J, J:]
    v_22 = v[J:, J:]
    
    schur = v_11 - v_12.dot(np.linalg.lstsq(v_22, v_12.T)[0])
    e_values = np.linalg.eigvalsh(schur)

    def get_ll(tau_squared):
        return np.sum(np.log(e_values + tau_squared)) + \
                mu_hat.T.dot(np.linalg.lstsq(v_11 + np.eye(J) * tau_squared, mu_hat)[0])

    def get_grad(tau_squared):
        tmp = np.linalg.lstsq(v_11 + np.eye(J) * tau_squared, mu_hat)[0]
        return np.sum(1 / (e_values + tau_squared)) - tmp.dot(tmp)

    def get_hess(tau_squared):
        tmp = v_11 +  np.eye(J) * tau_squared
        tmp_2 = np.linalg.lstsq(tmp, mu_hat)[0]
        return -1 * np.sum( 1/ ((e_values + tau_squared)**2)) + 2 * tmp_2.dot(np.linalg.lstsq(tmp, tmp_2)[0])

#    # Derivative check
#    print(check_grad(get_ll, get_grad, [[1]]) / np.abs(get_grad(1)))
#    print(check_grad(get_grad, get_hess, [[1]]) / np.abs(get_hess(1)))

    tau_sq = newton(get_grad, 1, fprime = get_hess) 
    g = g_hat - v_12.T.dot(np.linalg.lstsq(v_11 + tau_sq * np.eye(J), mu_hat)[0])
    
    return tau_sq, g


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
