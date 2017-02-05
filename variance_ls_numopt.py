"""
Functions to solve maximum likelihood problem
mu_hat_j | mu_j, V_j ~ N(mu_j, V_j)
mu_j | sigma ~ N(0, sigma^2)
Optimize over sigma.

As in Fessler and Kasy (2016), but assume V is diagonal
"""
import numpy as np
from scipy.optimize import check_grad, newton


def get_ll(sigma_squared, mu, V):
    sigma_squared_plus_V = sigma_squared + V
    return np.sum(np.log(sigma_squared_plus_V)) + np.sum(mu**2 / sigma_squared_plus_V)


def get_grad(sigma_squared, mu, V):
    sigma_squared_plus_V = sigma_squared + V
    return np.sum(1 / sigma_squared_plus_V) - np.sum(mu**2 / sigma_squared_plus_V**2)


def get_hessian(sigma_squared, mu, V):
    sigma_squared_plus_V = sigma_squared + V
    return -1 * np.sum((1 / sigma_squared_plus_V)**2) + 2 * np.sum(mu**2 / sigma_squared_plus_V**3)

# TODO: incorporate delta
def get_ll_grad_hess(sigma_squared, mu_hat, V):
    J = len(mu_hat)
    assert J == len(V)
    mu_squared = mu_hat**2
    sigma_squared_plus_V = sigma_squared + V
    
    log_like = -.5 * np.sum(np.log(sigma_squared_plus_V)) \
            - .5 * np.sum(mu_squared / sigma_squared_plus_V)
    grad = -.5 * np.sum(1 / sigma_squared_plus_V) + \
            .5 * np.sum(mu_squared / sigma_squared_plus_V**2)
    hessian = .5 * np.sum(1 / sigma_squared_plus_V**2) - \
            np.sum(mu_squared / sigma_squared_plus_V**3)

    return log_like, grad, hessian


# Check numerically that this works
# Cool it works
if __name__ == '__main__':
    # Generate Monte Carlo data
    sigma_squared = 2
    J = 1000

    answer = []
    for i in range(100):
        mu = np.random.normal(0, sigma_squared**.5, J)
        V = np.exp(np.random.normal(0, 1, J))
        mu_hat = np.random.normal(mu, V**.5)

        f = lambda x: get_ll_grad_hess(x, mu_hat, V)[0]
        g = lambda x: get_ll_grad_hess(x, mu_hat, V)[1]
        h = lambda x: get_ll_grad_hess(x, mu_hat, V)[2]

      #  result = check_grad(f, g, np.array([1]))
      #  print('Derivative check on gradient ', result)


      #  result_2 = check_grad(g, h, np.array([1]))
      #  print('Derivative check on Hessian ', result_2)

        # now try to recover parameters 
        answer.append(newton(g, 1, fprime=h))

    answer = np.sort(np.array(answer))
    
