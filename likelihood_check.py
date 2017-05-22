import numpy as np
from functools import reduce
import operator
from scipy.stats import norm

def likelihood_check(y, x, teacher_ids, class_ids, lambda_, beta, alpha, 
                     sigma_mu_sq, sigma_theta_sq, sigma_epsilon_sq, n):
    if n == 0:
        def inner_integral(mu, class_ids):
            def pdf(theta):
                individuals = norm.pdf(y[class_ids], x[class_ids, :].dot(beta) - mu - theta, sgima_epsilon_sq**.5)
                return norm.pdf(theta, 0, sigma_theta_sq**.5) * reduce(operator.mul, individuals)
           # now integrate pdf(theta) from -infinity to infinity 
        pass
    return ll
