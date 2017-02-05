"""
This file implements a value-added estimator inspired by
Fessler and Kasy 2016, "How to Use Economic Theory to Improve Estimators." Documentation available via pdf.
I believe something similar has been used in a 
different value-added paper.
"""

import pandas as pd
import numpy as np
import scipy.sparse as sps
from scipy.optimize import minimize
from variance_ls_numopt import get_ll, get_grad, get_hessian
import statsmodels.api as sm
import time


def calculate_va(data, covariates, jackknife, outcome='score', teacher='teacher', categorical_controls=None):
    # Not implemented yet
    assert categorical_controls is None
    assert teacher in data.columns
    n_teachers = len(set(data[teacher]))
    # if teacher is not already categorical, recode it
    try:
        assert np.max(data[teacher]) + 1 == n_teachers
    except AssertionError:
        print('n teachers ', n_teachers)
        print(np.max(data[teacher]))
        assert False
    n = len(data)
    
    # Get preliminary estimates of bureaucrat effects through OLS
    # And variance
    teacher_dummies = sps.csc_matrix((np.ones(n),
        (range(n), data[teacher])))

    # TODO: incorporate categorical controls
    x = sps.hstack((teacher_dummies, data[covariates]))
    b = sps.linalg.lsqr(x, data[outcome].values)[0]

    
    errors = data[outcome] - x.dot(b)
    # (x'x) as vector rather than diagonal matrix
    x_prime_x = teacher_dummies.multiply(teacher_dummies).T.dot(np.ones(n))

    V = errors.dot(errors) / (x_prime_x * (n - n_teachers - len(covariates)))
    assert V.ndim == 1

    mu_preliminary = b[:n_teachers]

    objfun = lambda x: get_ll(x, mu_preliminary, V)
    g = lambda x: get_grad(x, mu_preliminary, V)
    h = lambda x: get_hessian(x, mu_preliminary, V)

    sigma_mu_squared = minimize(objfun, 1, method='Newton-CG', jac=g, hess=h).x
   
    # TODO: individual estimates
    # TODO: standard errors
    return sigma_mu_squared

