"""
This file implements a value-added estimator inspired by
Fessler and Kasy 2016, "How to Use Economic Theory to Improve Estimators." Documentation available via pdf.
I believe something similar has been used in a 
different value-added paper.
"""

import pandas as pd
import numpy as np
import scipy.sparse as sps
from scipy.optimize import newton
from variance_ls_numopt import get_ll_grad_hess
import statsmodels.api as sm


def calculate_va(data, covariates, jackknife, outcome='score', teacher='teacher', categorical_controls=None):
    # Not implemented yet
    assert categorical_controls is None
    assert teacher in data.columns
    n_teachers = len(set(data[teacher]))
    # if teacher is not already categorical, recode it
    assert np.max(data[teacher]) + 1 == n_teachers
    n = len(data)
    
    # Get preliminary estimates of bureaucrat effects through OLS
    # And variance
    teacher_dummies = sps.csc_matrix((np.ones(n),
        (range(n), data[teacher])))
    assert(teacher_dummies.shape[0] == n)
    assert(teacher_dummies.shape[1] == n_teachers)

    # TODO: incorporate categorical controls
    x = sps.hstack((teacher_dummies, data[covariates]))
    b = sps.linalg.lsqr(x, data[outcome].values)[0]
    errors = data[outcome] - x.dot(b)
    # (x'x) as vector rather than diagonal matrix

    tmp = teacher_dummies.multiply(teacher_dummies).T.dot(np.ones(n))
    V = errors.dot(errors) * tmp / (n - n_teachers - len(covariates))
    assert V.ndim == 1

# Don't use robust standard errors because the relevant block is not diagonal
#    err = sps.csc_matrix((errors**2, (range(n), range(n))))
#    middle = (x.T.dot(err)).dot(x)
#    inv = sps.linalg.inv(x.T.dot(x))
#    V_hat = inv.dot(middle).dot(inv)

#    V_hat = V_hat[:n_teachers, :n_teachers]
    #assert V_hat.shape[0] == n_teachers
    # assert that variance is diagonal
    #assert V_hat.nnz == n_teachers
    # Since variance is diagonal, only keep the diagonal
    #V_hat = np.diagonal(V_hat)

    mu_preliminary = b[:n_teachers]

    f = lambda x: get_ll_grad_hess(x, mu_preliminary, V)[0]
    g = lambda x: get_ll_grad_hess(x, mu_preliminary, V)[1]
    h = lambda x: get_ll_grad_hess(x, mu_preliminary, V)[2]

    sigma_mu_squared = newton(g, 1, fprime=h)
     
    # TODO: individual estimates
    # TODO: standard errors
    return sigma_mu_squared

