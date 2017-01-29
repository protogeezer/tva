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
    assert teacher in data.columns
    n_teachers = len(set(data[teacher]))
    # if teacher is not already categorical, recode it
    assert np.max(data[teacher]) + 1 == n_teachers
    
    # Get preliminary estimates of bureaucrat effects through OLS
    # And variance
    teacher_dummies = sps.csc_matrix((np.ones(n_teachers),
        (range(n_teachers), data[teachers])))
    # TODO: incorporate categorical controls
    exog = sps.hstack((teacher_dummies, data[covariates]))
    model = sm.OLS(data[outcome], exog)
    results = model.fit()
    mu_preliminary = results.beta[:n_teachers]
    V_hat = results.cov_HC0
    assert V_hat.shape[0] == n_teachers
    # assert that variance is diagonal
    assert V_hat.nnz == n_teachers
    # Since variance is diagonal, only keep the diagonal
    V_hat = np.diagonal(V_hat)

    f = lambda x: get_ll_grad_hess(x, mu_hat, V)[0]
    g = lambda x: get_ll_grad_hess(x, mu_hat, V)[1]
    h = lambda x: get_ll_grad_hess(x, mu_hat, V)[2]

    sigma_mu_squared = newton(g, 1, fprime=h)
     
    # TODO: individual estimates
    # TODO: standard errors

