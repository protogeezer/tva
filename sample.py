import pandas as pd

data = pd.read_csv('myData.csv')
covariates = ['previous score', 'age']
use_jackknife = True

va_data, var_mu, var_theta, var_epsilon, CI = \
                                    estimate_va(data, covariates, use_jackknife)
