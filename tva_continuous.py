import pandas as pd
from calculate_va_continuous import calculate_va_continuous
import statsmodels.api as sm

filename = 'data_continuous.csv'

data = pd.read_csv(filename, sep=',')

data['year'] = data['class id']

class_df, var_mu_0, var_mu_1, var_theta, var_epsilon = calculate_va_continuous(data, [], residual = 'score')
print 'var mu 0 ', var_mu_0
print 'var mu 1 ', var_mu_1
print 'var theta ', var_theta
print 'var_epsilon ', var_epsilon

print '\n correct answers:'
print 'var mu 0 ', 1
print 'var mu 1 ', 1
print 'var theta ', .03
print 'var_epsilon ', .25

### Merge back into student-level data to test

data = pd.merge(data, class_df)
data['interact'] = data['continuous var'] * data['mu_1']

lhs_variables = ['mu_0', 'mu_1', 'continuous var', 'interact']

result = sm.OLS(data['score'], data[lhs_variables], hasconst = True, missing = 'drop')
result = result.fit()
print result.summary()
