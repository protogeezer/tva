from math import *
import numpy as np
import random
import statsmodels.api as sm
import pandas as pd

#### define all parameters

var_epsilon = .2455

var_mu_0 = .0135
var_mu_1 = var_mu_0
corr_mu = .8

var_theta_0 = .0295
var_theta_1 = var_theta_0
corr_theta = .9

class_size = 20
mean_classes_taught = 3
num_teachers = 10000

beta = [2, 3]

# derived distributions
Sigma_mu = [[var_mu_0, corr_mu*sqrt(var_mu_0*var_mu_1)], \
            [corr_mu*sqrt(var_mu_0*var_mu_1), var_mu_1]]
Sigma_theta = [[var_theta_0, corr_theta*sqrt(var_theta_0*var_theta_1)], \
               [corr_theta*sqrt(var_theta_0*var_theta_1), var_theta_1]]
     

dfs = []
# TODO: Check that mu's actually have the right covariance
for teacher in range(num_teachers):

    mu = np.random.multivariate_normal([0,0], Sigma_mu) 
       
    for class_id in range(np.random.poisson(mean_classes_taught)):
        theta = np.random.multivariate_normal([0,0], Sigma_theta)
        df = pd.DataFrame()
        
        df['student id'] = range(class_size)
        df['type'] = [int(random.random() < .5) for i in range(class_size)]
        df['class id'] = class_id
        df['teacher'] = teacher
        df['covariate 1'] = np.random.normal()
        df['covariate 2'] = np.random.normal()
        
        n_of_each_type = [class_size - np.sum(df['type'].values), np.sum(df['type'].values)]
        scores_by_type = [mu[0]+theta[0]+np.random.normal(0,sqrt(var_epsilon), n_of_each_type[i]) for i in [0,1]]
        
        df.loc[df['type'] == 0, 'score'] = scores_by_type[0]
        df.loc[df['type'] == 1, 'score'] = scores_by_type[1]
    
        dfs.append(df)
    
dfs = pd.concat(dfs)
print df

dfs.to_csv('data_I_made.csv')

