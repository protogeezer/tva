from math import *
import numpy as np
import random
import statsmodels.api as sm
import pandas as pd

#### define all parameters

var_epsilon = .25

var_mu_0 = 1
var_mu_1 = var_mu_0
corr_mu = 0

var_theta = .03

class_size = 28
mean_classes_taught = 3
num_teachers = 1000

## continuous var
continuous_vari_mean = 0
continuous_vari_std = 1

## derived distributions
Sigma_mu = [[var_mu_0, corr_mu*sqrt(var_mu_0*var_mu_1)], \
            [corr_mu*sqrt(var_mu_0*var_mu_1), var_mu_1]]

dfs = []
# TODO: Check that mu's actually have the right covariance
for teacher in range(num_teachers):

    mu = np.random.multivariate_normal([0,0], Sigma_mu) 
       
    for class_id in range(np.random.poisson(mean_classes_taught)):
        theta = np.random.normal(0, var_theta**(.5))
        df = pd.DataFrame()
        
        df['student id'] = range(class_size)
        df['continuous var'] = np.random.normal(continuous_vari_mean, continuous_vari_std, class_size)
        df['class id'] = class_id
        df['teacher'] = teacher
        
        df['score'] = (mu[0] + theta) * np.ones(class_size) + mu[1]*df['continuous var'].values + np.random.normal(0, sqrt(var_epsilon), class_size)
    
        dfs.append(df)
    
dfs = pd.concat(dfs)
print dfs

dfs.to_csv('data_continuous.csv')
