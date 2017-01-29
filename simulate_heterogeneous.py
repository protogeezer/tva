from math import sqrt
import numpy as np
import random
import pandas as pd
import multiprocessing
import time

#### define all parameters
var_epsilon = .25
beta = [2, -1]
num_teachers = 10000
parameters = {'cov mu':[[.7, .2], [.2, .4]], 'cov theta':cov_theta = [[.5, -.1], [-.1, .6]], 'mean class size':24, 'mean classes taught':3}

def generate_one_teacher_data(input_tuple):
    teacher_id, params = input_tuple
    teacher_df = []
    mu = np.random.multivariate_normal(0, params['cov mu'])
    for class_id in range(max(1, np.random.poisson(params['mean classes taught']))):
        theta = np.random.multivariate_normal(0, params['cov theta'])
        class_df = pd.DataFrame()
        class_size = np.random.poisson(params['mean class size'])
        class_df.loc[:, 'student id'] = range(class_size)
        class_df.loc[:, 'class id'] = class_id
        
        type_ = [int(random.random() < .5) for i in range(class_size)]
        class_df.loc[:, 'type'] = type_
        class_df.loc[:, 'score'] = [theta[t] for t in type_]
        
        teacher_df.append(class_df)
        
    teacher_df = pd.concat(teacher_df)
    teacher_df.loc[:, 'teacher'] = teacher_id
    teacher_df.loc[:, 'true va'] = mu
    teacher_df.loc[:, score] = teacher_df['score'] + [mu[t] for t in type_]
    
    return teacher_df

    
start = time.time()
num_cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(num_cores)
dfs = pool.map(generate_one_teacher_data, [(i, parameters) for i in range(num_teachers)])
pool.close()
pool.join()

dfs = pd.concat(dfs)
dfs['x1'] = [np.random.normal() for i in range(len(dfs))]
dfs['x2'] = [np.random.normal() for i in range(len(dfs))]
dfs['score'] = dfs['score'] + np.dot(dfs[['x1', 'x2']].as_matrix(), beta) + np.random.normal(0, sd_epsilon, len(dfs))

dfs.to_csv('baseline_simulated_data.csv')
print(str(time.time() - start))
