from math import sqrt
import numpy as np
import random
import pandas as pd
import multiprocessing

num_teachers = 10000
beta = [2, 3]
sd_epsilon = sqrt(.2455)
# approximate mean classes taught; zero is not allowed
parameters = {'sd mu':sqrt(.0135), 'sd theta':sqrt(.0295), 'mean class size':20, 'mean classes taught':3}

def generate_one_teacher_data(input_tuple):   
    teacher_id, params = input_tuple   
    teacher_df = []
    mu = np.random.normal(0, params['sd mu'])
    for class_id in range(max(1, np.random.poisson(params['mean classes taught']))):
        theta = np.random.normal(0, params['sd theta'])
        class_size = round(np.random.poisson(params['mean class size']))        
        class_df = pd.DataFrame()  
           
        class_df['student id'] = range(class_size)
        class_df['class id'] = class_id
        class_df['score'] = theta
    
        teacher_df.append(class_df)
        
    teacher_df = pd.concat(teacher_df)
    teacher_df['teacher'] = teacher_id
    teacher_df['true va'] = mu
    teacher_df['score'] = teacher_df['score'] + mu
        
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
