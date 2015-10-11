def simulate_two_types(sd_epsilon, beta, num_teachers, parameters, parallel=False):
    import numpy as np
    import random
    import pandas as pd
    import multiprocessing

    def generate_one_teacher_data(input_tuple):
        teacher_id, params = input_tuple
        teacher_df = []
        mu = np.random.multivariate_normal([0,0], params['cov mu'])
        for class_id in range(max(1, np.random.poisson(params['mean classes taught']))):
            theta = np.random.multivariate_normal([0,0], params['cov theta'])
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
        teacher_df.loc[:, 'true va 0'] = mu[0]
        teacher_df.loc[:, 'true va 1'] = mu[1]
        teacher_df.loc[:, 'score'] = teacher_df['score'] + [mu[t] for t in teacher_df['type'].values]
        
        return teacher_df
        
    if parallel:
        num_cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(num_cores)
        dfs = pool.map(generate_one_teacher_data, [(i, parameters) for i in range(num_teachers)])
        pool.close()
        pool.join()
    else:
        dfs = [generate_one_teacher_data((i, parameters)) for i in range(num_teachers)]

    dfs = pd.concat(dfs)
    dfs['x1'] = [np.random.normal() for i in range(len(dfs))]
    dfs['x2'] = [np.random.normal() for i in range(len(dfs))]
    dfs['score'] = dfs['score'] + np.dot(dfs[['x1', 'x2']].as_matrix(), beta) + np.random.normal(0, sd_epsilon, len(dfs))

    return dfs

if __name__ == "__main__":
    sd_epsilon = .2455**.5
    beta = [2, 3]
    num_teachers = 10000
    corr_mu = .7
    corr_theta = -.9
    parameters = {'cov mu':[[.018, corr_mu*(.018*.012)**(.5)], [corr_mu*(.018*.012)**(.5), .012]], 'cov theta':[[.03, corr_theta*(.03*.02)**.5], [corr_theta*(.03*.02)**.5, .02]], 'mean class size':24, 'mean classes taught':3}
    
    simulate_two_types(sd_epsilon, beta, num_teachers, parameters).to_csv('two_type_simulated_data.csv', parallel=True)
