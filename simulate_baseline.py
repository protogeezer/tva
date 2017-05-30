import numpy as np
import pandas as pd


def simulate(params, seed):
    np.random.seed(seed)
    num_teachers, beta = params['num teachers'], params['beta']
    sd_epsilon, sd_mu, sd_theta = params['sd epsilon'], params['sd mu'], params['sd theta']
    mean_class_size, mean_classes_taught = params['mean class size'], params['mean classes taught']
    lambda_ = np.array([0, 1])
    
    def simulate_one_teacher(teacher_id):
        classes = range(max(1, np.random.poisson(mean_classes_taught)))
        sizes = np.random.poisson(mean_class_size, len(classes))
        
        class_id_vector = np.hstack((np.tile(class_id, size) 
                                    for class_id, size in zip(classes, sizes)))
        # teacher, class id, student id, true va, score, x1, x2
        # 0     , 1         , 2         , 3     , 4     , 5, 6
        data = np.empty((len(class_id_vector), 8))
        data[:, 0] = teacher_id
        data[:, 1] = class_id_vector
        data[:, 2] = np.hstack((range(s) for s in sizes))+100*class_id_vector+10000*teacher_id
        data[:, 5:7] = np.random.normal(0, 1, (len(data), 2))
        mu = np.random.normal(np.mean(data[:, 5:7], 0).dot(lambda_), sd_mu)
        data[:, 3] = mu
        theta_vector = np.hstack((np.tile(np.random.normal(0, sd_theta), s)
                                  for s in  sizes))
        data[:, 4] = mu + theta_vector
        
        return data
        
    data = np.vstack((simulate_one_teacher(i) for i in range(num_teachers)))
    # data[:, 5] = np.random.normal(0, 1, len(data))
    # data[:, 6] = data[:, 3] + np.random.normal(0, 1, len(data))
    # Create categorical data
    categorical = np.random.choice(range(5), len(data))
    data[:, 7] = categorical
    #fixed_effects = np.random.normal(0, 1, 5)
    fixed_effects = np.zeros(5)
    data[:, 4] += np.random.normal(0, sd_epsilon, len(data)) \
                 + np.dot(data[:, 5:7], beta)\
                 + fixed_effects[categorical]
    data = pd.DataFrame(data, 
                        columns=('teacher', 'class id', 'student id', 
                                 'true va', 'score', 'x1', 'x2', 'c1'))
    for col in ['student id', 'teacher', 'class id', 'c1']:
        data[col] = data[col].astype(int)
    return data

    
if __name__=="__main__":
    simulate({'num teachers':1000, 'beta':[2, 3], 'sd mu':.024**.5, 'sd theta':.178**.5, 
              'sd epsilon':(1-.024-.178)**.5, 'mean class size':20, 'mean classes taught':3},
              2).to_csv('monte_carlo_data.csv', index=False)
