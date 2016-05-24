import pandas as pd
import numpy as np
from va_alg_two_groups import *
import calculate_va_continuous
from va_functions import *
import matplotlib.pyplot as plt
import scipy.stats as stats
from simulate_two_types import simulate_two_types

    
def test_mu_covariances():
    n = 300
    var_mu = [.3, .2]
    cov_mu = .1
    cov_mat_mu = [[var_mu[0], cov_mu], [cov_mu, var_mu[1]]]
    
    # Simulate data
    df = pd.DataFrame()
    df.loc[:, 'teacher'] = [int(elt) for elt in np.arange(0,n*.25,.25)]
    df.loc[:, 'class id'] = [int(elt) for elt in np.arange(0,n*.5, .5)]
    df.loc[:, 'type'] = [int(i%2 == 0) for i in range(n)]
    df.loc[:, 'size'] = [int(elt) for elt in np.random.normal(8,3,n)]
    
    teachers = set(df['teacher'].values)
    type_0_indices = pd.Series(df['type'] == 0)
    
    for teacher in teachers:
        mu = np.random.multivariate_normal([0,0], cov_mat_mu)
        teacher_indices = pd.Series(df['teacher'] == teacher)
        df.loc[teacher_indices & type_0_indices, 'mean score'] = mu[0]
        df.loc[teacher_indices & ~type_0_indices, 'mean score'] = mu[1]
        
    teacher_class_map = get_teacher_class_map(df, teachers)
    
    var_mu_hat, cov_mu_hat = estimate_mu_covariances(df, teachers, teacher_class_map)

    tolerance = 2
    assert var_mu_hat[0] > var_mu[0] / tolerance
    assert var_mu_hat[0] < var_mu[0] * tolerance
    assert var_mu_hat[1] > var_mu[1] / tolerance
    assert var_mu_hat[1] < var_mu[1] * tolerance
    assert cov_mu_hat > cov_mu / tolerance
    assert cov_mu_hat < cov_mu * tolerance
    
def test_estimate_theta_covariance():
    n = 300
    cov_mu = .1
    cov_mat_mu = [[.3, cov_mu], [cov_mu, .2]]
    cov_theta = .3
    cov_mat_theta = [[.4, cov_theta], [cov_theta, .35]]
    
    df = pd.DataFrame()
    df.loc[:, 'teacher'] = [int(elt) for elt in np.arange(0,n*.25,.25)]
    df.loc[:, 'class id'] = [int(elt) for elt in np.arange(0,n*.5, .5)]
    df.loc[:, 'type'] = [int(i%2 == 0) for i in range(n)]
    df.loc[:, 'size'] = [int(elt) for elt in np.random.normal(8,3,n)]
    
    teachers = set(df['teacher'].values)
    teacher_class_map = {}
    
    type_0_indices = pd.Series(df['type'] == 0)
    
    for teacher in teachers:
        mu = np.random.multivariate_normal([0,0], cov_mat_mu)
        teacher_indices = pd.Series(df['teacher'] == teacher)
        
        teacher_class_map[teacher] = set(df.loc[teacher_indices, 'class id'].values)
        for class_id in teacher_class_map[teacher]:
            class_indices = pd.Series(df['class id'] == class_id)
            theta = np.random.multivariate_normal([0,0], cov_mat_theta)
        
            df.loc[teacher_indices & type_0_indices & class_indices, \
                                         'mean score'] = mu[0] + theta[0]
            df.loc[teacher_indices & ~type_0_indices & class_indices, \
                                         'mean score'] = mu[1] + theta[1]

    cov_theta_hat = estimate_theta_covariance(df, teachers, teacher_class_map, cov_mu,)

    tolerance = 2
    assert cov_theta_hat > cov_theta / tolerance
    assert cov_theta_hat < cov_theta * tolerance
    
def test_estimate_variance():
    n = 300
    variance = abs(np.random.normal())
    noise_stdev = abs(np.random.normal())
    n_classes_per_teacher = 4
    
    df = pd.DataFrame()
    df['teacher'] = [int(elt) for elt in np.arange(0, n*1.0/n_classes_per_teacher, 1./n_classes_per_teacher)]
    df['class id'] = np.tile(range(n_classes_per_teacher), n/n_classes_per_teacher)
    df['size'] = 10
    
    teachers = set(df['teacher'].values)
    
    for teacher in teachers:
        variable = np.random.normal(0, variance**(.5))
        for class_id in df[df['teacher'] == teacher]['class id'].values:
            df.loc[(df['teacher'] == teacher) & (df['class id'] == class_id), 'var'] = variable + np.random.normal(0, noise_stdev)

    variance_estimate = calculate_va_continuous.estimate_variance(df, teachers, 'var')
    tolerance = 2
    assert variance_estimate > variance / tolerance
    assert variance_estimate < variance * tolerance

def test_estimate_var_epsilon_one_class():
    df = pd.DataFrame()
    n = 100
    mu_1 = np.random.normal()
    constant = np.random.normal()
    var_epsilon = abs(np.random.normal())
    
    df['continuous var'] = np.random.normal(0, 1, n)
    df['residual'] = constant + mu_1 * df['continuous var'] + np.random.normal(0, var_epsilon**.5, n)
    
    var_epsilon_hat = calculate_va_continuous.estimate_var_epsilon_one_class(df, mu_1)
    tolerance = 2
    assert var_epsilon_hat > var_epsilon / tolerance
    assert var_epsilon_hat < var_epsilon * tolerance
    
def test_get_mc1():
    constant = np.random.normal()
    mu_1 = 1
    n = 20
    var_epsilon = 1
    num_classes = 100

    m_c1 = [calculate_va_continuous.get_mc1(construct_class_df(0, 0, n, 0, 0, mu_1, var_epsilon)) \
                                              for i in range(num_classes)]
    
    estimate = np.mean(m_c1)
    se = (np.var(m_c1) / num_classes)**(.5)
    
    assert estimate > mu_1 - 2*se
    assert estimate < mu_1 + 2*se
        

def construct_class_df(teacher, class_, n_students, std_theta, mu_0, mu_1, var_epsilon):
    theta = np.random.normal(0, std_theta) if std_theta > 0 else 0
    df = pd.DataFrame()
    df['student id'] = range(n_students)
    df['year'] = 0      
    df['teacher'] = teacher
    df['class id'] = class_
    df['continuous var'] = np.random.normal(0, 1, n_students)
    df['residual'] = theta + mu_0 + mu_1 * df['continuous var'] + np.random.normal(0, var_epsilon**(.5), n_students)
    return df


    
def test_residualize():
    N, T, beta = 1000, 3, [3, 4]
    tolerance = .01
    
    df = pd.DataFrame()
    fixed_effects = np.random.normal(0, 1, N)
    time_effects = np.random.normal(0, 1, T)
    df['i'] = [int(i / T) for i in range(N * T)]
    df['t'] = [t % T for t in range(N * T)]
    fixed_effect = np.array([fixed_effects[x] for x in df['i'].values])
    time_effect = np.array([time_effects[x] for x in df['t'].values])
    df['x1'] = fixed_effect * .2 + np.random.normal(0, 1, N * T)
    df['x2'] = np.random.normal(0, 1, N * T) - time_effect*.1
    df['y'] = np.dot(df[['x1', 'x2']], beta) + np.random.normal(0, .1, N*T) + fixed_effect + time_effect
    _, beta_hat = residualize(df, 'y', ['x1', 'x2'],  'i', 't')
    assert beta_hat[0] > beta[0] - tolerance
    assert beta_hat[0] < beta[0] + tolerance
    assert beta_hat[1] > beta[1] - tolerance
    assert beta_hat[1] < beta[1] + tolerance
    

def test_two_types_covariances():
    var_tolerance = .01
    corr_tolerance = .1
    corr_mu = .7
    parameters = {'cov mu':[[.018, corr_mu*(.018*.012)**.5], [corr_mu*(.018*.012)**.5, .012]], 'cov theta':[[0,0], [0, 0]], 'mean class size':24, 'mean classes taught':3}
    data = simulate_two_types(.01, [0, 0], 1000, parameters)
    data.loc[:, 'year'] = data['class id']
    moments, class_level_data = calculate_covariances(data, [])
    print(moments)
    assert moments['var mu'][0] > parameters['cov mu'][0][0] - var_tolerance
    assert moments['var mu'][0] < parameters['cov mu'][0][0] + var_tolerance
    assert moments['var mu'][1] > parameters['cov mu'][1][1] - var_tolerance
    assert moments['var mu'][1] < parameters['cov mu'][1][1] + var_tolerance
    assert moments['corr mu'] > corr_mu - corr_tolerance
    assert moments['corr mu'] < corr_mu + corr_tolerance
