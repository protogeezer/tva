import numpy as np
import time
from config import *
import sys
sys.path += [hdfe_dir]
from hdfe import Groupby
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

sigma_theta = 2
n_classes_per_teacher = 2
n_teachers = 3
n_students = 110

np.random.seed(52617)
theta = np.random.normal(0, sigma_theta, n_classes_per_teacher * n_teachers)
teacher_ids = np.random.choice(range(n_teachers), n_students)
class_ids = np.random.choice(range(n_classes_per_teacher), n_students)\
            + teacher_ids * n_classes_per_teacher
mu = np.random.normal(0, 1, n_teachers)
sigma_epsilon = 3
y =  theta[class_ids] + mu[teacher_ids] + np.random.normal(0, sigma_epsilon, n_students)

# Hyperparamters to try
sigma_epsilon = [7, 4]
sigma_theta = [1, 3]
sigma_mu = [3, 5]

# With sigma_epsilon of 1
def get_ll_one_class(y,  mu, sigma_epsilon, sigma_theta):
    y_bar = np.mean(y)
    y_tilde = y - y_bar
    sqrt_ssr = np.sqrt(np.sum(y_tilde**2))

    ll = (2 - len(y)) * np.log(sigma_epsilon)\
         + norm.logpdf(np.sqrt(np.sum(y_tilde**2)), 0, sigma_epsilon)

    sigma_epsilon_over_n = sigma_epsilon / np.sqrt(len(y))
    theta_samples = np.random.normal(0, sigma_theta, 10**4)
    return ll + np.log(np.mean(norm.pdf(y_bar - mu - theta_samples, 0, sigma_epsilon_over_n)))


# As a function of mu, which needs to be integrated out
def get_ll(y, class_ids, mu, sigma_epsilon, sigma_theta):
    ll = 0
    for i in set(class_ids):
        idx = class_ids == i
        ll += get_ll_one_class(y[idx], mu, sigma_epsilon, sigma_theta)
    return ll

def get_ll_teacher(y, class_ids, sigma_mu, sigma_epsilon, sigma_theta):
    mu_samples = np.random.normal(0, sigma_mu, 2 * 10**4)
    return np.log(np.mean([np.exp(get_ll(y, class_ids, mu, sigma_epsilon, sigma_theta))
                          for mu in mu_samples]))


def get_ll_total(y, teacher_idx, class_idx, sigma_mu, sigma_epsilon,
                 sigma_theta):
    ll = 0
    for i in set(teacher_ids):
        idx = teacher_ids == i
        ll += get_ll_teacher(y[idx], class_ids[idx], sigma_mu, sigma_epsilon,
                             sigma_theta)
    return ll

def get_ll_two_teacher(y, class_ids, sigma_mu, sigma_epsilon, sigma_theta):
    grouped = Groupby(class_ids)
    n_students = grouped.apply(len, y, broadcast=False)
    h = 1 / (sigma_epsilon**2 / n_students + sigma_theta**2)

    y_bar = grouped.apply(np.mean, y, broadcast=False)
    y_bar_bar = y_bar.dot(h) / np.sum(h)
    y_bar_tilde = y_bar - y_bar_bar

    return -.5 * h.dot(y_bar_tilde**2)


def get_ll_two_total(y, teacher_ids, class_ids, sigma_mu, sigma_epsilon,
                     sigma_theta):
    # TODO: make sure class_ids are unique within teacher
    class_grouped = Groupby(class_ids)
    class_ids_unique = class_grouped.apply(lambda x: len(set(x[:, 0])) == 1, 
                                           teacher_ids)
    assert class_ids_unique.all()
    n_classes = class_grouped.n_keys
    y_tilde = class_grouped.apply(lambda x: x - np.mean(x), y)
    n_students = class_grouped.apply(len, y, broadcast=False)
    h = 1 / (sigma_epsilon**2 / n_students + sigma_theta**2)

    teacher_grouped = Groupby(teacher_ids[class_grouped.first_occurrences])
    h_sum_long = teacher_grouped.apply(np.sum, h)
    h_sum = h_sum_long[teacher_grouped.first_occurrences]

    y_bar = class_grouped.apply(np.mean, y, broadcast=False)
    y_bar_bar = teacher_grouped.apply(np.sum, y_bar * h) / h_sum_long
    y_bar_tilde = y_bar - y_bar_bar[:,0]

    ll = (n_classes - len(y)) * np.log(sigma_epsilon)\
         - np.sum(y_tilde**2) / (2 * sigma_epsilon**2)\
         - .5 * np.sum(np.log(sigma_mu**2 + 1 / h_sum))\
         + .5 * np.sum(np.log(h)) - .5 * np.sum(np.log(h_sum))
    ll -= .5 * np.sum(y_bar_bar[teacher_grouped.first_occurrences]**2 / (sigma_mu**2 + 1 / h_sum))
    ll -= .5 * h.dot(y_bar_tilde**2)
    return ll

start = time.clock()
ll_two = [get_ll_two_total(y, teacher_ids,  class_ids, sigma_mu[i], 
                           sigma_epsilon[i], sigma_theta[i]) 
          for i in [0, 1]]
print(time.clock() - start)

# start = time.clock()
# ll_one = [get_ll_total(y, teacher_ids, class_ids, sigma_mu[i], sigma_epsilon[i], 
#                        sigma_theta[i]) 
#          for i in [0, 1]]
# print(time.clock() - start)

print(ll_one[0] - ll_two[0] - (ll_one[1] - ll_two[1]))

