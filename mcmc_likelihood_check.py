import numpy as np
import time
from config import *
import sys
sys.path += [hdfe_dir]
from hdfe import Groupby
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

sigma_theta = 2
n_classes = 2
n_students = 110
np.random.seed(52617)
theta = np.random.normal(0, sigma_theta, n_classes)
class_ids = np.random.choice(range(2), n_students)
mu = 1
sigma_epsilon = 3
y = np.random.normal(mu, sigma_epsilon, n_students) + theta[class_ids]

# Hyperparamters to try
mu = [3, -3]
sigma_epsilon = [7, 4]
sigma_theta = [1, 3]
sigma_mu = [3, 5]


# f and g are equivalent
def f(y, sigma_epsilon):
    return np.sum(norm.logpdf(y, 0, sigma_epsilon))

def g(y, sigma_epsilon):
    return norm.logpdf(np.sqrt(np.sum(y**2)), 0, sigma_epsilon) \
            + (1 - len(y)) * np.log(sigma_epsilon)

# f_1 and g_1 are equivalent
def f_1(y, sigma_epsilon, mu):
    y_bar = np.mean(y)
    y_tilde = y - y_bar
    return norm.logpdf(np.sqrt(np.sum(y_tilde**2)), 0, sigma_epsilon)\
            + (2 - len(y)) * np.log(sigma_epsilon)\
            + norm.logpdf(y_bar, mu, sigma_epsilon / np.sqrt(len(y)))

def g_1(y, sigma_epsilon, mu):
    return np.sum(norm.logpdf(y, mu, sigma_epsilon))

def test():
    one = f_1(y, 1, 3)
    two = f_1(y, 2, 4)
    three = g_1(y, 1, 3)
    four = g_1(y, 2, 4)
    return (one - two) - (three - four)

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


def get_ll(y, class_ids, mu, sigma_epsilon, sigma_theta):
    ll = 0
    for i in set(class_ids):
        idx = class_ids == i
        ll += get_ll_one_class(y[idx], mu, sigma_epsilon, sigma_theta)
    return ll

def get_ll_teacher(y, class_ids, sigma_mu, sigma_epsilon, sigma_theta):
    mu_samples = np.random.normal(0, sigma_mu, 10**4)
    return np.log(np.mean([np.exp(get_ll(y, class_ids, mu, sigma_epsilon, sigma_theta))
                          for mu in mu_samples]))

def get_ll_two(y, class_ids, sigma_mu, sigma_epsilon, sigma_theta):
    grouped = Groupby(class_ids)
    y_bar = grouped.apply(np.mean, y, broadcast=False)
    y_tilde = y - y_bar[class_ids]
    n_students = grouped.apply(len, y, broadcast=False)
    h = 1 / (sigma_epsilon**2 / n_students + sigma_theta**2)
    y_bar_bar = y_bar.dot(h) / np.sum(h)
    y_bar_tilde = y_bar - y_bar_bar


    ll = (1 + grouped.n_keys - len(y)) * np.log(sigma_epsilon)
    ll += norm.logpdf(np.sqrt(np.sum(y_tilde**2)), 0, sigma_epsilon)
    ll += .5 * np.sum(np.log(h)) - .5 * np.log(np.sum(h)) - .5 * h.dot(y_bar_tilde**2)
    ll += norm.logpdf(y_bar_bar, 0, np.sqrt(sigma_mu**2 + 1 / np.sum(h)))

    return ll

start = time.clock()
ll_one = [get_ll_teacher(y, class_ids, sigma_mu[i], sigma_epsilon[i], sigma_theta[i]) for i in [0, 1]]
print(time.clock() - start)
start = time.clock()
ll_two = [get_ll_two(y, class_ids, sigma_mu[i], sigma_epsilon[i], sigma_theta[i]) for i in [0, 1]]
print(time.clock() - start)

print(ll_one[0] - ll_two[0] - (ll_one[1] - ll_two[1]))

