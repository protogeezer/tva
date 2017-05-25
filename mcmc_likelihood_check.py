import numpy as np
import time
from config import *
import sys
sys.path += [hdfe_dir]
from hdfe import Groupby
from scipy.stats import norm

sigma_theta = 2
n_classes = 2
n_students = 11
np.random.seed(52617)
theta = np.random.normal(0, sigma_theta, n_classes)
class_ids = np.random.choice(range(2), 5)
mu = 1
sigma_epsilon = 3
y = np.random.normal(mu, sigma_epsilon, n_students) + theta[class_ids]

# Hyperparamters to try
mu = [3, 5]
sigma_epsilon = [1, 4]
sigma_theta = [1, 2]


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
    theta_samples = np.random.normal(0, sigma_theta, 10**6)
    return ll + np.log(np.mean(norm.pdf(y_bar - mu - theta_samples, 0, sigma_epsilon_over_n)))


def get_ll_one(y, class_ids, mu, sigma_epsilon, sigma_theta):
    ll = 0
    for i in set(class_ids):
        idx = class_ids == i
        ll += get_ll_one_class(y[idx], mu, sigma_epsilon, sigma_theta)
    return ll


def get_ll_two(y, class_ids, mu, sigma_epsilon, sigma_theta):
    y_bar = np.mean(y)
    y_tilde = y - y_bar
    sqrt_ssr = np.sqrt(np.sum(y_tilde**2))

    ll = (2 - len(y)) * np.log(sigma_epsilon) + norm.logpdf(sqrt_ssr, 0, sigma_epsilon)
    h = np.sqrt(sigma_epsilon**2 / len(y) + sigma_theta**2)
    return ll + norm.logpdf(y_bar, mu, h) 

start = time.clock()
ll_one = [get_ll(y, mu[i], sigma_epsilon[i], sigma_theta[i]) for i in [0, 1]]
print(time.clock() - start)
start = time.clock()
ll_two = [get_ll_two(y, mu[i], sigma_epsilon[i], sigma_theta[i]) for i in [0, 1]]
print(time.clock() - start)

print(ll_one[0] - ll_two[0] - (ll_one[1] - ll_two[1]))


# Model: y_i ~ N(mu, sigma_epsilon); set sigma_epsilon = 1
model_one = """
data {
    int<lower=1> n_students; // number of students
    real y[n_students];     // outcomes
    // int<lower=1> n_classes;
    // int<lower=1, upper=n_classes> class_id[n_students];
}
parameters {
    real mu;
    real<lower=0> sigma_epsilon;
    // real theta[n_classes];
    real theta;
}

model {
    theta ~ normal(0, 1);
    y ~ normal(mu + theta, sigma_epsilon);
    // for (i in 1:n_students)
    //     y[i] ~ normal(mu + theta[class_id[i]], sigma_epsilon);
}
"""

model_one_alt = """
data {
    int<lower=0> n_students; // number of students
    real ssr;
    real y_bar;
}
parameters {
    real mu;
    real<lower=0> sigma_epsilon;
}
transformed parameters {
    real<lower=0> sqrt_h;
    sqrt_h = sqrt(1 + square(sigma_epsilon) / n_students);
}
model {
    ssr ~ normal(0, sigma_epsilon);
    y_bar ~ normal(mu, sqrt_h);
    target += (3 - n_students) * log(sigma_epsilon) + log(1) - log(sqrt_h);
}
"""

