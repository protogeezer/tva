import numpy as np
import pystan
import time
from config import *
import sys
sys.path += [hdfe_dir]
from hdfe import Groupby

def summary_stats(d):
    return {k: (np.mean(v), np.var(v)) for k, v in d.items()}

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

y = np.array([1, 2, 3, 4, 7])
class_id = np.array([0, 0, 0, 1, 1]) + 1
n_classes = len(set(class_id))

dat = {'n_students': len(y), 'y': y, 'class_id': class_id,
        'n_classes': n_classes}

start = time.time()
fit = pystan.stan(model_code=model_one, data=dat)
ext = fit.extract()
first_time = time.time() - start


dat_transformed = {'n_students': len(y),
                   'ssr': np.sqrt(np.sum((y - np.mean(y))**2)),
                   'y_bar': np.mean(y)}

start = time.time()
fit = pystan.stan(model_code=model_one_alt, data=dat_transformed)
ext_alt = fit.extract()
second_time = time.time() - start

print(summary_stats(ext))
print(summary_stats(ext_alt))

print('Time difference', second_time - first_time)

