import numpy as np
import pystan
import time


# Model: y_i ~ N(mu, sigma_epsilon); set sigma_epsilon = 1
model_one = """
data {
    int<lower=0> n_students; // number of students
    real y[n_students];     // outcomes
}
parameters {
    real mu;
}
model {
    y ~ normal(mu, 1);
}
"""

model_one_alt = """
data {
    int<lower=0> n_students; // number of students
    real y_tilde[n_students];     // outcomes
    real y_bar;
}
parameters {
    real mu;
}
transformed parameters {
    real sigma_epsilon_over_n;
    sigma_epsilon_over_n = sqrt(1.0 / n_students);
}
model {
    y_tilde ~ normal(0, 1);
    y_bar ~ normal(mu, sigma_epsilon_over_n);
}
"""

y = np.array([1, 2, 3])

if True:
    dat = {'n_students': len(y), 'y': y}

    start = time.time()
    fit = pystan.stan(model_code=model_one, data=dat, chains=4)
    ext = fit.extract()
    print(time.time() - start)

    dat_transformed = {'n_students': len(y),
                       'y_tilde': y - np.mean(y),
                       'y_bar': np.mean(y)}

    fit = pystan.stan(model_code=model_one_alt, data=dat_transformed, chains=4)
    ext_alt = fit.extract()

    print('Mean difference', np.mean(ext['mu']) - np.mean(ext_alt['mu']))
    print('Var difference', np.var(ext['mu']) - np.var(ext_alt['mu']))

assert False

# Now try to fit the variance of theta instead of forcing it to be 1
# This doesn't work with only one class, I think. Gives infinite sigma_theta
code_int_over_class = """
data {
    int<lower=0> n_students; // number of students
    real y[n_students];     // outcomes
}
parameters {
    real mu;    // teacher effect
    real theta;
    real<lower=0> sigma_theta; // classroom effect variance
    real<lower=0> sigma_epsilon; // variance
}
model {
    mu ~ normal(0, 1);
    theta ~ normal(0, sigma_theta);
    y ~ normal(mu + theta, sigma_epsilon);
}
"""

if False:
    start = time.time()
    fit = pystan.stan(model_code=code_int_over_class, data=dat, iter=4000, chains=4)
    ext = fit.extract()
    print(time.time() - start)

# Now define multiple classes
code_multiple_classes = """
"""


