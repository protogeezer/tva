# tva

basic_va_alg.py implements three different value-added estimators; see the
'method' parameter for more information. Parameters are

data: A Pandas DataFrame. It must contain the columns named in the parameters
outcome, teacher, covariates, class_level_vars, and categorical_controls.

covariates: A list of strings containing the names of columns that contain
covariate data. These should be scalars; to generate fixed effects from 
a vector of categorical_data, use the categorical_controls argument.
Example: ['previous test score', 'age']

class_level_vars: A list of column names that, incombination, uniquely
identify a classroom. For example, if a teacher, year, and time period
uniquely identify a classroom (Mrs. Smith's 9 am class in 2015-2016), 
class_level_vars might be ['teacher', 'year', 'time'].

categorical_controls (optional): List of columns that contain categorical data, which
will be expanded into fixed effects. For exmaple,
categorical_controls = ['ethnicity', 'home language'].

jackknife (optional, default = False): Whether to use a jackknife estimator for each teacher's value-added.

moments_only (optional, default = True): If moments_only=True, this returns only structural parameters and does not estimate VA for individuals.

method:

if method is 'ks' (default), implements the estimator in Kane and Staiger (2008).

if method is 'cfr', implements the estimator in Kane and Staiger (2008) but with
a tweak suggested by Chetty, Friedman, and Rockoff (2014): Coefficients on covariates
are estimated in the presence of teacher fixed effects.

if method is 'fk', implements an estimator inspired by Fessler and Kasy (2016).
