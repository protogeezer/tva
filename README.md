# tva

basic_va_alg.py implements the value-added algorithm described a 2008 NBER Working Paper: "Estimating Teacher Impacts on Student Acheivement", by Thomas Kane and Doug Staiger. All errors are mine. To run the VA algorithm, call the function calculate_va with the following parameters:

data (required, first parameter): A Pandas DataFrame that must contain the following columns, and may contain more:
    - 'teacher': A unique teacher identifier. 
    - 'class id': An identifier for a class.
    - 'score': The outcome of interest.
    - 'student id': Student identifier.
    If you have different column names, see the column_names argument.

covariates (required, second parameter): The covariates with which to residualize on. For fixed effects, see the categorical_controls argument.

jackknife (required, third parameter): Whether to use a jackknife estimator for each teacher's value-added.

residual: If you have already residualized, enter the name of the column containing the residual here.

column_names: If your columns don't have the names described in the data argument, use a dictionary in the column_names argument to specify this. For example, if you have teacher identifiers under 'teacher id' instead of 'teacher, use "column_names={'teacher id': 'teacher'}".

class_level_vars: The combination of variables needed to uniquely identify a classroom. For example, if 'class id' is a room number, class_level_vars might be ['class id', 'year', 'time of day']. By default, class_level_vars=['teacher', 'class id'].

categorical_controls: A categorical variable whose fixed effect you want to include when residualizing.

moments_only: If moments_only=True, this returns only structural parameters and does not estimate VA for individuals.

n_bootstrap_samples: Number of samples to take when calculating bootstrap standard errors for the variance of the teacher effect.
