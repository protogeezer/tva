import numpy as np
import sys
from config import *
sys.path += hdfe_dir
from hdfe import Groupby
import pandas as pd

def get_covariance_matrix(group_key, residuals):
    grouped = Groupby(group_key)
    h = residuals.shape[1]
    estimated_cov = np.zeros((h, h))
    mean_out = np.zeros(h)
    n = 0

    for idx in grouped.indices:
        if len(idx) > 1:
            estimated_cov += residuals[idx[0], :, None].dot(residuals[idx[1], :, None].T)
            mean_out += np.sum(residuals[idx[:2], :], 0)
            n += 1

    mean_out = mean_out[:, None]
    return (estimated_cov - mean_out.dot(mean_out.T) / (2 * n)) / n

def estimate_factor_model(df, teacher, classroom, outcomes, covariates, school):
    grouped = Groupby(df[teacher])
    x = grouped.apply(lambda x: x - np.mean(x, 0), df[covariates].values,
                      width=len(covariates))
    alpha_hat = np.linalg.lstsq(x, df[outcomes].values)[0]
    residuals = df[outcomes].values - df[covariates].values.dot(alpha_hat)
    residuals -= np.mean(residuals, 0)

    residual_cols = ['residual' + str(h) for h in range(n_outcomes)]
    for h, col in enumerate(residual_cols):
        df[col] = residuals[:, h]
    class_data = df[[school, classroom, teacher] + residual_cols].groupby(classroom).mean()
    estimated_cov = get_covariance_matrix(class_data[teacher], class_data[residual_cols].values)
    return estimated_cov


if __name__ == '__main__':
    n_schools = 1100
    n_classrooms_per_school = 70
    n_covariates = 5
    n_outcomes = 3
    n_students_per_class = 13

    n_students_per_school = n_classrooms_per_school + n_students_per_class
    student_df = pd.DataFrame({'school': np.repeat(range(n_schools), 
                                                   n_students_per_school)})
    for i in range(n_covariates):
        student_df['x' + str(i)] = np.random.normal(0, 1, len(student_df))
    covariates = ['x' + str(i) for i in range(n_covariates)]

    student_df['classroom'] = student_df['school'] * n_classrooms_per_school +\
             np.concatenate([np.random.choice(range(n_classrooms_per_school), n_students_per_school)
                             for _ in range(n_schools)])
    # Randomly assign a teacher to each classroom
    student_df['teacher'] = np.floor(student_df['classroom'] / 2)

    # Now we need to generate outcomes. First, let's generate teacher effects:
    n_teachers = len(set(student_df['teacher']))
    tmp = np.random.normal(0, 1, (n_outcomes, n_outcomes * 2))
    covariance_matrix = tmp.dot(tmp.T)
    teacher_effects = np.random.multivariate_normal(np.zeros(n_outcomes), covariance_matrix, n_teachers)
    # Each teacher has H value-added components
    teacher_df = pd.DataFrame(data = teacher_effects,
                              index = np.unique(student_df['teacher']))
    alpha = np.random.normal(0, 1, (n_covariates, n_outcomes))
    
    outcome = student_df[covariates].dot(alpha).values\
            + teacher_df.loc[student_df['teacher'], :].values\
            + np.random.normal(0, 1, (len(student_df), 1))
    for h in range(n_outcomes):
        student_df['outcome' + str(h)] = outcome[:, h]
    outcome_cols = ['outcome' + str(h) for h in range(n_outcomes)]

    estimated_cov = estimate_factor_model(student_df, 'teacher', 'classroom',
                                          outcome_cols, covariates, 'school')
    print(estimated_cov)
    print(covariance_matrix - estimated_cov)
    
