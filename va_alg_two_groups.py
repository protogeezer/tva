from multiprocessing import Pool as ThreadPool
from va_functions import *
from two_type_covariance import calculate_covariances
import copy
    
## VA for just one teacher   
def get_va(df, var_theta_hat, var_epsilon_hat, var_mu_hat, jackknife):
    df = copy.copy(df)
    if not jackknife:
        data = [[df['teacher'].values[0], 0, 0, 0], [df['teacher'].values[0], 1, 0, 0]]
        new_df = pd.DataFrame(data, columns=['teacher', 'type', 'va', 'va other'])
    for student_type in [0, 1]:
        df_type = df[df['type'] == student_type]
        classes = df_type['class id'].values
        assert len(classes) == len(set(classes))
        
        precisions = np.zeros(len(classes))
        numerators = np.zeros(len(classes))
        
        sizes = df_type['size'].values
        scores = df_type['mean score'].values
        
        for k in range(len(classes)):
            class_size =  sizes[k]
            precisions[k] = 1 / (var_theta_hat[student_type] + var_epsilon_hat[student_type] / class_size)
            numerators[k] = precisions[k] * scores[k]

        precision_sum = np.sum(precisions)
        num_sum = np.sum(numerators)
        
        if jackknife:
            va = np.zeros(len(classes))
            for k in range(len(classes)):
                va[k] = (num_sum-numerators[k]) * var_mu_hat[student_type] / ((precision_sum - precisions[k]) * var_mu_hat[student_type] + 1)
                df.loc[(df['type'] == 1-student_type)&(df['class id'] == classes[k]), 'va other'] = va[k]
            df.loc[df['type'] == student_type, 'va'] = va
        else:
            va = num_sum * var_mu_hat[student_type] / (precision_sum * var_mu_hat[student_type] + 1)
            new_df.loc[new_df['type'] == student_type, 'va'] = va
            new_df.loc[new_df['type'] == 1-student_type, 'va other'] = va
            
    return df   
    
def get_phi_coefficients(scores, sizes, var_mu, var_theta, var_epsilon, cov_mu, cov_theta):
    s_00, s_11 = [var_mu[i] + np.dot(sizes[i], sizes[i])*var_theta[i]/np.sum(sizes[i])**2 \
                  + var_epsilon[i]/np.sum(sizes[i]) for i in [0,1]]
    s_01 = cov_mu + np.dot(sizes[0], sizes[1])*cov_theta/(np.sum(sizes[0])*np.sum(sizes[1]))

    phi_0 = np.linalg.solve([[s_00, s_01], [s_01, s_11]], [var_mu[0], cov_mu])
    phi_1 = np.linalg.solve([[s_00, s_01], [s_01, s_11]], [cov_mu, var_mu[1]])
    
    return [phi_0, phi_1]
    
# Other way of calculating VA, uses data from both genders
def get_va_2(df, var_theta_hat, var_epsilon_hat, var_mu_hat, cov_theta_hat, cov_mu_hat):
    years = remove_duplicates(df['year'].values)
    # Jackknife estimator requires multiple years
    if len(years) < 2:
        return df
        
    # Identify classes when both girls and boys were taught
    classes = set(df[df['type'] ==0]['class id'].values) & set(df[df['type'] == 1]['class id'].values)
    # Keep only classes in which both girls and boys were taught
    df_good_classes = df[[id_ in classes for id_ in df['class id'].values]]

    for year in years:
        # Remove current year for jackknife estimation
        df_type_otheryears = [df_good_classes[(df_good_classes['type'] == i) & (df_good_classes['year'] != year)] for i in [0,1]]
        if (not df_type_otheryears[0].empty) and (not df_type_otheryears[1].empty):
            sizes = np.array([df_type_otheryears[i]['size'].values for i in [0,1]])
            assert len(sizes[0]) > 0
            assert len(sizes[1]) > 0
            scores = np.array([df_type_otheryears[i]['mean score'].values for i in [0,1]])

            phi = get_phi_coefficients(scores, sizes, var_mu_hat, var_theta_hat, var_epsilon_hat, cov_mu_hat, cov_theta_hat)

            va = [phi[i][0]*np.dot(sizes[0], scores[0])/np.sum(sizes[0]) 
                + phi[i][1]*np.dot(sizes[1], scores[1])/np.sum(sizes[1]) for i in [0,1]]

            indices = pd.Series(df['year'] == year)
            df.loc[(indices & (df['type'] == 0)), 'va'] = va[0]
            df.loc[(indices & (df['type'] == 1)), 'va other'] = va[0]
            df.loc[(indices & (df['type'] == 1)), 'va'] = va[1]
            df.loc[(indices & (df['type'] == 0)), 'va other'] = va[1]
            
    return df
   
def get_va_from_teacher_list(input_tuple):
    data, teacher_list, var_theta_hat, var_epsilon_hat, var_mu_hat, cov_theta_hat, cov_mu_hat, method = input_tuple
    results = list(np.zeros(len(teacher_list)))
    if method == 1:
        for i in range(len(teacher_list)):
            results[i] = get_va(data[data['teacher'] == teacher_list[i]], var_theta_hat, var_epsilon_hat, var_mu_hat, jackknife)
    else:
        for i in range(len(teacher_list)):
            results[i] = get_va_2(data[data['teacher'] == teacher_list[i]], var_theta_hat, var_epsilon_hat, var_mu_hat, cov_theta_hat, cov_mu_hat, jackknife)

    return pd.concat(results)

    
## Returns VA's and important moments
## a residual can be specified
## Covariates is a list like ['prev score', 'free lunch']
## Column names can specify 'class id', 'student id', and 'type'
def calculate_va(data, covariates, residual = None, moments = None, column_names = None, method = 1, parallel = False, jackknife = True):
    assert method in [1,2]
    
    moments, class_type_df, teachers = calculate_covariances(data, covariates, residual, moments, column_names)
    var_theta_hat, var_epsilon_hat, var_mu_hat, cov_theta_hat, cov_mu_hat, corr_mu_hat = \
        moments['var theta'], moments['var epsilon'], moments['var mu'], moments['cov theta'], moments['cov mu'], moments['corr mu']
    
    results = list(np.zeros(len(teachers)))

    if parallel:
        pool = ThreadPool(num_cores)
        sublists = [(class_type_df, teachers[i::num_cores], var_theta_hat, var_epsilon_hat, var_mu_hat, cov_theta_hat, cov_mu_hat, method) for i in range(num_cores)]
        results = pool.map(get_va_from_teacher_list, sublists)
        pool.close()
        pool.join()
    else:
        for i in range(len(teachers)):
            if method == 1:
                results[i] = get_va(class_type_df[class_type_df['teacher'] == teachers[i]], \
                                    var_theta_hat, var_epsilon_hat, var_mu_hat, jackknife)
            else:
                results[i] = get_va_2(class_type_df[class_type_df['teacher'] == teachers[i]], \
                                     var_theta_hat, var_epsilon_hat, var_mu_hat, cov_theta_hat, cov_mu_hat)
            
    
    class_type_df = pd.concat(results)
    
    return [class_type_df, var_mu_hat, corr_mu_hat, var_theta_hat, var_epsilon_hat]
