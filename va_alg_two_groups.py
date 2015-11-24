from multiprocessing import Pool as ThreadPool
from va_functions import *
from two_type_covariance import calculate_covariances
import copy


def get_each_va(df, var_theta_hat, var_epsilon, var_mu_hat, jackknife):
    @profile
    def f(df): # df columns are type, size, mean score
        type_0_vas = get_va(df[::2], var_theta_hat[0], var_epsilon[0], var_mu_hat[0], jackknife)
        type_1_vas = get_va(df[1::2], var_theta_hat[1], var_epsilon[1], var_mu_hat[1], jackknife)
        if jackknife: 
            va_vector = []
            for i in range(int(len(df)/2)):
                va_vector += [type_0_vas[i], type_1_vas[i]]
            assert len(va_vector) == len(df)
            return va_vector        
        else:
            return [type_0_vas, type_1_vas]

        
    if jackknife:
        results = df.groupby('teacher')[['size', 'mean score']].apply(f).values
        df.loc[:, 'va'] = np.hstack(results)
    else:
        results = pd.DataFrame(df.groupby('teacher')[['size', 'mean score']].apply(f))
        values = np.array([elt[0] for elt in results.values])
        results.drop(results.columns[0], 1, inplace=True)
        results['type 0 va'] = values[:, 0]
        results['type 1 va'] = values[:, 1]
        df = pd.merge(df, results.reset_index())
    print(df)
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
@profile
def calculate_va(data, covariates, jackknife, residual = None, moments = None, column_names = None, method = 1, parallel = False):
    assert method in [1,2]
    
    moments, n, class_type_df, teachers = calculate_covariances(data, covariates, residual, moments, column_names)
    var_theta_hat, var_epsilon_hat, var_mu_hat, cov_theta_hat, cov_mu_hat, corr_mu_hat = \
        moments['var theta'], moments['var epsilon'], moments['var mu'], moments['cov theta'], moments['cov mu'], moments['corr mu']
    
    results = list(np.zeros(len(teachers)))

    if parallel:
        pool = ThreadPool(num_cores)
        sublists = [(class_type_df, teachers[i::num_cores], var_theta_hat, var_epsilon_hat, var_mu_hat, cov_theta_hat, cov_mu_hat, method) for i in range(num_cores)]
        results = pool.map(get_va_from_teacher_list, sublists)
        pool.close()
        pool.join()
        class_type_df = pd.concat(results)
    else:
        if method == 1:
            class_type_df = get_each_va(class_type_df, var_theta_hat, var_epsilon_hat, var_mu_hat, jackknife)
        else:
            for i in range(len(teachers)):
                results[i] = get_va_2(class_type_df[class_type_df['teacher'] == teachers[i]], \
                                     var_theta_hat, var_epsilon_hat, var_mu_hat, cov_theta_hat, cov_mu_hat)
            class_type_df = pd.concat(results)
    
    return [class_type_df, var_mu_hat, corr_mu_hat, var_theta_hat, var_epsilon_hat]
