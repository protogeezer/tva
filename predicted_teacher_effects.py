from va_functions import *

# TODO: everything

def collapse(data):
    class_df = data.groupby(('teacher', 'class id', 'year'))['student id'].count().reset_index()
    class_df.columns = ['teacher', 'class id', 'year', 'size']
    return class_df
    
def unbiased_VA_vector(df_with_one_classroom):
    
    
# columns should contain 'class id', 'continuous var', and 'student id'
# moments can contain 'var epsilon'
def get_va_vector(data, covariates, columns = {}, moments = None):
    # Correct variables for column names
    data.rename(columns = {columns[k]:k for k in columns})
            
    df = drop_one_class_teachers(data)

    class_df = collapse(df) # Create dataframe at class level
    classes = class_df['class id'].remove_duplicates()
    
#    # Add column for each control variable
#    for covariate in covariates:
#        df[covariate] = None
    
    ## Create unbiased estimate of VA for each teacher_level_df
    
    
    return whatever
