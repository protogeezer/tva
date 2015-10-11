import pandas as pd
import va_functions

# Basic algorithm: Effect is constant within teacher
directory = '~/Documents/tva/algorithm/'
filename = 'baseline_simulated_data'

data = pd.read_csv(directory+filename+'.csv', sep=',', nrows=10000)

_, beta = va_functions.residualize(data, 'score', ['x1', 'x2'], 'teacher')
print('beta ' + str(beta))
