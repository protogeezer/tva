import sys
sys.path = sys.path + ['/home/lizs/Dropbox/lizs_backup/Documents/tva/algorithm/']
import pandas as pd
from basic_va_alg import calculate_va
import numpy as np
from matplotlib import pyplot as plt
    
params = {'n collectors':2846, 'n districts':530, 'var mu': .024, 'var theta': .178, 'var delta': .4, 'T': 18*4, 'mean posting length':10}
column_names = {'person':'teacher', 'distcode':'class id', 'quarter':'student id', 'outcome':'score'}

data = pd.read_csv('simulated_data')
calculate_va(data, [], False, column_names=column_names, parallel=True, moments_only=True)
