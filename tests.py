import pandas as pd
import numpy as np
from va_functions import *
import matplotlib.pyplot as plt
import scipy.stats as stats
import unittest
import sys
from config import *
sys.path += [hdfe_dir]
from hdfe import Groupby

class TestFunctions(unittest.TestCase):
    def test_log_det_formula(self):
        J = 5
        k = 5
        x = np.random.normal(0, 1, (1000, J + k))
        V = x.T.dot(x)
        Sigma = V
        tau_squared = 2

        for i in range(J):
            Sigma[i,i] += tau_squared

        log_det_1 = np.log(np.linalg.det(Sigma))
        # Now do it the other way
        v_11 = V[:J, :J]
        v_12 = V[:J, J:]
        v_22 = V[J:, J:]
        schur = v_11 - v_12.dot(np.linalg.lstsq(v_22, v_12.T)[0])
        e_values = np.linalg.eigvalsh(schur)
        log_det_2 = np.sum(np.log(e_values + tau_squared)) + np.log(np.linalg.det(v_22))
        print(log_det_1)
        print(log_det_2)
        self.assertTrue(abs(log_det_1 - log_det_2) < 10**(-1))

           
    def test_normalize(self):
        vector = np.random.normal(np.random.normal(), abs(np.random.normal()), 100)
        normalized = normalize(vector)
        self.assertTrue( round(np.mean(normalized), 3) == 0)
        self.assertTrue( round(np.var(normalized), 3) == 1  )

    def test_groupby(self):
        ids = np.array([1, 1, 1, 0, 0])
        y = np.array([1, 2, 3, 4, 7])
        grouped_1 = Groupby(ids)
        means_1 = grouped_1.apply(np.mean, y)
        self.assertFalse(grouped_1.already_sorted)
        grouped_2 = Groupby(ids[::-1])
        means_2 = grouped_2.apply(np.mean, y[::-1])
        self.assertTrue(grouped_2.already_sorted)
        self.assertTrue((means_1 == means_2[::-1]).all())

    
if __name__ == '__main__':
    unittest.main()
