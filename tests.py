import pandas as pd
import numpy as np
from va_functions import *
import matplotlib.pyplot as plt
import scipy.stats as stats
import unittest

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
        import ipdb; ipdb.set_trace()
        self.assertTrue(abs(log_det_1 - log_det_2) < 10**(-3))

           
    def test_normalize(self):
        vector = np.random.normal(np.random.normal(), abs(np.random.normal()), 100)
        normalized = normalize(vector)
        self.assertTrue( round(np.mean(normalized), 3) == 0)
        self.assertTrue( round(np.var(normalized), 3) == 1  )
    
if __name__ == '__main__':
    unittest.main()
