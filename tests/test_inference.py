import unittest
import numpy as np
import pandas as pd
from sparseldv_qa.inference import paired_configuration_sensitivity

class PairedBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.data=pd.DataFrame([{"condition":i,"method":m,"count":24,"nrmse":float(i+shift)}
            for i in range(6) for m,shift in [("logdet",.5),("geometry",.0)]])
    def test_paired_and_reproducible(self):
        a=paired_configuration_sensitivity(self.data,n_boot=100,seed=1)
        b=paired_configuration_sensitivity(self.data,n_boot=100,seed=1)
        self.assertEqual(a,b)
        self.assertAlmostEqual(a["median_paired_difference"],.5)
        self.assertEqual(a["n_configuration"],6)
    def test_missing_configurations_rejected(self):
        with self.assertRaises(ValueError):paired_configuration_sensitivity(self.data.iloc[1:])
    def test_duplicate_configurations_rejected(self):
        with self.assertRaises(ValueError):paired_configuration_sensitivity(pd.concat([self.data,self.data.iloc[[0]]]))
    def test_distinct_methods_required(self):
        with self.assertRaises(ValueError):paired_configuration_sensitivity(self.data,"geometry","geometry")
