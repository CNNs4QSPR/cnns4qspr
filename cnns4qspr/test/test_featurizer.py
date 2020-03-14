import torch
import numpy as np
from biopandas.pdb import PandasPdb
import unittest

import cnns4qspr.loader as loader
import cnns4qspr.featurizer as featurizer

class test_featurizer(unittest.TestCase):

    def test_load_cnn(self):
        """
        This method tests the load_cnn function
        """
        path = 'cnns4qspr/formatting_data/sample_pdbs/6fww.pdb'


    def test_featurize(self):
        """
        This method tests the featurizer method
        """


    def test_gen_feature_set(self):
        """
        This method tests the gen_feature_set function
        """
