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
        # make sure the model can actually load
        cnn = featurizer.load_cnn('cnn_no_vae.ckpt')
        self.assertTrue(cnn != None)


    def test_featurize(self):
        """
        This method tests the featurizer method
        """
        path = 'examples/sample_pdbs/6fww.pdb'
        fields = loader.voxelize(path, channels=['LYS'])
        features = featurizer.featurize(fields)

        # make sure the two dictionaries contain same
        # number of entries
        self.assertTrue(len(fields.keys()) == len(features.keys()))
        for key in features:
            self.assertTrue(len(features[key][0]) == 256)




    def test_gen_feature_set(self):
        """
        This method tests the gen_feature_set function
        """


    if __name__ == '__main__':
        unittest.main()
