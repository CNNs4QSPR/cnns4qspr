import torch
import numpy as np
import unittest
import os
import math
import sys
sys.path.append(os.path.abspath('../../')) # to be able to import
cwd = os.getcwd()
import cnns4qspr.trainer as trainer
from cnns4qspr.loader import *
import cnns4qspr.featurizer as featurizer



class test_trainer(unittest.TestCase):


    def test_Trainer(self):
        """
        This method tests the Trainer class
        """
        ####
        #### testing the __init__ and predict methods
        ####
        output_size = 3
        laten_size = 3
        #grab a protein from the sample pdbs and voxelize it
        #os.chdir('../../examples/sample_pdbs/')
        os.chdir('examples/sample_pdbs/') # if running unittest discover
        fields = voxelize('6fww.pdb', channels=['CA'])

        # return to the testing directory, business as usual
        os.chdir(cwd)
        feature = featurizer.featurize(fields)

        # default trainer model, classifier, vae, etc...
        model = trainer.Trainer()
        vae_input = feature['CA'].astype('double')

        # assert the fact we can make a prediction, and it's a single int
        prediction = model.predict(vae_input)
        self.assertTrue(len(prediction) == 1)


        ####
        #### testing the train and save methods
        ####
        #os.chdir('../../examples/cath_data/')
        os.chdir('examples/cath_data/')
        train = np.load('train_cath.npy')
        os.chdir(cwd)

        features = train[:,:-1]
        labels = train[:,-1]
        model.train(features, labels, n_epochs=1, batch_size=8)
        accuracy1 = model.checkpoint['best_accuracy']
        model.train(features, labels, n_epochs=3, batch_size=8)
        accuracy2 = model.checkpoint['best_accuracy']

        # assert the thing can actually train itself
        self.assertTrue(accuracy1 < accuracy2)

        #save it, so we can load the state and compare
        model.save()


        ####
        #### testing the load method
        ####
        model2 = trainer.Trainer()
        model2.load('best_vae.ckpt')

        # assert that the architectures are equal in the 2 models
        self.assertEqual(model.latent_size, model2.latent_size)
        self.assertEqual(model.network_type, model2.network_type)
        for key in model.architecture.keys():
            # if the entry is a list, assert every entry in the list is equal
            if type(model.architecture[key]) is type([]):
                for i, entry in enumerate(model.architecture[key]):
                    self.assertEqual(entry, model2.architecture[key][i])

            # if the entry is an int, assert the int is equal
            if type(model.architecture[key] is type(1)):
                self.assertEqual(model.architecture[key], model2.architecture[key])


if __name__ == '__main__':
    unittest.main()
