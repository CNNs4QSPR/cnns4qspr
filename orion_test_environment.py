import numpy as np
import pandas as pd
from cnns4qspr.loader import voxelize
from cnns4qspr.visualizer import plot_field
from cnns4qspr.featurizer import featurize, gen_feature_set
from cnns4qspr.trainer import Trainer

### Download/format protherm data
import os
from Bio.PDB import *

mutant_asa = pd.read_csv('../se3_data/protherm/mutant_asa.csv')
pdb_id = list(mutant_asa['PDB_mutant'])
pdb_id = [x.split(',')[0] if ',' in x else x for x in pdb_id]
asa = list(mutant_asa['ASA'])
asa = [float(x) for x in asa]

seen_ids = []
good_idxs = []
for i, id in enumerate(pdb_id):
    if id in seen_ids:
        pass
    else:
        seen_ids.append(id)
        good_idxs.append(i)

pdb_id = np.array(pdb_id)[good_idxs]
asa = np.array(asa)[good_idxs]

# Download pdbs
# pdbl = PDBList()
# for i, id in enumerate(pdb_id):
#     print(i)
#     pdbl.retrieve_pdb_file(id, pdir='../se3_data/protherm/pdbs', file_format='pdb')
# gen_feature_set('../se3_data/protherm/pdbs', save=True, save_fn='mutant_asa_features.npy', save_path='../se3_data/protherm/')

asa = list(asa)
asa.pop(125)
asa = np.array(asa).reshape(-1,1)
features = np.load('../se3_data/protherm/mutant_asa_features.npy')

vae = Trainer(type='regressor', network_type='feedforward', predictor=[128,128,128,64,28,10])
vae.train(features, asa, 100, 10, learning_rate=1e-4)
# predictions = vae.predict(features[:10,:])
# print(predictions)



### Create input tensor
# test_pdb_fn = 'cnns4qspr/formatting_data/sample_pdbs/1a00B00'
#
# field = voxelize(test_pdb_fn, channels=['polar', 'nonpolar'])
# plot_field(field['polar'])
# plot_field(field['nonpolar'])


### VAE Testing
# feature_set = gen_feature_set('cnns4qspr/formatting_data/sample_pdbs')
# features = np.load('feature_set.npy')
# labels = np.random.choice([0,1,2], size=features.shape[0], replace=True).reshape(-1,1)
#
# features = np.repeat(features, 100, axis=0)
# labels = np.repeat(labels, 100, axis=0)
#
# vae = VAETrainer(latent_size=3)
# # vae.train(features, labels, 20, 2)
# # vae.save()
# vae.load('best_vae.ckpt')
# vae.train(features, labels, 20, 2)
