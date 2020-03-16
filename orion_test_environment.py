import numpy as np
import pandas as pd
from cnns4qspr.loader import voxelize
from cnns4qspr.visualizer import plot_field
from cnns4qspr.featurizer import featurize, gen_feature_set
from cnns4qspr.trainer import Trainer

### Download/format protherm data
import os
from Bio.PDB import *

wild_tm = pd.read_csv('../se3_data/protherm/wild_tm.csv')
pdb_id = list(wild_tm['PDB_wild'])
pdb_id = [x.split(',')[0] if ',' in x else x for x in pdb_id]
tm = []
for t in list(wild_tm['Tm']):
    try:
        tm.append(float(t))
    except ValueError:
        t = t.split(' ')
        tm.append(float(t[0]))

seen_ids = []
good_idxs = []
for i, id in enumerate(pdb_id):
    if id in seen_ids:
        pass
    else:
        seen_ids.append(id)
        good_idxs.append(i)

pdb_id = np.array(pdb_id)[good_idxs]
tm = np.array(tm)[good_idxs]

# Download pdbs
# pdbl = PDBList()
# for i, id in enumerate(pdb_id):
#     print(i)
#     pdbl.retrieve_pdb_file(id, pdir='../se3_data/protherm/tm_pdbs', file_format='pdb')
# gen_feature_set('../se3_data/protherm/tm_pdbs', save=True, save_fn='wild_tm_features.npy', save_path='../se3_data/protherm/')

tm = list(tm)
pop_idxs = [374,341,294,263,259,242,237,126,24]
for idx in pop_idxs:
    tm.pop(idx)
tm = np.array(tm).reshape(-1,1)
features = np.load('../se3_data/protherm/wild_tm_features.npy')

vae = Trainer(type='regressor', network_type='feedforward', predictor=[128,128,128,64,28,10])
vae.train(features, tm, 1000, 10, learning_rate=0.01)#, verbose=False)


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
