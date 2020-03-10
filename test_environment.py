from cnns4qspr.featurizer import load_cnn
from cnns4qspr.loader import *

test_pdb_fn = 'cnns4qspr/formatting_data/sample_pdbs/1a00B00'
pdb_dict = load_pdb(test_pdb_fn)
fields = voxelize(pdb_dict)['CA']
cnn = load_cnn('cnn_no_vae.ckpt')

feature_vector = cnn(fields)
print(feature_vector.shape)
