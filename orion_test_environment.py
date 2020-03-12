from cnns4qspr.loader import voxelize
from cnns4qspr.featurizer import featurize, gen_feature_set
from cnns4qspr.trainer import VAE

### Create input tensor
# test_pdb_fn = 'cnns4qspr/formatting_data/sample_pdbs/1a00B00'
#
# field = voxelize(test_pdb_fn, channels=['CA', 'CD', 'NH2'])
# features = featurize(field, channels='all')

# vae = VAE(latent_size=10)
# print(vae.layers)

feature_set = gen_feature_set('cnns4qspr/formatting_data/sample_pdbs')
print(feature_set.shape)
