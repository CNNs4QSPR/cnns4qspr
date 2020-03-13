import numpy as np
from cnns4qspr.loader import voxelize
from cnns4qspr.visualizer import plot_field
from cnns4qspr.featurizer import featurize, gen_feature_set
from cnns4qspr.trainer import VAETrainer

### Create input tensor
test_pdb_fn = 'cnns4qspr/formatting_data/sample_pdbs/1a00B00'

field = voxelize(test_pdb_fn, channels=['polar', 'nonpolar'])
plot_field(field['polar'])
plot_field(field['nonpolar'])



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
