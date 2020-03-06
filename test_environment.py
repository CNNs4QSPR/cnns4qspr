import os
from se3cnn_v3.util import util

model = util.load_model('SE3ResNet34Small', 20, 'trial_8_best.ckpt')

# density_tensor = voxelizer(pdb_filename)
# decoded_vector, mean_vector, variance_vector = model(density_tensor)
# density_tensor -> pytorch.tensor (1x50x50x50)
