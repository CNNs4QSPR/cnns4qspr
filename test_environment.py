from cnns4qspr.featurizer import load_cnn
from cnns4qspr.loader import *

### Create input tensor
test_pdb_fn = 'cnns4qspr/formatting_data/sample_pdbs/1a00B00'
pdb_dict = load_pdb(test_pdb_fn)
fields = voxelize(pdb_dict)['CA']
x = torch.autograd.Variable(fields, volatile=True)

### Create input tensor from cath data
# from cnns4qspr.se3cnn_v3.util import *
# from cnns4qspr.se3cnn_v3.util.format_data import CathData
#
# validation_set = CathData(
#          'cath_3class_ca.npz', split=7, download=False,
#          randomize_orientation=False,
#          discretization_bins=50,
#          discretization_bin_size=2.0)
# validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=1, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
#
# for batch_idx, (data, target) in enumerate(validation_loader):
#     if batch_idx > 0:
#         break
#     test_input = data
# print(test_input.shape)

### Load model
cnn = load_cnn('cnn_no_vae.ckpt')

# feature_vector = cnn(x)
# print(feature_vector.shape)
