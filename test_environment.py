from cnns4qspr.featurizer import load_cnn
from cnns4qspr.visualizer import plot_field, plot_internals
from cnns4qspr.loader import *

### Create input tensor
test_pdb_fn = 'cnns4qspr/formatting_data/sample_pdbs/1a00B00'
pdb_dict = load_pdb(test_pdb_fn)
field = voxelize(pdb_dict)['CA']
# x = torch.autograd.Variable(field, volatile=True)

### Load model
cnn = load_cnn('cnn_no_vae.ckpt')

### Internals plots
plot_field(field, threshold=0.0001)
plot_internals(cnn, field, 0, 0, threshold_on=False)
plot_internals(cnn, field, 0, 2, threshold_on=False)
plot_internals(cnn, field, 0, 3, threshold_on=False)
