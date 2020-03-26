from cnns4qspr.loader import voxelize
from cnns4qspr.visualizer import plot_field, plot_internals
from cnns4qspr.featurizer import featurize, gen_feature_set
from cnns4qspr.trainer import Trainer, CNNTrainer
import argparse
import numpy as np

def run():
    voxels = voxelize('examples/sample_pdbs', path_type='folder', channels=['backbone', 'polar'])
    targets = np.random.rand(voxels.shape[0])

    cnn = CNNTrainer(input_size=2, output_size=3, args=args)
    cnn.train(voxels, targets)
    # cnn.save(save_fn='test_ff_class_best'.format(test_type), save_path='test_checkpoints')
    # cnn.load('test_checkpoints/test_ff_class_best.ckpt')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Model initiation
    parser.add_argument("--model", default='CNN',
                        help="Which model definition to use")
    parser.add_argument("--predictor", choices=["feedforward", "vae"], default="feedforward",
                        help="Which model predictor class to use")
    parser.add_argument("--predictor-type", choices=["classifier", "regressor"], default="classifier",
                        help="Type of prediction")
    parser.add_argument("--report-frequency", default=1, type=int,
                        help="The frequency with which status reports will be written")
    parser.add_argument("--restore-checkpoint-filename", type=str, default=None,
                        help="Read model from checkpoint given by filename (assumed to be in checkpoint folder)")


    # Training parameters
    parser.add_argument("--training-epochs", default=100, type=int,
                        help="Which model definition to use")
    parser.add_argument("--batch-size", default=8, type=int,
                        help="Size of mini batches to use per iteration, can be accumulated via argument batchsize_multiplier (default: %(default)s)")
    parser.add_argument("--batchsize-multiplier", default=1, type=int,
                        help="number of minibatch iterations accumulated before applying the update step, effectively multiplying batchsize (default: %(default)s)")
    parser.add_argument("--initial_lr", default=1e-4, type=float,
                        help="Initial learning rate (without decay)")
    parser.add_argument("--lr_decay_start", type=int, default=40,
                        help="epoch after which the exponential learning rate decay starts")
    parser.add_argument("--lr_decay_base", type=float, default=.996,
                        help="exponential decay factor per epoch")

    # CNN parameters
    parser.add_argument("--kernel-size", type=int, default=3,
                        help="convolution kernel size")
    parser.add_argument("--p-drop-conv", type=float, default=0.2,
                        help="convolution/capsule dropout probability")
    parser.add_argument("--p-drop-fully", type=float, default=0.2,
                        help="fully connected layer dropout probability")
    parser.add_argument("--bandlimit-mode", choices={"conservative", "compromise", "sfcnn"}, default="compromise",
                        help="bandlimiting heuristic for spherical harmonics")
    parser.add_argument("--SE3-nonlinearity", choices={"gated", "norm"}, default="gated",
                        help="Which nonlinearity to use for non-scalar capsules")
    parser.add_argument("--normalization", choices={'batch', 'group', 'instance', None}, default='batch',
                        help="Which nonlinearity to use for non-scalar capsules")
    parser.add_argument("--downsample-by-pooling", action='store_true', default=True,
                        help="Switches from downsampling by striding to downsampling by pooling")

    # Predictor parameters
    parser.add_argument("--latent-size", default=20, type=int,
                        help="Size of latent space")
    parser.add_argument("--predictor-layers", default="[128,128]", type=str,
                        help="# nodes in each predictor layer")
    parser.add_argument("--encoder-layers", default="[128]", type=str,
                        help="# nodes in each encoder layer")
    parser.add_argument("--decoder-layers", default="[128]", type=str,
                        help="# nodes in each decoder layer")

    # Weights
    parser.add_argument("--lamb_conv_weight_L1", default=1e-7, type=float,
                        help="L1 regularization factor for convolution weights")
    parser.add_argument("--lamb_conv_weight_L2", default=1e-7, type=float,
                        help="L2 regularization factor for convolution weights")
    parser.add_argument("--lamb_bn_weight_L1", default=1e-7, type=float,
                        help="L1 regularization factor for batchnorm weights")
    parser.add_argument("--lamb_bn_weight_L2", default=1e-7, type=float,
                        help="L2 regularization factor for batchnorm weights")
    parser.add_argument("--lamb_linear_weight_L1", default=0, type=float,
                        help="L1 regularization factor for fully connected layer weights (except last / classification layer)")
    parser.add_argument("--lamb_linear_weight_L2", default=0, type=float,
                        help="L2 regularization factor for fully connected layer weights (except last / classification layer)")
    parser.add_argument("--lamb_softmax_weight_L1", default=0, type=float,
                        help="L1 regularization factor for classification layer weights")
    parser.add_argument("--lamb_softmax_weight_L2", default=0, type=float,
                        help="L2 regularization factor for classification layer weights")

    # Biases
    parser.add_argument("--lamb_conv_bias_L1", default=0, type=float,
                        help="L1 regularization factor for convolution biases")
    parser.add_argument("--lamb_conv_bias_L2", default=0, type=float,
                        help="L2 regularization factor for convolution biases")
    parser.add_argument("--lamb_norm_activ_bias_L1", default=0, type=float,
                        help="L1 regularization factor for norm activation biases")
    parser.add_argument("-lamb_norm_activ_bias_L2", default=0, type=float,
                        help="L2 regularization factor for norm activation biases")
    parser.add_argument("--lamb_bn_bias_L1", default=0, type=float,
                        help="L1 regularization factor for batchnorm biases")
    parser.add_argument("--lamb_bn_bias_L2", default=0, type=float,
                        help="L2 regularization factor for batchnorm biases")
    parser.add_argument("--lamb_linear_bias_L1", default=0, type=float,
                        help="L1 regularization factor for fully connected layer biases (except last / classification layer)")
    parser.add_argument("--lamb_linear_bias_L2", default=0, type=float,
                        help="L2 regularization factor for fully connected layer biases (except last / classification layer)")
    parser.add_argument("--lamb_softmax_bias_L1", default=0, type=float,
                        help="L1 regularization factor for classification layer biases")
    parser.add_argument("--lamb_softmax_bias_L2", default=0, type=float,
                        help="L2 regularization factor for classification layer biases")

    args, unparsed = parser.parse_known_args()

    run()
