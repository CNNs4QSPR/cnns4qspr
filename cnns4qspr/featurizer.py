"""
This module contains functions for loading a pre-trained convolutional
neural network and getting convolutional vector from the tensor fields
which can be used for VAE or for plotting the internals.
"""
import os
import sys
import torch
import numpy as np
import pkg_resources
from cnns4qspr.loader import voxelize
from cnns4qspr.se3cnn_v3.networks.CNN import CNN

SE3CNN_LIB = pkg_resources.resource_filename('cnns4qspr', 'se3cnn_v3')

def load_cnn(checkpoint_fn, n_input=1):
    """
    This function loads the pretrained SE3 Convolutional Neural Net
    to be used as a feature extractor from the users input pdb file.

    Parameters:
        checkpoint_fn (str, required): The filename of the ckpt file
            used to load the model.

    Returns:
        model (pytorch.network): A fully trained pytorch network object
            with 256 output nodes.
    """

    # Read and load checkpoint files
    checkpoint_path_restore = '{:s}/checkpoints/{:s}'.format(SE3CNN_LIB, checkpoint_fn)
    checkpoint = torch.load(checkpoint_path_restore, map_location=torch.device('cpu'))

    # Load model
    cnn = CNN.network(n_input=n_input)
    cnn.load_state_dict(checkpoint['state_dict'])
    cnn.eval()

    return cnn

def featurize(field, channels='all'):
    """
    Takes a dictionary of voxelized tensor fields (50x50x50) and
    convolutes to a single 256 element scalar feature vector

    Parameters:
        field (dict, required): Dictionary storing channel type as key
            and voxelized input tensor as value

    Returns:
        feature_vec (dict): Dictionary storing channel type as key and
            dense structural feature vector as value
    """
    feature_vec = {}
    cnn = load_cnn('cnn_no_vae.ckpt')

    if channels == 'all':
        channels = list(field.keys())

    for atom in channels:
        input_tensor = torch.Tensor(field[atom]).unsqueeze(0).unsqueeze(0)
        feature_vec[atom] = cnn(input_tensor).data.cpu().numpy()

    return feature_vec

def gen_feature_set(pdb_path, channels=['CA'], save=False, \
                    save_fn='feature_set.npy', save_path='./'):
    """
    This function takes a directory of pdbs and either saves them or
    returns an nx256 np array where n is number of pdbs.

    Parameters:
        pdb_path (path, required): path of the pdb file

        channels (list-like, optional): for the specific channels we
            want the arrays for.

    Returns:
        feature_set ( numpy array): returns the array of nx256 where n
            is the number of pdbs.
    """

    pdb_fns = os.listdir(pdb_path)
    feature_set = np.zeros((len(channels), len(pdb_fns), 256))

    for j, pdb_fn in enumerate(pdb_fns):
        progress = '{}/{} pdbs voxelized ({}%)'.format(j, len(pdb_fns), \
                    round(j / len(pdb_fns) * 100, 2))
        sys.stdout.write('\r'+progress)
        path = os.path.join(pdb_path, pdb_fn)
        field = voxelize(path, channels=channels)
        features = featurize(field)
        for i, channel in enumerate(channels):
            feature_set[i, j, :] = features[channel]

    sys.stdout.write("\r\033[K")
    out_statement = 'voxelization complete!\n'
    sys.stdout.write('\r'+out_statement)

    feature_set = np.squeeze(feature_set)

    if save:
        path = os.path.join(save_path, save_fn)
        np.save(path, feature_set)
    else:
        return feature_set
