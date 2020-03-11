import os
import torch
import importlib
import numpy as np
import pkg_resources
from cnns4qspr.se3cnn_v3.networks.CNN import CNN

SE3CNN_LIB = pkg_resources.resource_filename('cnns4qspr', 'se3cnn_v3')

def load_cnn(checkpoint_fn):
    """
    This function loads the pretrained SE3 Convolutional Neural Net
    to be used as a feature extractor from the users input pdb file.

    Parameters:
    ___________

    checkpoint_fn (str, required): The filename of the ckpt file used to load
    the model.

    Returns:
    ________

    model (pytorch.network): A fully trained pytorch network object with 256
    output nodes.
    """

    # Read and load checkpoint files
    checkpoint_path_restore = '{:s}/checkpoints/{:s}'.format(SE3CNN_LIB, checkpoint_fn)
    checkpoint = torch.load(checkpoint_path_restore, map_location=torch.device('cpu'))

    # Load model
    cnn = CNN.network()
    cnn.load_state_dict(checkpoint['state_dict'])
    cnn.eval()

    return cnn
