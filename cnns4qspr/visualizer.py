"""
This module contains functions to plot atomic density fields before
they go into a model, as well as what the density fields have been
transformed into at certain points within the model.
"""
import torch
import numpy as np
import pandas as pd
import plotly.express as px


def outer_block1_hook(module, input_, output):
    global outer_block1_out
    outer_block1_out = output


def outer_block2_hook(module, input_, output):
    global outer_block2_out
    outer_block2_out = output


def outer_block3_hook(module, input_, output):
    global outer_block3_out
    outer_block3_out = output


def outer_block4_hook(module, input_, output):
    global outer_block4_out
    outer_block4_out = output


def outer_block5_hook(module, input_, output):
    global outer_block5_out
    outer_block5_out = output


def plot_field(
        field,
        color='deep',
        threshold=0.2,
        alpha=0.7,
        show=True,
        title=''):
    """
    This function takes a tensorfield and plots the field density in 3D
    space. The field describes an atomic "density" at each voxel.

    Parameters:
        field (pytorch tensor, required): A field from a field
            dictionary that was output by
            either `voxelize` or `make_fields`.

        color (str, optional): The color scheme to plot the field. Any
            of the Plotly continuous color schemes. 'deep' and 'ice_r'
            are recommended as good baselines.

        threshold (float, optional): The threshold intensity that a
            voxel must have in order to be included in the plot.

        alpha (float, optional): Amount of transparency to use in
            plotted marks.

        show (boolean, optional): Whether to show the plot. If false,
            the plotly fig object is returned.

    Returns:
        plotly figure object: If show=False, a plotly figure object is
        returned
    """
    # if the cube is coming for the CathData class, it's indexed differently
    cube = field[0][0].numpy()

    cube /= cube.max()

    cubelist = []
    xval = np.linspace(-len(cube[0]) + 1, len(cube[0]) - 1, 50)
    yval = np.linspace(-len(cube[0]) + 1, len(cube[0]) - 1, 50)
    zval = np.linspace(-len(cube[0]) + 1, len(cube[0]) - 1, 50)

    # make a dataframe of x,y,z,intensity for each point in the cube
    # to do this have to loop through the cube
    for i, xval2 in enumerate(xval):

        for j, yval2 in enumerate(yval):

            for k, zval2 in enumerate(zval):

                cubelist.append([xval2, yval2, zval2, cube[i][j][k]])

    cube_df = pd.DataFrame(cubelist, columns=['x', 'y', 'z', 'intensity'])

    # only show the voxels with some intensity
    cube_df = cube_df[cube_df['intensity'] > threshold]

    fig = px.scatter_3d(cube_df, x='x', y='y', z='z',
                        color='intensity', opacity=alpha,
                        color_continuous_scale=color)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-25, 25]),
            yaxis=dict(range=[-25, 25]),
            zaxis=dict(range=[-25, 25])
        ), title=title
    )
    if show:
        fig.show()
        figret = None
    else:
        figret = fig
    return figret


def plot_internals(
        model,
        field,
        block=0,
        channel=0,
        threshold_on=True,
        feature_on=False):
    """
    This function enables visualization of the output of various
    convolutional layers inside the model.

    Parameters:
        model (pytorch neural network, required): The CNN model that
            data can be sent through. The model object can be
            constructed by using featurizer.load_cnn()

        field (pytorch tensor, required): A field from a field
            dictionary (see make_fields or voxelize)

        block (int, optional): Any of the 5 major blocks contained in
            the model object. This integer determines at which stage of
            the CNN the data will be plotted from.

        channel (int, optional):

        threshold_on (boolean, optional): If threshold_on=True, plot
            will be constructed with the default threshold value from
            plot_fields (0.2). If threshold_on=False, plot is
            constructed with threshold 0.0001

        feature_on (boolean, optional): Whether or not to return the
            feature vector of the field that was sent through the model
            to make this plot.

    Returns:
        pytorch tensor: the feature vector that results from the
        particular field being sent all the way through the network.
        Only returns if feature_on=True
    """

    model.blocks[0].register_forward_hook(outer_block1_hook)
    model.blocks[1].register_forward_hook(outer_block2_hook)
    model.blocks[2].register_forward_hook(outer_block3_hook)
    model.blocks[3].register_forward_hook(outer_block4_hook)
    model.blocks[4].register_forward_hook(outer_block5_hook)
    model.eval()

    xval = torch.autograd.Variable(field)
    feature_vec = model(xval)
    blocks = [outer_block1_out,
              outer_block2_out,
              outer_block3_out,
              outer_block4_out,
              outer_block5_out]

    internal_field = blocks[block][:, channel, :, :, :].detach().view(1,1,50,50,50)
    if threshold_on:
        fig = plot_field(internal_field, show=False)
    else:
        fig = plot_field(internal_field, threshold=0.0001, show=False)
    fig.show()
    if feature_on:
        feature = feature_vec
    else:
        feature = None
    return feature
