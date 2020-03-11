"""
This module contains functions to plot atomic density fields before they
go into a model, as well as what the density fields have been transformed into
at certain points within the model.
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


def plot_field(field, color='ice_r', from_voxelizer=True, threshold=0.2, alpha=0.7, show=True):
    """
    This function takes a field datacube from voxelizer and plots a heat map of
    the 50x50x50 field. The field describes an atomic "denisty" at each voxel.

    Parameters:
    ___________
    data (pytorch tensor): The data cube that comes out of the voxelizer

    color (str): The color scheme to plot the field with.
                 Any of the Plotly continuous color schemes.
                 deep and ice_r are recommended.
    """
    # if the cube is coming for the CathData class, it's indexed differently
    cube = field[0][0].numpy()
    if from_voxelizer:
        cube = field.numpy()
        cube = np.reshape(cube, (50,50,50))

    cube /= cube.max()

    cubelist = []
    x = np.linspace(-len(cube[0]) + 1, len(cube[0]) - 1, 50)
    y = np.linspace(-len(cube[0]) + 1, len(cube[0]) - 1, 50)
    z = np.linspace(-len(cube[0]) + 1, len(cube[0]) - 1, 50)

    # make a dataframe of x,y,z,intensity for each point in the cube
    # to do this have to loop through the cube
    for i, x_ in enumerate(x):

        for j, y_ in enumerate(y):

            for k, z_ in enumerate(z):

                cubelist.append([x_, y_, z_, cube[i][j][k]])

    cube_df = pd.DataFrame(cubelist, columns=['x', 'y', 'z', 'intensity'])

    # only show the voxels with some intensity
    cube_df = cube_df[cube_df['intensity'] > threshold]

    fig = px.scatter_3d(cube_df, x='x', y='y', z='z',
                        color='intensity', opacity=alpha,
                        color_continuous_scale=color)
    fig.update_layout(
        scene = dict(
            xaxis = dict(range=[-25,25]),
            yaxis = dict(range=[-25,25]),
            zaxis = dict(range=[-25,25])
        )
    )
    if show:
        fig.show()
    else:
        return fig

def plot_internals(model, field, block=0, channel=0, threshold_on=True):
    """
    This function enables visualization of the output of various convolutional
    layers inside the model.
    """

    model.blocks[0].register_forward_hook(outer_block1_hook)
    model.blocks[1].register_forward_hook(outer_block2_hook)
    model.blocks[2].register_forward_hook(outer_block3_hook)
    model.blocks[3].register_forward_hook(outer_block4_hook)
    model.blocks[4].register_forward_hook(outer_block5_hook)
    model.eval()

    x = torch.autograd.Variable(field)
    feature_vec = model(x)
    blocks = [outer_block1_out,
              outer_block2_out,
              outer_block3_out,
              outer_block4_out,
              outer_block5_out]

    internal_field = blocks[block][:,channel,:,:,:].detach()
    if threshold_on:
        fig = plot_field(internal_field, show=False)
    else:
        fig = plot_field(internal_field, threshold=0.0001, show=False)
    fig.show()
