"""
This module contains functions to plot atomic density fields before they
go into a model, as well as what the density fields have been transformed into
at certain points within the model.
"""
import numpy as np
import pandas as pd
import plotly.express as px


def plot_field(field, color='ice_r', from_voxelizer=True, threshold=0.0001, alpha=0.7):
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
    fig.show()

def plot_internals(model, field):
    """
    This function enables visualization of the output of various convolutional
    layers inside the model.
    """
    
