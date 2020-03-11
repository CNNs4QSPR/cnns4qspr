"""
This module contains functions for loading a pdb file and calculating
the atomic density fields for different atom types. The fields can then be
used for plotting, or to send into the convolutional neural network.
"""

import torch
import numpy as np
# from biopandas.mol2 import PandasMol2
# import sys
# sys.path.append('/Users/nisargjoshi/Desktop/direct_project/cnns4qspr/cnns4qspr/se3cnn_v3')
# import dave_viz

# this is to import a protein file
# sys.path.append('/Users/nisargjoshi/Desktop/direct_project/cnns4qspr/cnns4qspr/formatting_data/sample_pdbs')

from biopandas.pdb import PandasPdb

# from

def load_pdb(path):
    """
    This function will load all of the atomic positioning/type arrays
    from a pdb file. This arrays can then be transformed into
    density (or "field") tensors before being sent through the neural
    network.

    Parameters:
    ___________

    path (str, required): The full path to the pdb file being voxelized.

    Returns:
    ________

    protein_dict (dict): A dictionary containing the following arrays from
        the pdb file: n_atoms, atom_types, positions, atom_type_set, ...
    """

    pdb = PandasPdb().read_pdb(path)


    # This just creates a dataframe from the pdb file using biopandas
    #print('This is vars',vars(pdb))
    pdf = pdb.df['ATOM']

    # atomic coordinates
    x_coords = pdf['x_coord'].values
    y_coords = pdf['y_coord'].values
    z_coords = pdf['z_coord'].values

    # create an array containing tuples of x,y,z for every atom
    positions = []
    for i, x in enumerate(x_coords):
        position_tuple = (x_coords[i], y_coords[i], z_coords[i])
        positions.append(position_tuple)
    positions = np.array(positions)

    # names of all the atoms contained in the protein
    atom_types = pdf['atom_name'].values
    num_atoms = len(atom_types)
    atom_type_set = np.unique(atom_types)
    num_atom_types = len(atom_type_set)

    # residue names
    residue_names = pdf['residue_name'].values

    protein_dict = {'x_coords':x_coords, 'y_coords':y_coords, 'z_coords':z_coords,
                    'positions':positions, 'atom_types':atom_types,
                    'num_atoms':num_atoms, 'atom_type_set':atom_type_set,
                    'num_atom_types':num_atom_types, 'residue_names':residue_names}

    # add a value to the dictionary, which is all of the atomic coordinates just
    # shifted to the origin
    protein_dict = shift_coords(protein_dict)

    return protein_dict


def shift_coords(protein_dict):
    """
    This function shifts the coordinates of a protein so that it's coordinates are in the center
    of the field tensor.

    Parameters:
    ___________
    protein_dict (dict): A dictionary of information from the first part of the load_pdb function.

    Returns:
    ________
    protein_dict (dict): Just the original protein dict but with an added value,
    """
    # find the extreme x, y, and z values that exist in the protein atomic coordinates
    x_extremes = np.array([protein_dict['x_coords'].min(), protein_dict['x_coords'].max()])
    y_extremes = np.array([protein_dict['y_coords'].min(), protein_dict['y_coords'].max()])
    z_extremes = np.array([protein_dict['z_coords'].min(), protein_dict['z_coords'].max()])

    # calculate the midpoints of the extremes
    midpoints = [np.sum(x_extremes)/2, np.sum(y_extremes)/2, np.sum(z_extremes)/2]

    # shift the coordinates by the midpoints of those extremes (center the protein on the origin)
    protein_dict['shifted_positions'] = protein_dict['positions'] - midpoints


    return protein_dict

def  grid_positions(grid_array):
    xx = grid_array.view(-1, 1, 1).repeat(1, len(grid_array), len(grid_array))
    yy = grid_array.view(1, -1, 1).repeat(len(grid_array), 1, len(grid_array))
    zz = grid_array.view(1, 1, -1).repeat(len(grid_array), len(grid_array), 1)
    return (xx, yy, zz)

def voxelize(protein_dict, channels=['CA'], bin_size=2.0, num_bins=50):
    """
    This function takes a protein dict (from load_pdb function) and outputs a
    large tensor containing many atomic "fields" for the protein. The fields
    describe the atomic "density" (an exponentially decaying function
    of number of atoms in a voxel) of any particular atom type.

    Parameters:
    ___________
    protein_dict (dict, requred): dictionary from the load_pdb function
    channels (list-like, optional): the different atomic densities we want fields for
        theoretically these different fields provide different chemical information
        full list of available channels is in protein_dict['atom_type_set']

    bin_size (float, optional): the side-length (angstrom) of a given voxel in the box
        that atomic densities are placed in

    num_bins (int, optional): how big is the cubic field tensor side length
        (i.e., num_bins is box side length)


    Returns:
    ________
    fields (numpy array or pytorch tensor): A list of atomic density tensors
        (50x50x50), one for each channel in channels
    """

    # this is weird tuple construction going into this zeros call
    # basically if the protein had 27 atom types, and we use 50 bins,
    # this ends up being zeros((37,50,50,50))
    empty_field = torch.zeros((num_bins, num_bins, num_bins))
    # this used to be fields = torch.zeros(*(len(channels),) + (num_bins, num_bins, num_bins))
    fields = {atom_type:empty_field for atom_type in channels}

    # create linearly spaced grid (default is -49 to 49 in steps of 2)
    grid_1d = torch.linspace(start=-num_bins / 2 * bin_size + bin_size / 2,
                             end=num_bins / 2 * bin_size - bin_size / 2,
                             steps=num_bins)

    # This makes three 3D meshgrids in for the x, y, and z positions
    # These cubes will be flattened and then used to normalize atomic positions
    # in the middle of a datacube
    xx, yy, zz = grid_positions(grid_1d)

    # this used to be enumerate(protein_dict['atom_type_set']), but we only want
    # a few atomic channels
    for atom_type_index, atom_type in enumerate(channels):

        # Extract positions of only the current atom type (use the origin-centered coordinates)
        atom_positions = protein_dict['shifted_positions'][protein_dict['atom_types'] == atom_type]
        atom_positions = torch.FloatTensor(atom_positions)

        # xx.view(-1, 1) is 125,000 long, because it's viewing a 50x50x50 cube in one column
        # then you repeat that column horizontally for each atom
        xx_xx = xx.view(-1, 1).repeat(1, len(atom_positions))
        yy_yy = yy.view(-1, 1).repeat(1, len(atom_positions))
        zz_zz = zz.view(-1, 1).repeat(1, len(atom_positions))
        # at this point we've created 3 arrays that are 125,000 long
        # and as wide as the number of atoms that are the current atom type
        # these 3 arrays just contain the flattened x,y,z positions of our 50x50x50 box


        # now do the same thing as above, just with the ACTUAL atomic position data
        posx_posx = atom_positions[:, 0].contiguous().view(1, -1).repeat(len(xx.view(-1)), 1)
        posy_posy = atom_positions[:, 1].contiguous().view(1, -1).repeat(len(yy.view(-1)), 1)
        posz_posz = atom_positions[:, 2].contiguous().view(1, -1).repeat(len(zz.view(-1)), 1)
        # three tensors of the same size, with actual atomic coordinates

        # normalizes the atomic positions with respect to the center of the box
        # and calculates density of atoms in each voxel
        sigma = 0.5*bin_size
        density = torch.exp(-((xx_xx - posx_posx)**2
                              + (yy_yy - posy_posy)**2
                              + (zz_zz - posz_posz)**2) / (2 * (sigma)**2))

        # Normalize so each atom density sums to one
        density /= torch.sum(density, dim=0)

        # Sum densities and reshape to original shape
        sum_densities = torch.sum(density, dim=1).view(xx.shape)

        # set all nans to 0
        sum_densities[sum_densities != sum_densities] = 0

        # add an empty dimmension to make it 1x50x50x50
        sum_densities = sum_densities.unsqueeze(0)
        sum_densities = sum_densities.unsqueeze(0)

        #fields[atom_type_index] = sum_densities
        fields[atom_type] = sum_densities


    return fields

class loader():
    """
    docstring
    """

    def __init__(self, path_to_pdb_file):
        """
        docstring
        """

    def __getitem__(self, channel='CA'):
        """
        docstring
        """
        fields = voxelize(self.protein_dict)
        return fields[channel]

def main():
    """
    this is main method to call the defined functions
    """
    protein_path = '/Users/nisargjoshi/Desktop/direct_project/cnns4qspr/cnns4qspr/formatting_data/sample_pdbs/6fww.pd'
    protein_dict = load_pdb(protein_path)
    fields = voxelize(protein_dict)
    dave_viz.plot_field(fields['CA'])

# main()
