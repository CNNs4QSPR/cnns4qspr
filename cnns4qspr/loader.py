"""
This module contains functions for loading a pdb file and calculating
the atomic density fields for different atom types. The fields can then be
used for plotting, or to send into the convolutional neural network.
"""

import torch
import numpy as np

from biopandas.pdb import PandasPdb

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
    residue_set = np.unique(residue_names)

    protein_dict = {'x_coords':x_coords, 'y_coords':y_coords, 'z_coords':z_coords,
                    'positions':positions, 'atom_types':atom_types,
                    'num_atoms':num_atoms, 'atom_type_set':atom_type_set,
                    'num_atom_types':num_atom_types, 'residues':residue_names,
                    'residue_set':residue_set}

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

def grid_positions(grid_array):
    xx = grid_array.view(-1, 1, 1).repeat(1, len(grid_array), len(grid_array))
    yy = grid_array.view(1, -1, 1).repeat(len(grid_array), 1, len(grid_array))
    zz = grid_array.view(1, 1, -1).repeat(len(grid_array), len(grid_array), 1)
    return (xx, yy, zz)

def make_fields(protein_dict, channels=['CA'], bin_size=2.0, num_bins=50):
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
    # list of
    allowed_residues = protein_dict['residue_set']

    allowed_atoms = protein_dict['atom_type_set']

    allowed_filters = np.array(['backbone', 'sidechains'])

    # construct a single empty field, then initialize a dictionary with one
    # empty field for every channel we are going to calculate the density for
    empty_field = torch.zeros((num_bins, num_bins, num_bins))
    fields = {channel:empty_field for channel in channels}

    # create linearly spaced grid (default is -49 to 49 in steps of 2)
    grid_1d = torch.linspace(start=-num_bins / 2 * bin_size + bin_size / 2,
                             end=num_bins / 2 * bin_size - bin_size / 2,
                             steps=num_bins)

    # This makes three 3D meshgrids in for the x, y, and z positions
    # These cubes will be flattened, then used as a reference coordinate system
    # to place the actual channel densities into
    xx, yy, zz = grid_positions(grid_1d)

    for channel_index, channel in enumerate(channels):

        # no illegal channels allowed, assume the channel sucks
        channel_allowed = False
        for channel_type in [allowed_atoms, allowed_residues, allowed_filters]:
            if channel in channel_type:
                channel_allowed = True

        if channel_allowed:
            pass
        else:
            raise ValueError('The channel ', channel, ' is not allowed for this protein.',\
                             'Allowed channels are: in a protein\'s atom_type_set, residue_set',\
                             ' or the \'sidechains\' and \'backbone\' channels.')


        # Extract positions of atoms that are part of the current channel
        if channel in allowed_atoms:
            atom_positions = protein_dict['shifted_positions'][protein_dict['atom_types'] == channel]

        elif channel in allowed_residues:
            atom_positions = protein_dict['shifted_positions'][protein_dict['residues'] == channel]

        else: # the channel is an agregate filter, either backbone or sidechains
            if channel == 'backbone':
                # create boolean arrays for backbone atoms
                bool_O = protein_dict['atom_types'] == 'O'
                bool_C = protein_dict['atom_types'] == 'C'
                bool_CA = protein_dict['atom_types'] == 'CA'
                bool_N = protein_dict['atom_types'] == 'N'

                # sum of all the backbone channels into one boolean array
                bool_backbone = bool_O + bool_C + bool_CA + bool_N

                # select the backbone atoms
                atom_positions = protein_dict['shifted_positions'][bool_backbone]

            else: # it was 'sidechain' filter, so grab sidechain atoms
                backbone_atom_set = np.array(['O', 'C', 'CA', 'N'])
                sidechain_atom_set = np.array([atom for atom in protein_dict['atom_type_set'] \
                                               if atom not in backbone_atom_set])

                for index, sidechain_atom in enumerate(sidechain_atom_set):
                    if index == 0:
                        # create the first sidechains boolean array, will be edited
                        bool_sidechains = protein_dict['atom_types'] == sidechain_atom
                    else:
                        # single boolean array for the current sidechain atom
                        bool_atom = protein_dict['atom_types'] == sidechain_atom

                        # sum this boolean array with the master boolean array
                        bool_sidechains += bool_atom

                # grab all sidechain atom positions
                atom_positions = protein_dict['shifted_positions'][bool_sidechains]




        atom_positions = torch.FloatTensor(atom_positions)

        # xx.view(-1, 1) is 125,000 long, because it's viewing a 50x50x50 cube in one column
        # then you repeat that column horizontally for each atom
        xx_xx = xx.view(-1, 1).repeat(1, len(atom_positions))
        yy_yy = yy.view(-1, 1).repeat(1, len(atom_positions))
        zz_zz = zz.view(-1, 1).repeat(1, len(atom_positions))
        # at this point we've created 3 arrays that are 125,000 long
        # and as wide as the number of atoms that are the current channel type
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

        # add two empty dimmensions to make it 1x1x50x50x50, needed for CNN
        sum_densities = sum_densities.unsqueeze(0)
        sum_densities = sum_densities.unsqueeze(0)

        #fields[atom_type_index] = sum_densities
        fields[channel] = sum_densities


    return fields

def voxelize(path, channels=['CA']):
    """
    This function creates a dictionary of tensor fields directly from a pdb file.
    These tensor fields can be plotted, or sent directly into the cnn for
    plotting internals, or sent all the way through a cnn/vae to be used for
    training.

    Parameters:
    ___________
    path (str, required): path to a .pdb file

    channels (list, optional): The list of atomic channels to be included in
        the output dictionary, one field for every channel. Channel options are:
        ['C' 'CA' 'CB' 'CD' 'CD1' 'CD2' 'CE' 'CE1' 'CE2' 'CE3' 'CG' 'CG1' 'CG2'
         'CH2' 'CZ' 'CZ2' 'CZ3' 'N' 'ND1' 'ND2' 'NE' 'NE1' 'NE2' 'NH1' 'NH2' 'NZ'
         'O' 'OD1' 'OD2' 'OE1' 'OE2' 'OG' 'OG1' 'OH' 'OXT' 'SD' 'SG']

        Channels with good diversity of densities include: C/CA, CD, NH2, OH, SD/SG

    Returns:
    ________
    a dictionary of fields, each one with shape =  ([1, 1, 50, 50, 50])

    """
    protein_dict = load_pdb(path)
    return make_fields(protein_dict, channels=channels)
