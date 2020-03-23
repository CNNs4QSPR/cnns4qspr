"""
This module contains functions for loading a pdb file and calculating
the atomic density fields for different atom types. The fields can then be
used for plotting, or to send into the convolutional neural network.
"""

import os
import sys
import torch
import numpy as np

from biopandas.pdb import PandasPdb

def load_pdb(path):
    """
    Loads all of the atomic positioning/type arrays from a pdb file.

    The arrays can then be transformed into density (or "field") tensors before
        being sent through the neural network.

    Parameters:
        path (str, required): The full path to the pdb file being voxelized.

    Returns:
        dictionary: A dictionary containing the following arrays from
            the pdb file: num_atoms, atom_types, positions, atom_type_set,
            xcoords, ycoords, zcoords, residues, residue_set
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
        protein_dict (dict): A dictionary of information from the first part of the load_pdb function.

    Returns:
        dictionary: The original protein dict but with an added value containing
            the coordinates of the protein shifted to the origin.
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
    """
    This function returns the 3D meshgrids of x, y and z positions.
    These cubes will be flattened, then used as a reference coordinate system
    to place the actual channel densities into.

    Parameters:
        grid_positions (pytorch tensor): lineraly spaced grid

    Returns:
        array: meshgrid array of the x, y and z positions.
    """
    xgrid = grid_array.view(-1, 1, 1).repeat(1, len(grid_array), len(grid_array))
    ygrid = grid_array.view(1, -1, 1).repeat(len(grid_array), 1, len(grid_array))
    zgrid = grid_array.view(1, 1, -1).repeat(len(grid_array), len(grid_array), 1)
    return (xgrid, ygrid, zgrid)

def make_fields(protein_dict, channels=['CA'], bin_size=2.0, num_bins=50, return_bins=False):
    """
    This function takes a protein dict (from load_pdb function) and outputs a
        large tensor containing many atomic "fields" for the protein.

    The fields describe the atomic "density" (an exponentially decaying function
        of number of atoms in a voxel) of any particular atom type.

    Parameters:
        protein_dict (dict, requred): dictionary from the load_pdb function

        channels (list-like, optional): the different atomic densities we want fields for
            theoretically these different fields provide different chemical information
            full list of available channels is in protein_dict['atom_type_set']

        bin_size (float, optional): the side-length (angstrom) of a given voxel in the box
            that atomic densities are placed in

        num_bins (int, optional): how big is the cubic field tensor side length
            (i.e., num_bins is box side length)


    Returns:
        dictionary: A list of atomic density tensors (50x50x50), one for each
            channel in channels
    """
    # sets of allowed filters to build channels with
    residue_filters = protein_dict['residue_set']
    atom_filters = protein_dict['atom_type_set']
    residue_property_filters = np.array(['acidic', 'basic', 'polar', 'nonpolar',\
                                         'charged', 'amphipathic'])
    other_filters = np.array(['backbone', 'sidechains'])

    # consolidate into one set of filters
    filter_set = {'atom':atom_filters, 'residue':residue_filters,\
                  'residue_property':residue_property_filters, 'other':other_filters}

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
    xgrid, ygrid, zgrid = grid_positions(grid_1d)

    for channel_index, channel in enumerate(channels):

        # no illegal channels allowed, assume the channel sucks
        channel_allowed = check_channel(channel, filter_set)

        if channel_allowed:
            pass
        else:
            #err_string = 'Allowed channels are: in a protein\'s atom_type_set,
                        # residue_set',or the \'sidechains\' and \'backbone\' channels.'
            raise ValueError('The channel ', channel, ' is not allowed for this protein.')


        # Extract positions of atoms that are part of the current channel
        atom_positions = find_channel_atoms(channel, protein_dict, filter_set)
        # print('This is channel ', channel)
        atom_positions = torch.FloatTensor(atom_positions)

        # xgrid.view(-1, 1) is 125,000 long, because it's viewing a 50x50x50 cube in one column
        # then you repeat that column horizontally for each atom
        xx_xx = xgrid.view(-1, 1).repeat(1, len(atom_positions))
        yy_yy = ygrid.view(-1, 1).repeat(1, len(atom_positions))
        zz_zz = zgrid.view(-1, 1).repeat(1, len(atom_positions))
        # at this point we've created 3 arrays that are 125,000 long
        # and as wide as the number of atoms that are the current channel type
        # these 3 arrays just contain the flattened x,y,z positions of our 50x50x50 box


        # now do the same thing as above, just with the ACTUAL atomic position data
        posx_posx = atom_positions[:, 0].contiguous().view(1, -1).repeat(len(xgrid.view(-1)), 1)
        posy_posy = atom_positions[:, 1].contiguous().view(1, -1).repeat(len(ygrid.view(-1)), 1)
        posz_posz = atom_positions[:, 2].contiguous().view(1, -1).repeat(len(zgrid.view(-1)), 1)
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
        sum_densities = torch.sum(density, dim=1).view(xgrid.shape)

        # set all nans to 0
        sum_densities[sum_densities != sum_densities] = 0

        # add two empty dimmensions to make it 1x1x50x50x50, needed for CNN
        # sum_densities = sum_densities.unsqueeze(0)
        # sum_densities = sum_densities.unsqueeze(0)

        #fields[atom_type_index] = sum_densities
        fields[channel] = sum_densities.numpy()

    if return_bins:
        return fields, num_bins
    else:
        return fields

def check_channel(channel, filter_set):
    """
    This function checks to see if a channel the user is asking to make a field
        for is an allowed channel to ask for.

    Parameters:
        channel (str, required): The atomic channel being requested

        filter_set (dict, required): The set of defined atomic filters

    Returns:
        boolean: indicator for if the channel is allowed
    """
    channel_allowed = False
    for key in filter_set:
        if channel in filter_set[key]:
            channel_allowed = True

    return channel_allowed

def find_channel_atoms(channel, protein_dict, filter_set):
    """
    This function finds the coordinates of all relevant atoms in a channel.

    It uses the filter set to constrcut the atomic channel (i.e., a channel can
        be composed of multiple filters).

    Parameters:
        channel (str, required): The atomic channel being constructed

        protein_dict (dict, required): The dictionary of the protein, returned from
            load_pdb()

        filter_set (dict, required): The set of available filters to construct channels with

    Returns:
        numpy array:  array containing the coordinates of each atom that is relevant
            to the channel
    """
    if channel in filter_set['atom']:
        atom_positions = protein_dict['shifted_positions'][protein_dict['atom_types'] == channel]

    elif channel in filter_set['residue']:
        atom_positions = protein_dict['shifted_positions'][protein_dict['residues'] == channel]

    elif channel in filter_set['other']: # backbone or sidechain
        if channel == 'backbone':
            # create boolean arrays for backbone atoms
            bool_oxygen = protein_dict['atom_types'] == 'O'
            bool_carbon = protein_dict['atom_types'] == 'C'
            bool_alphacarbon = protein_dict['atom_types'] == 'CA'
            bool_nitrogen = protein_dict['atom_types'] == 'N'

            # sum of all the backbone channels into one boolean array
            bool_backbone = bool_oxygen + bool_carbon + bool_alphacarbon + bool_nitrogen

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

    else: # it was a residue property channel
        acidic_residues = np.array(['ASP', 'GLU'])
        basic_residues = np.array(['LYS', 'ARG', 'HIS'])
        polar_residues = np.array(['GLN', 'ASN', 'HIS', 'SER', 'THR', 'TYR', 'CYS'])
        nonpolar_residues = np.array(['GLY', 'ALA', 'VAL', 'LEU', \
                                        'ILE', 'MET', 'PRO', 'PHE', 'TRP'])
        amphipathic_residues = np.array(['TRP', 'TYR', 'MET'])
        charged_residues = np.array(['ARG', 'LYS', 'ASP', 'GLU'])
        # custom_residues = something
        property_dict = {'acidic':acidic_residues, 'basic':basic_residues,\
                         'polar':polar_residues, 'nonpolar':nonpolar_residues,\
                         'amphipathic':amphipathic_residues, 'charged':charged_residues}

        atom_positions = atoms_from_residues(protein_dict, property_dict[channel])

    return atom_positions

def atoms_from_residues(protein_dict, residue_list):
    """
    This function finds all the atoms in a protein that are members of any residues
    in a list of residues.

    Parameters:
        protein_dict (dict, required): The dictionary of the protein, returned from
            load_pdb()

        residue_list (list-like, required): The list of residues whose atoms we are
            finding coordinates for
    """
    # construct the appropriate boolean array to index the atoms in the protein_dict
    for index, residue in enumerate(residue_list):
        if index == 0:
            bool_residue = protein_dict['residues'] == residue
        else:
            bool_residue += protein_dict['residues'] == residue

    atom_positions = protein_dict['shifted_positions'][bool_residue]

    return atom_positions

def voxelize(path, channels=['CA'], path_type='file', save=False, save_fn='voxels.npy', save_path='./'):
    """
    This function creates a dictionary of tensor fields directly from a pdb file.

    These tensor fields can be plotted, or sent directly into the cnn for
        plotting internals, or sent all the way through a cnn/vae to be used for
        training.

    Parameters:
        path (str, required): path to a .pdb file

        channels (list of strings, optional): The list of atomic channels to be included in
            the output dictionary, one field for every channel.

            Any channels from points 1-4 below may be combined in any order.
            i.e., one could call voxelize with the channels parameter as
            channels=['charged', 'CB', 'GLY', 'polar', ...etc]. Note that voxelization
            for channels containing more atoms will take longer.

            1. any of the following atom types

                ['C' 'CA' 'CB' 'CD' 'CD1' 'CD2' 'CE' 'CE1' 'CE2' 'CE3' 'CG' 'CG1' 'CG2'
                'CH2' 'CZ' 'CZ2' 'CZ3' 'N' 'ND1' 'ND2' 'NE' 'NE1' 'NE2' 'NH1' 'NH2' 'NZ'
                'O' 'OD1' 'OD2' 'OE1' 'OE2' 'OG' 'OG1' 'OH' 'OXT' 'SD' 'SG']

            2. Any canonical residue in the protein, using the three letter residue code, all caps
               (NOTE: the residue must actually exist in the protein)

                e.g., ['LYS', 'LEU', 'ALA']

            3. The 'other' channel options: 'backbone', 'sidechains'

            4. There are 6 channels corresponding to specific types of residues:

                'charged', 'polar', 'nonpolar', 'amphipathic', 'acidic', 'basic'

    Returns:
        dictionary: a dictionary containing a voxelized atomic fields, one for each
        channel requested. Each field has shape = ([1, 1, 50, 50, 50])

    """
    if path_type == 'file':
        protein_dict = load_pdb(path)
        return make_fields(protein_dict, channels=channels)

    # ------------FOLDER FUNCTIONALITY INCOMPLETE---------------
    elif path_type == 'folder':
        fields = []
        pdb_fns = os.listdir(path)
        for j, fn in enumerate(pdb_fns):
            progress = '{}/{} pdbs voxelized ({}%)'.format(j, len(pdb_fns), \
                        round(j / len(pdb_fns) * 100, 2))
            sys.stdout.write('\r'+progress)
            protein_dict = load_pdb(os.path.join(path, fn))
            field, bins = make_fields(protein_dict, channels=channels, return_bins=True)
            channel_list = []
            for channel in channels:
                channel_list.append(field[channel].reshape(1,
                                                           bins,
                                                           bins,
                                                           bins))
            field = np.concatenate(channel_list).reshape(1,len(channels),
                                                         bins, bins, bins)
            fields.append(field)
        sys.stdout.write("\r\033[K")
        out_statement = 'voxelization complete!\n'
        sys.stdout.write('\r'+out_statement)
        fields = np.concatenate(fields)

        if save:
            np.save(os.path.join(save_path, save_fn), fields)
        else:
            return fields
