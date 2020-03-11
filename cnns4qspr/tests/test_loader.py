import torch
import numpy as np
from biopandas.pdb import PandasPdb
import unittest
import sys
sys.path.insert(1,'/Users/nisargjoshi/Desktop/direct_project/cnns4qspr/cnns4qspr')
import loader

path = '/Users/nisargjoshi/Desktop/direct_project/cnns4qspr/cnns4qspr/formatting_data/sample_pdbs/1a00B00'

def test_load_pdb():
    """
    This is for testing the load_pdb function.
    """

    dict_protein = loader.load_pdb(path)
    assert dict_protein['positions'][0][0]==dict_protein['x_coords'][0]
    assert len(dict_protein['atom_types'])>len(dict_protein['atom_type_set'])
    return dict_protein

def test_shift_coords():
    """
    This is for testing the shift_coords function.
    """
    protein_dict = loader.load_pdb(path)
    new_pos = loader.shift_coords(protein_dict)
    assert new_pos['shifted_positions'].mean() < protein_dict['positions'].mean()
    return new_pos

def test_voxelize():
    """
    This unit test is for testing the voxelize funtion.
    """
    protein_dict = loader.load_pdb(path)
    dfields = loader.voxelize(path)
    assert dfields['CA'].shape == torch.Size([1, 1, 50, 50, 50])
    return dfields
