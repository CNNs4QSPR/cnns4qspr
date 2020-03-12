import torch
import numpy as np
from biopandas.pdb import PandasPdb
import unittest

import cnns4qspr.loader as loader


class test_loader(unittest.TestCase):



    def test_load_pdb(self):
        """
        This is for testing the load_pdb function.
        """
        self.path = 'cnns4qspr/formatting_data/sample_pdbs/1a00B00'
        self.protein_dict = loader.load_pdb(self.path)
        self.num_bins = 50
        self.bin_size = 2
        dict_protein = loader.load_pdb(self.path)
        self.assertTrue(dict_protein['positions'][0][0]==dict_protein['x_coords'][0])
        self.assertTrue(len(dict_protein['atom_types'])>len(dict_protein['atom_type_set']))
        return dict_protein

    def test_shift_coords(self):
        """
        This is for testing the shift_coords function.
        """
        # protein_dict = loader.load_pdb(path)
        self.path = 'cnns4qspr/formatting_data/sample_pdbs/1a00B00'
        self.protein_dict = loader.load_pdb(self.path)
        self.num_bins = 50
        self.bin_size = 2
        new_pos = loader.shift_coords(self.protein_dict)
        self.assertTrue(new_pos['shifted_positions'].mean() < self.protein_dict['positions'].mean())
        return new_pos

    def test_voxelize(self):
        """
        This unit test is for testing the voxelize funtion.
        """
        # protein_dict = loader.load_pdb(self.path)
        self.path = 'cnns4qspr/formatting_data/sample_pdbs/1a00B00'
        self.protein_dict = loader.load_pdb(self.path)
        self.num_bins = 50
        self.bin_size = 2
        dfields = loader.voxelize(self.path)
        self.assertTrue(dfields['CA'].shape == torch.Size([1, 1, 50, 50, 50]))
        self.assertTrue(dfields['CA'].dtype == torch.float32)
        # assert dfields['CA'].device == torch.device('cpu')
        return dfields

    def test_grid_positions(self):
        """
        This unit test is for testing the grid_positions function.
        """
        self.path = 'cnns4qspr/formatting_data/sample_pdbs/1a00B00'
        self.protein_dict = loader.load_pdb(self.path)
        self.num_bins = 50
        self.bin_size = 2
        grid_array = torch.linspace(start=-self.num_bins / 2 * self.bin_size + self.bin_size / 2,
                                 end=self.num_bins / 2 * self.bin_size - self.bin_size / 2,
                                 steps=self.num_bins)
        grid_out = loader.grid_positions(grid_array)
        self.assertTrue(len(grid_out[0]) == 50)
        self.assertTrue(len(grid_out[1]) == 50)
        self.assertTrue(len(grid_out[2]) == 50)
        self.assertTrue(grid_out[0].shape == torch.Size([50, 50, 50]))
        self.assertTrue(grid_out[1].shape == torch.Size([50, 50, 50]))
        self.assertTrue(grid_out[2].shape == torch.Size([50, 50, 50]))
        return grid_out

    def test_make_fields(self):
        """
        This is for testing the make_fields function.
        """
        self.path = 'cnns4qspr/formatting_data/sample_pdbs/1a00B00'
        self.protein_dict = loader.load_pdb(self.path)
        self.num_bins = 50
        self.bin_size = 2
        output_fields = loader.make_fields(self.protein_dict)
        self.assertTrue(output_fields['CA'].shape == torch.Size([1, 1, 50, 50, 50]))
        self.assertTrue(output_fields['CA'].dtype == torch.float32)
        return output_fields
