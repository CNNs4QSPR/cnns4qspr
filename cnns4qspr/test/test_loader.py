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
        path = 'cnns4qspr/examples/sample_pdbs/1a00B00'
        dict_protein = loader.load_pdb(path)
        self.assertTrue(dict_protein['positions'][0][0]==dict_protein['x_coords'][0])
        self.assertTrue(len(dict_protein['atom_types'])>len(dict_protein['atom_type_set']))
        return dict_protein


    def test_shift_coords(self):
        """
        This is for testing the shift_coords function.
        """
        # protein_dict = loader.load_pdb(path)
        self.path = 'cnns4qspr/examples/sample_pdbs/1a00B00'
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
        path = 'cnns4qspr/examples/sample_pdbs/1a00B00'
        num_bins = 50
        bin_size = 2
        protein_dict = loader.load_pdb(path)
        # this both make 'CA' field as default
        fields1 = loader.voxelize(path)
        fields2 = loader.make_fields(protein_dict)

        self.assertTrue(fields1['CA'].shape == torch.Size([1, 1, 50, 50, 50]))
        self.assertTrue(fields1['CA'].dtype == torch.float32)

        # check that the output of voxelizer, and the output of load_pdb >> make_fields
        # is the same in various ways
        self.assertEqual(len(fields1['CA']), len(fields2['CA']))
        self.assertEqual(len(fields1['CA'][0][0]), len(fields2['CA'][0][0]))
        self.assertEqual(fields1['CA'].shape, fields2['CA'].shape)


    def test_grid_positions(self):
        """
        This unit test is for testing the grid_positions function.
        """
        self.path = 'cnns4qspr/examples/sample_pdbs/1a00B00'
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


    def test_make_fields(self):
        """
        This is for testing the make_fields function.
        """
        self.path = 'cnns4qspr/examples/sample_pdbs/1a00B00'
        self.protein_dict = loader.load_pdb(self.path)
        self.num_bins = 50
        self.bin_size = 2
        output_fields = loader.make_fields(self.protein_dict)
        self.assertTrue(output_fields['CA'].shape == torch.Size([1, 1, 50, 50, 50]))
        self.assertTrue(output_fields['CA'].dtype == torch.float32)


    def test_check_channel(self):
        """
        This function tests the check_channel() function
        """
        # sets of allowed filters to build channels with
        path = 'cnns4qspr/examples/sample_pdbs/1a00B00'
        protein_dict = loader.load_pdb(path)
        residue_filters = protein_dict['residue_set']
        atom_filters    = protein_dict['atom_type_set']
        residue_property_filters = np.array(['acidic', 'basic', 'polar', 'nonpolar',\
                                             'charged', 'amphipathic'])
        other_filters = np.array(['backbone', 'sidechains'])

        # consolidate into one set of filters
        filter_set = {'atom':atom_filters, 'residue':residue_filters,\
                      'residue_property':residue_property_filters, 'other':other_filters}

        self.assertFalse(loader.check_channel('strawberries', filter_set))
        self.assertTrue(loader.check_channel('polar', filter_set))
        self.assertTrue(loader.check_channel('backbone', filter_set))
        self.assertTrue(loader.check_channel('LEU', filter_set))

    def test_find_channel_atoms(self):
        """
        This method tests the find_channel_atoms function
        """
        # sets of allowed filters to build channels with
        path = 'cnns4qspr/examples/sample_pdbs/6fww.pdb'
        protein_dict = loader.load_pdb(path)
        residue_filters = protein_dict['residue_set']
        atom_filters    = protein_dict['atom_type_set']
        residue_property_filters = np.array(['acidic', 'basic', 'polar', 'nonpolar',\
                                             'charged', 'amphipathic'])
        other_filters = np.array(['backbone', 'sidechains'])

        filter_set = {'atom':atom_filters, 'residue':residue_filters,\
                      'residue_property':residue_property_filters, 'other':other_filters}

        # test 1: assert that fin_channel_atoms is constructing the channels for all of these
        # residues correctly
        for residue in ['LYS', 'MET', 'VAL', 'PRO', 'ALA', 'SER', 'LEU', 'ILE', 'ARG', 'ASP']:
            true_res_atoms = protein_dict['shifted_positions'][protein_dict['residues'] == residue]
            test_res_atoms = loader.find_channel_atoms(residue, protein_dict, filter_set)
            self.assertEqual(true_res_atoms.all(), test_res_atoms.all())

    def test_atoms_from_residues(self):
        """
        This method tests the atoms_from_residues function
        """
        path = 'cnns4qspr/examples/sample_pdbs/6fww.pdb'
        protein_dict = loader.load_pdb(path)
        residue_filters = protein_dict['residue_set']
        atom_filters    = protein_dict['atom_type_set']
        residue_property_filters = np.array(['acidic', 'basic', 'polar', 'nonpolar',\
                                             'charged', 'amphipathic'])
        other_filters = np.array(['backbone', 'sidechains'])

        filter_set = {'atom':atom_filters, 'residue':residue_filters,\
                      'residue_property':residue_property_filters, 'other':other_filters}

        # test 1: assert that fin_channel_atoms is constructing the channels for all of these
        # residues correctly
        for residue in ['LYS', 'MET', 'VAL', 'PRO', 'ALA', 'SER', 'LEU', 'ILE', 'ARG', 'ASP']:
            true_res_atoms = protein_dict['shifted_positions'][protein_dict['residues'] == residue]
            test_res_atoms = loader.find_channel_atoms(residue, protein_dict, filter_set)
            self.assertEqual(true_res_atoms.all(), test_res_atoms.all())



    if __name__ == '__main__':
        unittest.main()
