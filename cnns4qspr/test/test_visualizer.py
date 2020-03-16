import torch
import numpy as np
import pandas as pd
import cnns4qspr.visualizer as visualizer
import plotly.express as px
import unittest
from unittest.mock import patch


import cnns4qspr.visualizer as visualizer
import cnns4qspr.loader as loader


class test_visualizer(unittest.TestCase):

    def test_plot_field(self):
        """
        This method tests the plot_field function
        """
        path = 'cnns4qspr/examples/sample_pdbs/1a00B00'
        protein_dict = loader.load_pdb(path)
        fields1 = loader.make_fields(protein_dict)
        fields2 = loader.voxelize(path)

        # makes sure that visualizer plotting function gets called properly
        with patch("cnns4qspr.visualizer.plot_field") as mock_plot_field:
            visualizer.plot_field(fields1)
            self.assertTrue(mock_plot_field.called)
