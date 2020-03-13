import torch
import numpy as np
import pandas as pd
import plotly.express as px
import unittest
from unittest import mock


import cnns4qspr.visualizer as visualizer
import cnns4qspr.loader as loader


class test_visualizer(unittest.TestCase):

    def test_plot_field():
        path = 'cnns4qspr/formatting_data/sample_pdbs/1a00B00'
        protein_dict = loader.load_pdb(path)
        fields1 = loader.make_fields(protein_dict)
        fields2 = loader.voxelize(path)

        plot_obj1 = visualizer.plot_field(fields1['CA'], show=False)
        plot_obj2 = visualizer.plot_field(fields2['CA'], show=False)

        self.assertTrue(True)
        self.assertTrue(True)

        return
