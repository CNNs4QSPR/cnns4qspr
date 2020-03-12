import torch
import numpy as np
import pandas as pd
import plotly.express as px
import unittest
from unittest import mock


import cnns4qspr.visualizer as visualizer
import cnns4qspr.loader as loader


class test_visualizer(unittest.TestCase):

    self.path = 'cnns4qspr/formatting_data/sample_pdbs/1a00B00'
    self.protein_dict = loader.load_pdb(self.path)
    self.fields = loader.make_fields(self.protein_dict)
    def test_plot_field():
        showf = visualizer.plot_field(self.fields)
        assert mock_plt.figure.called
        return showf
