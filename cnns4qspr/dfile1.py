import torch
import os
import numpy as np
import pandas as pd
from biopandas.mol2 import PandasMol2
from biopandas.pdb import PandasPdb
from scipy.stats import special_ortho_group

# class CathData(torch.utils.data.Dataset):

    def load_pdb(file,discretization_bins=50, discretization_bin_size=2.0, use_density=True):

        self.discretization_bins = discretization_bins
        self.discretization_bin_size = discretization_bin_size
        self.use_density = use_density

        ppdb = PandasPdb().fetch_pdb('3eiy')


        openf=ppdb.read_pdb('file')
        openf.df['ATOM'].head()

        x=[]
        y=[]
        z=[]
        xcoord=openf.df['ATOM']['x_coord'].values.tolist()
        ycoord=openf.df['ATOM']['y_coord'].values.tolist()
        zcoord=openf.df['ATOM']['z_coord'].values.tolist()
        x.append(xcoord)
        y.append(ycoord)
        z.append(zcoord)

        corods=[]
        for i in range(len(x)):
            for j in range(len(x[i])):
                ls=[x[i][j], y[i][j], z[i][j]]
                coords.append(ls)

        atom_types=openf.df['ATOM']['atom_name'].values
        atom_numbers=openf.df['ATOM']['atom_number'].values

    #def __getitem__(self, index):

        n_atoms=self.atom_numbers[ind]
        positions=self.coords[index][:n_atoms]
        atom_types

        p=self.discretization_bin_size
        n=self.discretization_bins

        if torch.cuda.is_available():
            fields=torch.cuda.FloatTensor(*(self.n))
