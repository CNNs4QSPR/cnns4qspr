

# class CathData(torch.utils.data.Dataset):

def load_pdb(file):

    discretization_bins=50
    discretization_bin_size=2.0
    # discretization_bins = discretization_bins
    # discretization_bin_size = discretization_bin_size

    ppdb = PandasPdb().fetch_pdb('3eiy')


    openf=ppdb.read_pdb(file)
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

    positions=[]
    for i in range(len(x)):
        for j in range(len(x[i])):
            ls=[x[i][j], y[i][j], z[i][j]]
            positions.append(ls)

    atom_types=openf.df['ATOM']['atom_name'].values
    atom_numbers=openf.df['ATOM']['atom_number'].values

    #def __getitem__(self, index):

    # n_atoms=self.atom_numbers[ind]
    # positions=self.coords[index][:n_atoms]
    # atom_types
    #
    p=discretization_bin_size
    n=discretization_bins


    atom_type_set=np.unique(atom_types)
    a=len(atom_type_set)
    if torch.cuda.is_available():
        fields=torch.cuda.FloatTensor(*(a,)+(n,n,n)).fill_(0)
    else:
        fields=torch.zeros(*(a,)+(n,n,n))

    a = torch.linspace(start=-n / 2 * p + p / 2, end=n / 2 * p - p / 2, steps=n)
#     if torch.cuda.is_available():
#         a = a.cuda()


    xx = a.view(-1, 1, 1).repeat(1, len(a), len(a))
    yy = a.view(1, -1, 1).repeat(len(a), 1, len(a))
    zz = a.view(1, 1, -1).repeat(len(a), len(a), 1)

#     atom_type_set=np.unique(atom_types[atom_numbers])

     ##changing the atom_type_set for now using just a single input
    for i, atom_type in enumerate(['CA']):
        print(atom_types)
        print(type(atom_types))
        print(atom_types == 'CA')
        print('This is the type of positions',type(positions))
        positions = np.array(positions)
        pos = positions[atom_types == atom_type]
#         pos = positions[0] #changing pos here
        pos = torch.FloatTensor(pos)
#         if torch.cuda.is_available():
#             pos=pos.cuda()

        xx_xx = xx.view(-1,1).repeat(1, len(pos))
        posx_posx = pos[:, 0].contiguous().view(1, -1).repeat(len(xx.view(-1)),1)
        yy_yy = yy.view(-1, 1).repeat(1, len(pos))
        posy_posy = pos[:, 1].contiguous().view(1, -1).repeat(len(yy.view(-1)), 1)
        zz_zz = zz.view(-1, 1).repeat(1, len(pos))
        posz_posz = pos[:, 2].contiguous().view(1, -1).repeat(len(zz.view(-1)), 1)

        sigma = 0.5*p
        density=torch.exp(-((xx_xx - posx_posx)**2 + (yy_yy - posy_posy)**2 + (zz_zz - posz_posz)**2) / (2 * (sigma)**2))

        density /= torch.sum(density, dim=0)

        fields[i] = torch.sum(density, dim=1).view(xx.shape)

    return fields
