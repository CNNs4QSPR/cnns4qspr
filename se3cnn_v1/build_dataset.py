import os
import torch
import mdtraj as md
import pandas as pd
import numpy as np

### Save pdb -> class dataframe

# domain_df = pd.read_csv('cath-domain-list-S100.txt', delim_whitespace=True, header=None, usecols=[0,1], names=['Domain','Class'])
#
# pdbs = os.listdir('dompdb')
#
# purge_rows = []
# for i, row in domain_df.iterrows():
#     if row['Domain'] not in pdbs:
#         purge_rows.append(i)
#
# domain_df.drop(purge_rows, inplace=True)
# domain_df.to_pickle('domain_df.pkl')


def pdb_to_grid(pdb):

    path = 'cath/datasets/dompdb/'+pdb

    p = 2.0
    n = 50

    a = torch.linspace(start=-n / 2 * p + p / 2, end=n / 2 * p - p / 2, steps=n)

    xx = a.view(-1, 1, 1).repeat(1, len(a), len(a))
    yy = a.view(1, -1, 1).repeat(len(a), 1, len(a))
    zz = a.view(1, 1, -1).repeat(len(a), len(a), 1)

    xs = []
    ys = []
    zs = []
    atom_types = []

    with open(path, 'r') as f:
        for line in f:
            line = line.split()
            atom_types.append(line[2])
            xs.append(float(line[6]))
            ys.append(float(line[7]))
            zs.append(float(line[8]))

    atom_type_set = np.unique(atom_types)

    fields = torch.zeros(*(len(atom_type_set),)+(n, n, n))

    positions = np.zeros((len(xs),3))
    positions[:,0] = xs
    positions[:,1] = ys
    positions[:,2] = zs




    for i, atom_type in enumerate(atom_type_set):
        atom_idxs = []
        for j, type in enumerate(atom_types):
            if type == atom_type:
                atom_idxs.append(j)
        pos = positions[atom_idxs,:]
        pos = torch.FloatTensor(pos)

        xx_xx = xx.view(-1, 1).repeat(1, len(pos))
        posx_posx = pos[:,0].contiguous().view(1,-1).repeat(len(xx.view(-1)), 1)
        yy_yy = yy.view(-1, 1).repeat(1, len(pos))
        posy_posy = pos[:, 1].contiguous().view(1, -1).repeat(len(yy.view(-1)), 1)
        zz_zz = zz.view(-1, 1).repeat(1, len(pos))
        posz_posz = pos[:, 2].contiguous().view(1, -1).repeat(len(zz.view(-1)), 1)

        # Calculate density
        sigma = 0.5*p
        density = torch.exp(-((xx_xx - posx_posx)**2 + (yy_yy - posy_posy)**2 + (zz_zz - posz_posz)**2) / (2 * (sigma)**2))
        density /= torch.sum(density, dim=0)
        fields[i] = torch.sum(density, dim=1).view(xx.shape)

    labels = pd.read_pickle('domain_df.pkl')
    label = labels.loc[labels['Domain'] == pdb]['Class'].values[0]

    return fields, label

### Find pdb with most atoms

domain_df = pd.read_pickle('cath/datasets/domain_df.pkl')

max_atoms = 9951
num_samples = len(os.listdir('cath/datasets/dompdb'))
split_start_indices = np.arange(0,num_samples,int(num_samples / 10))
n_atoms = np.zeros((num_samples,))
atom_types = np.full((num_samples, max_atoms),'')
positions = np.zeros((num_samples, max_atoms, 3))
labels = np.zeros((num_samples,))

# fn = '5j1gB00'
# pdb = md.load_pdb('cath/datasets/dompdb/'+fn)
# print(pdb.n_atoms)
#
# topo = pdb.topology
# for i, atom in enumerate(topo.atoms):
#     print(i, atom.name)
#
# print(pdb.xyz.shape)


sample_id = 0
for fn in os.listdir('cath/datasets/dompdb'):
    print(sample_id, fn)

    pdb = md.load_pdb('cath/datasets/dompdb/'+fn)
    topo = pdb.topology

    n_atoms[sample_id] = pdb.n_atoms
    for i, atom in enumerate(topo.atoms):
        atom_types[sample_id, i] = atom.name

    positions[sample_id, :pdb.n_atoms, :] = pdb.xyz

    idx = domain_df.index[domain_df['Domain'] == fn].tolist()[0]
    label = domain_df.loc[idx]['Class']
    labels[sample_id] = label

    sample_id += 1

arr_dict = {'n_atoms': n_atoms, 'atom_types': atom_types, 'positions': positions, 'labels': labels, 'split_start_indices':split_start_indices}
np.savez('cath/datasets/cath_3class.npz', **arr_dict)



# data = np.load('cath/datasets/cath_3class_ca.npz')
# split_start_indices = data['split_start_indices']
# split_range = list(zip(split_start_indices[0:], list(split_start_indices[1:])+[None]))[0]
# positions = data['positions'][split_range[0]:split_range[1]]
# atom_types = data['atom_types'][split_range[0]:split_range[1]]
# n_atoms = data['n_atoms'][split_range[0]:split_range[1]]
# print(max(n_atoms))
# print(max(data['n_atoms']))
# print(positions.shape)
# print(atom_types.shape)
# for key in data.keys():
#     print(key)
# print(data['labels'].shape)

# domain_df = pd.read_pickle('cath/datasets/domain_df.pkl')
# pdbs = list(domain_df['Domain'].values)
# num_train_samples = int(len(pdbs)*0.7)
# train_set = random.sample(pdbs, num_train_samples)
# remaining = list(set(pdbs) - set(train_set))
# val_set = remaining[:int(len(remaining)/2)]
# test_set = remaining[int(len(remaining)/2):]
