[![Build Status](https://travis-ci.org/CNNs4QSPR/cnns4qspr.svg?branch=master)](https://travis-ci.org/CNNs4QSPR/cnns4qspr) [![Coverage Status](https://coveralls.io/repos/github/CNNs4QSPR/cnns4qspr/badge.svg?branch=master)](https://coveralls.io/github/CNNs4QSPR/cnns4qspr?branch=master) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# cnns4qspr

A package for creating rich, equivariant, structural and chemical features from protein structure data.

## Overview

Scientists are continually finding applications for machine learning in all branches of science, and the field of structural biology is no exception. The purpose of the cnns4qspr package is to make extraction of high quality features from 3D protein structures as easy as possible for a user. Once a user has their desired features, they may use them for whatever machine learning task they desire.

#### Who can make use of this package:

This package is great for anyone trying to investigate quantitative structure-property relationships (QSPR) in proteins. Some examples include researchers studying de novo design, protein crystal-solvent interactions, solid interactions, and protein-ligand interactions. Generally speaking, anyone wanting to map protein-crystal structures to a property may find cnns4qspr useful.

#### Feature vector:

1.	The user must input the path of the pdb file to the functions featurize or gen_feature_set from featurizer.py.
2.	The function would return a set of feature vectors based on the channels specified.

#### Uses:

Compression of protein structural data into a feature vector. This can be used to convert pdb protein data in a structural information-dense vector space. This structural information vector can be used for:
1.	Training models for Structural classification prediction. (See examples)
2.	Reducing the computation expense for structure-to-property predictions.
3.	Decoders for identifying the features of the amino acid residues primarily responsible for the secondary structure.
4.	Training models for structure prediction in different solutions/environments.
5.	Recommender systems for protein sequence prediction.

### Visual demonstrations

cnns4qspr "voxelizes" protein structure data, so that data are in a form which is acceptable as input to a 3D convolutional neural network (CNN). Voxelization simply means the atomic coordinates are transformed from descrete points in 3D space, to slightly smeared atomic densities that fill "voxels" (3D pixels) in a new 3D picture of the protein.

#### Voxelization of protein data
Here, we demonstrate what voxelization of all the backbone atoms in green flourescent protein (GFP) results in. Clearly, the molecular sctructure of GFP is maintained throughout the transformation. This tells us the network will be able to "see" key structural information unique to GFP.

<p align="center">
<img align="middle" src="cnns4qspr/figs/backbone_exploded.gif" width="475" height="375" >
</p>

#### Flexibility of feature extraction

It is well known in machine learning that a model will only be as good as the data you feed it. Therefore, in a regression or classification task using protein structures as the input, it is essential to engineer the data in such a way that the most relevant features for the task at hand are highlighted. As an example, consider a task of binary classification: is the protein a membrane protein, or not? In order to answer this question accurately, a model probably needs to see the distribution of various types of amino acids within a protein. cnns4qspr makes extraction of different chemical features as easy as typing them in.

Below is a demonstration of the differences between cnns4qspr's voxelization of 'backbone', 'polar', and 'nonpolar' atomic channel selections a user can make when voxelizing a protein. The differences in chemical information are clear.
<p align="center">
<img align="middle" src="cnns4qspr/figs/polar_nonpolar_backbone.gif" width="475" height="375" >
</p>
