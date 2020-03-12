[![Build Status](https://travis-ci.org/CNNs4QSPR/cnns4qspr.svg?branch=master)](https://travis-ci.org/CNNs4QSPR/cnns4qspr)

# cnns4qspr
A package for creating rotationally and translationally equivariant structural and chemical features for molecules. 

>>cnns4qspr>loader.py: This module contains functions for loading a pdb file and calculating
the atomic density fields for different atom types. The fields can then be
used for plotting, or to send into the convolutional neural network.

Functions: load_pdb(),shift_coords(),grid_positions(),make_fields(), and voxelize()

>>cnns4qspr>visualizer.py: This module contains functions to plot atomic density fields before they
go into a model, as well as what the density fields have been transformed into
at certain points within the model.

Functions: plot_field(),plot internal.
