#!/bin/bash
#SBATCH -J 1.0CPUscale
#SBATCH --account=stf
#SBATCH --partition=stf-gpu
#SBATCH --nodes=1
#SBATCH --mem=120G
#SBATCH --gres=gpu:P100:1
#SBATCH --time=120:00:00

source ~/.login
module load icc_17-impi_2017
module load cuda/10.1.105_418.39
#module load cuda/9.1.85.3
#source /gscratch/pfaendtner/sarah/codes/gromacs_gpu/gromacs-2018.3/bin/bin/GMXRC

#mpiexec -np 1 gmx_mpi mdrun -deffnm nvt_md -ntomp 28 -nb gpu
#mpiexec -np 1 python model_test.py
mpiexec -np 1 python run.py --model SE3ResNet34Small --data-filename cath_3class_ca.npz --training-epochs 100 --batch-size 8 --batchsize-multiplier 1 --kernel-size 3 --initial_lr=0.0001 --lr_decay_base=.996 --p-drop-conv 0.1 --downsample-by-pooling
