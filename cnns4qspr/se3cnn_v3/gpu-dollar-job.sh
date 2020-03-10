#!/bin/bash
#SBATCH -J 1.0CPUscale
#SBATCH --account=stf
#SBATCH --partition=stf-gpu
#SBATCH --nodes=1
#SBATCH --mem=120G
#SBATCH --gres=gpu:P100:1
#SBATCH --time=90:00:00

source ~/.login
module load icc_17-impi_2017
module load cuda/10.1.105_418.39
#module load cuda/9.1.85.3
#source /gscratch/pfaendtner/sarah/codes/gromacs_gpu/gromacs-2018.3/bin/bin/GMXRC

#mpiexec -np 1 gmx_mpi mdrun -deffnm nvt_md -ntomp 28 -nb gpu
#mpiexec -np 1 python model_test.py
mpiexec -np 2 python model_analysis.py --model SE3ResNet34Small --data-filename cath_3class_ca.npz --training-epochs 100 --batch-size 8 --restore-checkpoint-filename trial_8_latest.ckpt
