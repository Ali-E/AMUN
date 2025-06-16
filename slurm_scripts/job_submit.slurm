#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=betf-delta-gpu
#SBATCH --job-name=clippedn_cifar_clipped
#SBATCH --time=26:00:00      # hh:mm:ss for the job
##SBATCH --constraint="scratch"
### GPU options ###
##SBATCH --gpu-bind=none     # <- or closest
#SBATCH --mail-user=ae20@illinois.edu
#SBATCH --mail-type="BEGIN,END" # See sbatch or srun man pages for more email options 
 
 
# module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set

# module load modtree/gpu  # ... or any appropriate modules
# module load gcc anaconda3_gpu
# conda activate delta

# module load modtree/gpu  # ... or any appropriate modules
module load gcc python
source ~/main/bin/activate

module list  # job documentation and metadata
echo "job is starting on `hostname`"

# srun python main_val_jobsubmit.py --dataset cifar --seed 1
# srun python main_jobsubmit.py --dataset $1 --method $2 --mode $3 --seed $4 --steps $5 --model $6 ##### The one used for all simultaneously
# srun python train/Empirical/train_vanilla.py --method orig --mode wBN --seed 1 ##### The one used for all simultaneously
srun python train_vanilla.py --method $1 --mode $2 --seed $3 --convsn $4 --widen_factor $5 --arch $6 --lr $7 --dataset $8 ##### The one used for all simultaneously

# srun python main_val_jobsubmit_others.py
# srun python main_val_farnia.py
# srun python whitebox_multi.py cifar logs/cifar/final_100/resnet18_clipIn_c1_sedghi_ fgsm
# srun python whitebox_multi.py cifar logs/cifar/clipped/resnet18_clipIn_c1_farnia_val_ fgsm

# srun python main.py
# srun python main.py --dataset mnist
