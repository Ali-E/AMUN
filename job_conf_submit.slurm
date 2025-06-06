#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --mem=48g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bcwm-delta-gpu
#SBATCH --job-name=clippedn_all_cifar_clipped
#SBATCH --time=2:00:00      # hh:mm:ss for the job
##SBATCH --constraint="scratch"
### GPU options ###
##SBATCH --gpu-bind=none     # <- or closest
 
 
module load gcc python
source ~/main/bin/activate

module list  # job documentation and metadata
echo "job is starting on `hostname`"

srun python evaluate_conf.py --method $1 --mode $2 --convsn $3 --widen_factor $4 --arch $5 --lr $6 --dataset $7 --model_path $8 --epoch ${9} --unlearn_indices ${10} ##### The one used for all simultaneously

