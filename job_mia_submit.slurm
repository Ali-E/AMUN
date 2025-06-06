#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --mem=48g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4-preempt      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bcwm-delta-gpu
#SBATCH --job-name=SVC_MIA
#SBATCH --time=00:16:00      # hh:mm:ss for the job
##SBATCH --constraint="scratch"
### GPU options ###
##SBATCH --gpu-bind=none     # <- or closest

module load gcc python
source ~/main/bin/activate

module list  # job documentation and metadata
echo "job is starting on `hostname`"

srun python mia_evaluate.py --method $1 --mode $2 --seed $3 --model $4 --dataset $5 --adv_images $6 --adv_delta $7 --unlearn_indices $8 --source_model_path $9 --epoch ${10} --mask_path ${11} --lr ${12} --LRsteps ${13} --trials ${14} --norm_cond ${15}  ##### The one used for all simultaneously
