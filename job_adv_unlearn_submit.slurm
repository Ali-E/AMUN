#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --mem=48g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4 #-preempt      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --job-name=adv_unlearn
#SBATCH --time=04:16:00      # hh:mm:ss for the job
##SBATCH --constraint="scratch"
### GPU options ###
##SBATCH --gpu-bind=none     # <- or closest
 
 
module load gcc python
source ~/main/bin/activate

module list  # job documentation and metadata
echo "job is starting on `hostname`"

srun python adversarial_unlearn.py --method $1 --mode $2 --seed $3 --convsn $4 --model $5 --lr $6 --dataset $7 --adv_images $8 --adv_delta $9 --unlearn_indices ${10} --source_model_path ${11} --unlearn_method ${12} --mask_path ${13} --LRsteps ${14} --norm_cond ${15} --sample_count ${16} --noise_ratio ${17} --epochs ${18} --remain ${19} --attack ${20} --salun_ratio ${21} --req_mode ${22} ##### The one used for all simultaneously
