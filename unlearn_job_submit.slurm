#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --mem=8g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bblr-delta-gpu
#SBATCH --job-name=clippedn_all_cifar_clipped
#SBATCH --time=2:30:00      # hh:mm:ss for the job
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
# module load gcc python
# source /taiga/illinois/engineering/cs/hs1/ali/delta/bin/activate

module load gcc python
source ~/main/bin/activate

module list  # job documentation and metadata
echo "job is starting on `hostname`"


# srun python compute_trained_SVs.py mnist fast logs/mnist/resnet_div2_bncs50_Adam_StepLR_WD/resnet18_ 
# srun python compute_trained_SVs.py cifar all logs/cifar/ResNet18_div2_bncs50_SGD_StepLR_WD_lr0.1/resnet18_

# srun python compute_trained_SVs.py cifar all logs/cifar/DLA_bncs50_SGD_StepLR_WD/dla_ --model DLA
# srun python compute_trained_SVs.py mnist all logs/mnist/DLA_bncs50_SGD_StepLR_WD/dla_ --model DLA

# srun python time_benchmark_qr_vs_power.py

# srun python batch_job_submit.py --method fastclip --model ResNet18 --seed 1 --mode wBN
# srun python batch_job_submit.py --method orig --model ResNet18 --seed -1 --mode wBN

# srun python main.py --method orig --model ResNet18 --seed -1 --mode wBN --save_checkpoints False --filter random --path logs/cifar/ResNet18_models/method_lr0.5_orig__1/
srun python main.py --dataset $1 --method $2 --model $3 --mode $4 --seed $5 --filter $6 --unlearn $7 --up_sample $8

# python featurize_and_score.py --model_id 1 --indir logs/cifar/ResNet18_models/method_lr0.5_orig__1/
# python gather.py

# python viz_scores.py $1

# srun python compute_trained_SVs.py mnist miyato logs/mnist/resnet_div2_bncs50_cosine_noWD/resnet18_
# srun python compute_trained_SVs.py mnist orig logs/mnist/clipped/simpleConv_mine_k3_pad1_zeros_st1_di1_test_cosine --direct True
