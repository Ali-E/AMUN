# Using slurm for parallel training, hyper-parameter tuning, and evaluations:

In this section we focus on how to run the experiments in a parallel setting when we have access to several nodes controlled by slurm.

## Training the original models and running the unlearning methods

1. We first train the original models on a dataset (```cifar``` for CIFAR-10 or ```tinynet``` for Tiny Imagenet):

The ```seed``` argument can be chosen arbitrarily, but if set to ```-1``` it automatically runs it for the three seeds of ```1,10,100``` simultaneously (which is the settup used in our experiments).

`python batch_job_submit.py --job_type train --dataset cifar --model ResNet18 --seed -1`

2. The next step is to use AMUN is to comptue the adversarial examples corresponding to the forget samples. Here we generate the adversarial examples for all the training samples because we used different subsets as the forgetsets for evaluations in our experiments. The following command generates the adversarial samples and a csv file for their corresponding $\epsilon$ and $\delta$ values along with original predictions and the adversarial labels:

`python batch_job_submit.py --job_type advdata --method orig --seed 1 --dataset cifar --model_path <path to folder containing the checkpoints of the trained model> --epoch <epoch to read from for the trained model, e.g., 200>`

The command above generates the adversarial examples for the model trained previously with ```seed=1```. To generate adversarial examples for other models, the command has to be updated accordingly.

3. Next, we generate three random subsets of indices to be used as the forgetset in our experiments. 

`python forget_index_generate.py <dataset name> <forget set size>`

For example for the problem of forgeting $10\%$ of the training set in cifar10, we use the following command:

`python forget_index_generate.py cifar 5000`

4. As the gold-standard of unlearning, we use the models retrained on the remaining data ($D_\mathrm{R}$). The following command trains the retrained models. It traines ```3``` retrained models for each of the a given set of unlearning indices. In our experiments we repeat this both for each of the ```3``` unlearning indices. In our experiments this consititute ```10%``` and ```50%``` of the training samples of CIFAR-10 and ```10%``` of the samples in the Tiny Imagenet dataset.

`python batch_job_submit.py --job_type advunlearn --seed -1 --dataset cifar --adv_images <address to adversarial examples> --adv_delta <address to the file containing delta values for adv examples> --unlearn_indices <address to indices corresponding to the forget set> --model_path <path to the trained models>  --unlearn_method retrain`

For example for ```seed=1``` of the forget set indices, assuming the default paths are used in the previous steps, the following command will retrain ```3``` models on the remaining data ($D_\mathrm{R} = D - D_\mathrm{F}$), where $D_\mathrm{F}$ is specified by the indices in file ```unlearn_indices/cifar/5000/seed_1.csv```:

`python batch_job_submit.py --job_type advunlearn --seed -1 --dataset cifar --adv_images logs/scratch/cifar_ResNet18_unnorm/vanilla_orig_wBN_1/adv_data/seed_1/adv_tensor.pt --adv_delta logs/scratch/cifar_ResNet18_unnorm/vanilla_orig_wBN_1/adv_data/seed_1/smallest_eps.csv --unlearn_indices unlearn_indices/cifar/5000/seed_1.csv --model_path logs/scratch/cifar_ResNet18_unnorm/vanilla_orig_wBN_  --unlearn_method retrain`

5. We use ```hyperparam_tune.py``` to run the unlearning methods (or any subset of them) for various hyper-parameters on various forgetsets for all the ```3``` original models. You can modify the file as needed to change the default settings. For example for unlearning using ```AMUN``` for unlearning ```5000``` samples from ```ResNet18``` models trained on ```CIFAR-10```, we use the following command: 

`python hyperparam_tune.py --method amun --dataset cifar --model ResNet18 --count 5000`

Implemented unlearning methods are:
```
amun: AMUN
advonly: AMUN without using $R_\mathrm{F}$ (i.e., only the adversarial counterparts).
FT: Fine-tuning
RL: Random labling
BS: Boundary shrink
salun: SalUn (Random labling + masking)
l1: $l1$-sparse (Fine tuning + sparsification)
GA: Gradient ascent
amun-sa: AMUN + SalUn's masking
advonly-sa: AMUN (advonly setting) + SalUn's masking
```

check the arguments for other settings.

### SalUn:

To use the SalUn unlearning method, or any other method along with SalUn, we first need to compute the mask for each original model. The following command computes masks with ratios from ```[0.1,...,0.9]``` for all the three default original models for one of the set of unlearning indices (```seed=10```):

`python batch_job_submit.py --job_type advunlearn --seed 1 --dataset cifar --unlearn_indices unlearn_indices/cifar/5000/seed_10.csv --model_path logs/scratch/cifar_ResNet18/vanilla_orig_wBN_ --unlearn_method genmask --model_count 3`

we should repeat this for all unlearning indices.


## Evaluation using SVC MIA used by prior work:

1. To evaluate using more basic MIA, we first need to compute the metrics for the retrain models that are used as the point of comparison. For this purpose you can use the following command:

`python batch_job_submit.py --job_type MIA --method orig --mode wBN --seed 1  --dataset cifar --adv_images <address to adversarial images> --adv_delta <address to adv delta file> --unlearn_indices <address to unlearn indices> --model_path <address to the trained models> --epoch 199  --model_count 3 --LRsteps 40 --lr 0.1 --trials 1`

2. We can collect the results from the retrained models' evaluation for all the retrained models and unlearning indices using the following command: 

`python summarize_mia_retrain.py <address to one of the result files>`

For example, if the previous commands have been followed as is, you can use the following command:

`python summarize_mia_retrain.py logs/scratch/cifar_ResNet18/unlearn/retrain/5000/unl_idx_seed_1/ResNet18_orig__1/200_mia_SVC_results.csv`

3. For computing MIA scores for each unlearned model we can use the following command for each of the unlearning indices:

`python batch_job_submit.py --job_type MIA --seed 1  --dataset cifar --model ResNet18 --adv_images logs/scratch/cifar_ResNet18/vanilla_orig_wBN_1/adv_data/seed_1/adv_tensor.pt --adv_delta logs/scratch/cifar_ResNet18/vanilla_orig_wBN_1/adv_data/seed_1/smallest_eps.csv --unlearn_indices unlearn_indices/cifar/5000/seed_1.csv --model_path logs/scratch/cifar_ResNet18/unlearn/amun/5000/unl_idx_seed_1/vanilla_orig_wBN_1/use_remain_True/ResNet18_orig__ --epoch 9  --model_count 3 --LRsteps 1 --lr 0.01 --trials 1`

This computes the MIA scores for each of the three unlearned models derived from the three original models based on unlearning indices ```seed=1```.


## Evaluation using stronger MIA (RMIA):

1. For our evaluations of the unlearning mehtods, we use RMIA which is the SOTA membership inference attack at the time of sumbitting our work. We use the online version of RMIA for enhanced evaluations. The online version, requires training of some reference models which are trained on half of the available data (training and test) such that each sample will be used as part of the training samples in half of the reference models. To train these reference models, you can use the following command:

`python batch_job_submit.py --job_type advunlearn --model <model_name> --seed -<num reference models> --dataset <dataset> --unlearn_method reference`

For example, this command can be used to train 128 ResNet-18 reference models on cifar-10:

`python batch_job_submit.py --job_type advunlearn --model ResNet18 --seed -128 --dataset cifar --unlearn_method reference`

2. compute the reference_mat for RMIA evaluations:

`python batch_job_submit.py --job_type RMIA_ref --dataset <dataset> --model_path <prefix address to the trained models> --epoch <epoch number of checkpoint to use> --trials -<num reference models>`

for example:

`python batch_job_submit.py --job_type RMIA_ref --dataset cifar --model_path logs/scratch/cifar_ResNet18/unlearn/reference/ResNet18_orig__ --epoch 160 --trials -128`

(update the reference_mat in batch_job_submit.py accordingly)


3. You can summarize the results for one of the methods applied to all the original models and all the unlearning indices using the following command:

`python summarize_rmia.py <address to one of the result files>`

for example, if the previous commands are followed, the following command summarized the results for ```AMUN```:

`python summarize_rmia.py logs/scratch/cifar_ResNet18/unlearn/amun/5000/unl_idx_seed_1/vanilla_orig_wBN_10/use_remain_True/ResNet18_orig__keep_m128_d60000_s0/m128_d60000_s0_160_prob_matrix_logits_onehot/LRs_1_lr_0.01_9_acc_rmia_taylor-soft-margin_excFalse.csv`