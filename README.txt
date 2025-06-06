1. we first train 3 models:

python batch_job_submit.py --job_type train --dataset cifar --method clip --model ResNet18 --mode wBN --seed -1

2. comptue the adversarial examples for each original model. This generates the adversarial samples and a csv file for their delta values along with original and adversarial labels:

python batch_job_submit.py --job_type advdata --method clip --mode wBN --seed 1 --dataset cifar --model_path <path to the trained models> --epoch <epoch to read from for the trained model>

3. train the retrained models. 3 retrained models for each of the 3 unl_indices. Do this both for 5000 and 25000

python batch_job_submit.py --job_type advunlearn --method clip  --mode wBN --seed -1 --dataset cifar --adv_images <address to adversarial examples> --adv_delta <address to the file containing delta values for adv examples> --unlearn_indices <address to indices corresponding to the forget set> --model_path <path to the trained models>  --unlearn_method retrain

4. compute the reference models for RMIA:

python batch_job_submit.py --job_type advunlearn --method clip --mode noBN --seed -128 --dataset cifar --adv_images <address to adversarial images> --adv_delta <> --unlearn_indices <address to unlearn indices> --model_path <address to the originally trained models>  --unlearn_method reference

5. compute the reference_mat for RMIA evaluations:

python batch_job_submit.py --job_type RMIA_ref --method clip --mode noBN --seed 1  --dataset cifar --adv_images <address to adv images> --adv_delta <address to adv delta values> --unlearn_indices <address to unlearn indices> --model_path <address to the trained models> --epoch 160 --model_count 1 --trials -128



Evaluation:

(update the reference_mat in batch_job_submit.py accordingly)

1. Compute the MIA and RMIA for retrain models to use as a reference point. Also use different seed values 

python batch_job_submit.py --job_type MIA --method orig --mode wBN --seed 1  --dataset cifar --adv_images <address to adversarial images> --adv_delta <address to adv delta file> --unlearn_indices <address to unlearn indices> --model_path <address to the trained models> --epoch 199  --model_count 3 --LRsteps 40 --lr 0.1 --trials 1


2. Use hyperparam_tune.py to run all the unlearning methods for various hyper-parameters

python hyperparam_tune.py

3. Use compute_all_MIA.py to evaluate all the unlearned models for all the hyper parameters:

python compute_all_MIA.py

4. summarize the results for all the MIS scores:

