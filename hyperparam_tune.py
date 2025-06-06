import os
import argparse

if __name__ == '__main__':

    count = 5000

    pgd_attack = True
    adaptive = True

    norm_cond = 'unnorm'
    # norm_cond = 'norm'

    use_remain = 'False'
    lipschitz = False

    if lipschitz:
        bn_flag = 'noBN'
        method_short = 'clip'
        method_complete = 'clip1.0'
    else:
        bn_flag = 'wBN'
        method_short = 'orig'
        method_complete = 'orig'

    if pgd_attack:
        attack_name = ''
        attack = 'pgdl2'
    else:
        attack_name = '_fgsm' 
        attack = 'fgsm'

    if use_remain == 'True':
        remain = 'use'
    else:
        remain = 'no_use'

    if norm_cond == 'norm':
        dataset_name = 'cifar'
    else:
        dataset_name = 'cifar_unnorm'


    lr_list = [1/10**i for i in range(1, 6)] + [0.05 * 1/10**i for i in range(0, 4) ] 
    print(lr_list)
    unlearn_seed_list = [1, 10, 100]
    step_list = [1,5,10]

    methods = ['amun_sa', 'advonly_sa' 'amun', 'salun', 'RL', 'FT', 'GA', 'BS', 'l1']
    salun_ratio = ['0.1', '0.3', '0.5','0.7','0.9']

    for unlearn_seed in unlearn_seed_list:
        for lr in lr_list:
            for step in step_list:
                for method in methods:
                    sample_count_list = [1]
                    noise_ratio_list = [120]

                    if method in ['salun', 'amun_sa', 'advonly_sa']:
                        salun_ratio_list = salun_ratio
                    else:
                        salun_ratio_list = ['0.5']

                    for sample_count in sample_count_list:
                        for noise_ratio in noise_ratio_list:
                            for ratio in salun_ratio_list:
                                command = f"python batch_job_submit.py --job_type advunlearn --attack {attack} --method {method_short} --mode {bn_flag} --seed 1 --dataset cifar --adv_images ~/amun_exps/logs/correct/scratch/{dataset_name}/vanilla_{method_complete}_{bn_flag}_1/adv_data/seed_1/adv_tensor{attack_name}.pt --adv_delta ~/amun_exps/logs/correct/scratch/{dataset_name}/vanilla_{method_complete}_{bn_flag}_1/adv_data/seed_1/smallest_eps{attack_name}.csv --unlearn_indices ~/amun_exps/logs/correct/scratch/{dataset_name}/unlearn/unlearn_idx/{count}/seed_{unlearn_seed}.csv --model_path ~/amun_exps/logs/correct/scratch/{dataset_name}/vanilla_{method_complete}_{bn_flag}_  --unlearn_method {method} --model_count 3 --LRsteps {step} --lr {lr} --sample_count {sample_count} --noise_ratio {noise_ratio}  --norm_cond {norm_cond} --remain {remain} --salun_ratio {ratio}"

                                if adaptive:
                                    command += ' --adaptive'

                                print(command)
                                os.system(command)


