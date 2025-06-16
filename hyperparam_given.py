import os
import argparse
import pandas as pd
import sys

if __name__ == '__main__':

    filename = sys.argv[1]
    df = pd.read_csv(filename)
    print(df)

    # count = int(filename.split('_')[-1].split('.')[0])
    count = int(filename.split('_')[-2].split('.')[0])
    # count = 5000
    print('count: ', count)


    df = df[['method', 'step', 'lr', 'mask_val']]
    # df = df[['method', 'step', 'lr', 'mask_val', 'unl_seed']]
    df = df.drop_duplicates()
    print(df)

    norm_cond = 'unnorm'

    # dataset = 'cifar'
    dataset = 'tinynet'

    # model = 'ResNet18'
    model = 'VGG'

    adaptive = False

    use_remain = 'False'
    pgd_attack = True

    if pgd_attack:
        # attack_name = ''
        attack_name = '_pgd10'
        attack = 'pgdl2'
    else:
        attack_name = '_fgsm' 
        attack = 'fgsm'

    if use_remain == 'True':
        remain = 'use'
    else:
        remain = 'no_use'

    # if norm_cond == 'norm':
    dataset_name = dataset
    if norm_cond != 'norm':
        if dataset == 'tinynet':
            dataset_name = f'{dataset}_{model}'
        dataset_name += '_unnorm'


    unlearn_seed_list = [10, 100]
    # unlearn_seed_list = [1] #10, 100]

    sample_count_list = [1]
    noise_ratio_list = [120]

    # methods_list = ['amun', 'amun_sa', 'advonly', 'advonly_sa', 'salun', 'RL', 'FT', 'GA', 'BS', 'l1']
    # methods_list = ['advonly_sa', 'advonly', 'salun', 'RL', 'FT', 'GA', 'BS', 'l1']
    methods_list = ['advonly', 'amun', 'advonly_sa']

    for idx, row in df.iterrows():
        print(row)
        for unlearn_seed in unlearn_seed_list:
        
            # unlearn_seed = row['unl_seed']
            method = row['method']
            if method not in methods_list:
                continue

            if not pgd_attack:
                method = method.split('_')[0]

            step = row['step']
            lr = row['lr']

            # lr_list = [1/10**i for i in range(2, 5)] #+ [0.05, 0.005] # 25000
            # if lr not in lr_list:
            #     continue
            mask_val = row['mask_val']

            print(method, step, lr, mask_val)

            for sample_count in sample_count_list:
                for noise_ratio in noise_ratio_list:
                    command = f"python batch_job_submit.py --job_type advunlearn --attack {attack} --model {model} --method orig --mode wBN --seed 1 --dataset {dataset} --adv_images ~/amun_exps/logs/correct/scratch/{dataset_name}/vanilla_orig_wBN_1/adv_data/seed_1/adv_tensor{attack_name}.pt --adv_delta ~/amun_exps/logs/correct/scratch/{dataset_name}/vanilla_orig_wBN_1/adv_data/seed_1/smallest_eps{attack_name}.csv --unlearn_indices ~/amun_exps/logs/correct/scratch/{dataset_name}/unlearn/unlearn_idx/{count}/seed_{unlearn_seed}.csv --model_path ~/amun_exps/logs/correct/scratch/{dataset_name}/vanilla_orig_wBN_  --unlearn_method {method} --model_count 3 --LRsteps {step} --lr {lr} --sample_count {sample_count} --noise_ratio {noise_ratio}  --norm_cond {norm_cond} --remain {remain} --salun_ratio {mask_val}"

                    if adaptive:
                        command += ' --adaptive'

                    print(command)
                    os.system(command)


    print('count: ', count)
    print('use_remain: ', use_remain)