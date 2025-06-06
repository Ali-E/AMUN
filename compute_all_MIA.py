import os

if __name__ == '__main__':

    count = 25000

    # MIA_method = 'MIA'
    MIA_method = 'RMIA'

    adaptive = False


    epochs = 9
    l1_val = 0.0005


    pgd_attack = True

    norm_cond = 'unnorm'
    # norm_cond = 'norm'

    use_remain = 'False'
    lipschitz = False

    if adaptive:
        req_count = 5
    else:
        req_count = 1

    if lipschitz:
        bn_flag = 'noBN'
        bn_name = 'noBN'
        method_short = 'clip'
        method_complete = 'clip1.0'
    else:
        bn_flag = 'wBN'
        bn_name = ''
        method_short = 'orig'
        method_complete = 'orig'

    if pgd_attack:
        attack_name = ''
        attack = 'pgdl2'
    else:
        attack_name = '_fgsm' 
        attack = 'fgsm'

    if norm_cond == 'norm':
        dataset_name = 'cifar'
    else:
        dataset_name = 'cifar_unnorm'



    # if count == 25000:
    #     lr_list = [1/10**i for i in range(2, 7)] + [0.05, 0.005] # 25000
    # else:
    #     lr_list = [1/10**i for i in range(1, 6)] + [0.05, 0.005]

    lr_list = [1/10**i for i in range(5, 6)] # + [0.05, 0.005] # 25000
    # lr_list = [0.01]
    
    # step_list = [5]#,10]


    unlearn_seed_list = [1,10]

    # methods = ['amun_sa', 'amunra_sa', 'amun', 'amunra', 'amun_l1', 'amunra_l1', 'salun', 'RL', 'FT', 'GA', 'BS', 'l1']
    # methods = ['amunra', 'amun', 'salun', 'RL', 'FT', 'GA', 'BS', 'l1']
    # methods = ['amunra_new', 'amun', 'salun', 'RL', 'FT', 'GA', 'BS', 'l1']
    # methods = ['amun', 'salun', 'RL', 'FT', 'GA', 'BS', 'l1']
    # methods = ['advonly_sa']#, 'advonly_others']
    # methods = ['advonly_others']
    # methods = ['adv', 'adv_RL', 'adv_RS']
    # methods = ['amun_fgsm']
    # methods = ['amun']
    # methods = ['amun_sa', 'amun_rand', 'advonly', 'amun_s2']
    # methods = ['advonly']
    # methods = ['salun']
    methods = ['salun', 'RL', 'BS']

    # methods = ['amun_others']
    # step_list = [1,5,10]
    step_list = [1]
    # lr_list = [0.01]
    # lr_list = [0.02]

    for unlearn_seed in unlearn_seed_list:
        for method in methods:
            if method in ['salun','amun_sa', 'advonly_sa']:
                # salun_ratio_list = ['0.5', '0.7', '0.9']
                salun_ratio_list = ['0.5']
            else:
                salun_ratio_list = ['0.5']
            for mask_val in salun_ratio_list:

                for req_idx in range(req_count):

                    addresses = {'amun_sa': f"/<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/amun/{count}/unl_idx_seed_{unlearn_seed}/vanilla_{method_complete}_{bn_flag}_1/mask_{mask_val}/sc_1_nr_120/use_remain_{use_remain}/ResNet18_{method_short}_{bn_name}_",
                                'amun': f"/<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/amun/{count}/unl_idx_seed_{unlearn_seed}/vanilla_{method_complete}_{bn_flag}_1/sc_1_nr_120/use_remain_{use_remain}/ResNet18_{method_short}_{bn_name}_",
                                'amunra_sa': f"/<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/amunra/{count}/unl_idx_seed_{unlearn_seed}/vanilla_{method_complete}_{bn_flag}_1/mask_{mask_val}/sc_1_nr_120/use_remain_{use_remain}/ResNet18_{method_short}_{bn_name}_",
                                'amun_l1': f"/<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/amun/{count}/unl_idx_seed_{unlearn_seed}/vanilla_{method_complete}_{bn_flag}_1/l1_{l1_val}/sc_1_nr_120/use_remain_{use_remain}/ResNet18_{method_short}_{bn_name}_",
                                'amunra_l1': f"/<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/amun/{count}/unl_idx_seed_{unlearn_seed}/vanilla_{method_complete}_{bn_flag}_1/l1_{l1_val}/sc_1_nr_120/use_remain_{use_remain}/ResNet18_{method_short}_{bn_name}_",
                                'amun_s2': f"/<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/amun/{count}/unl_idx_seed_{unlearn_seed}/vanilla_{method_complete}_{bn_flag}_1/sc_2_nr_120/use_remain_{use_remain}/ResNet18_{method_short}_{bn_name}_",
                                'advonly': f"/<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/advonly/{count}/unl_idx_seed_{unlearn_seed}/vanilla_{method_complete}_{bn_flag}_1/use_remain_{use_remain}/ResNet18_{method_short}_{bn_name}_",
                                'advonly_sa': f"/<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/advonly/{count}/unl_idx_seed_{unlearn_seed}/vanilla_{method_complete}_{bn_flag}_1/mask_{mask_val}/use_remain_{use_remain}/ResNet18_{method_short}_{bn_name}_",
                                'advonly_others': f"/<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/advonly_others/{count}/unl_idx_seed_{unlearn_seed}/vanilla_{method_complete}_{bn_flag}_1/use_remain_{use_remain}/ResNet18_{method_short}_{bn_name}_",
                                'advonly_fgsm': f"/<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/advonly_fgsm/{count}/unl_idx_seed_{unlearn_seed}/vanilla_{method_complete}_{bn_flag}_1/use_remain_{use_remain}/ResNet18_{method_short}_{bn_name}_",
                                'amun_others': f"/<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/amun_others/{count}/unl_idx_seed_{unlearn_seed}/vanilla_{method_complete}_{bn_flag}_1/sc_1_nr_1/use_remain_{use_remain}/ResNet18_{method_short}_{bn_name}_",
                                'adv': f"/<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/adv/{count}/unl_idx_seed_{unlearn_seed}/vanilla_{method_complete}_{bn_flag}_1/use_remain_{use_remain}/ResNet18_{method_short}_{bn_name}_",
                                'adv_RL': f"/<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/adv_RL/{count}/unl_idx_seed_{unlearn_seed}/vanilla_{method_complete}_{bn_flag}_1/use_remain_{use_remain}/ResNet18_{method_short}_{bn_name}_",
                                'adv_RS': f"/<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/adv_RS/{count}/unl_idx_seed_{unlearn_seed}/vanilla_{method_complete}_{bn_flag}_1/use_remain_{use_remain}/ResNet18_{method_short}_{bn_name}_",
                                'amun_rand': f"/<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/amun_rand/{count}/unl_idx_seed_{unlearn_seed}/vanilla_{method_complete}_{bn_flag}_1/sc_1_nr_120/use_remain_{use_remain}/ResNet18_{method_short}_{bn_name}_",
                                'amun_fgsm': f"/<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/amun_fgsm/{count}/unl_idx_seed_{unlearn_seed}/vanilla_{method_complete}_{bn_flag}_1/sc_1_nr_120/use_remain_{use_remain}/ResNet18_{method_short}_{bn_name}_",
                                'amunra': f"/<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/amunra/{count}/unl_idx_seed_{unlearn_seed}/vanilla_{method_complete}_{bn_flag}_1/sc_1_nr_120/use_remain_{use_remain}/ResNet18_{method_short}_{bn_name}_",
                                'amunra_new': f"/<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/amunra_new/{count}/unl_idx_seed_{unlearn_seed}/vanilla_{method_complete}_{bn_flag}_1/sc_1_nr_120/use_remain_{use_remain}/ResNet18_{method_short}_{bn_name}_",
                                'salun': f"/<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/RL/{count}/unl_idx_seed_{unlearn_seed}/vanilla_{method_complete}_{bn_flag}_1/mask_{mask_val}/use_remain_{use_remain}/ResNet18_{method_short}_{bn_name}_",
                                'RL': f"/<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/RL/{count}/unl_idx_seed_{unlearn_seed}/vanilla_{method_complete}_{bn_flag}_1/use_remain_{use_remain}/ResNet18_{method_short}_{bn_name}_",
                                'GA': f"/<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/GA/{count}/unl_idx_seed_{unlearn_seed}/vanilla_{method_complete}_{bn_flag}_1/use_remain_{use_remain}/ResNet18_{method_short}_{bn_name}_",
                                'BS': f"/<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/BS/{count}/unl_idx_seed_{unlearn_seed}/vanilla_{method_complete}_{bn_flag}_1/use_remain_{use_remain}/ResNet18_{method_short}_{bn_name}_",
                                'FT': f"/<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/FT/{count}/unl_idx_seed_{unlearn_seed}/vanilla_{method_complete}_{bn_flag}_1/use_remain_{use_remain}/ResNet18_{method_short}_{bn_name}_",
                                'l1': f"/<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/l1/{count}/unl_idx_seed_{unlearn_seed}/vanilla_{method_complete}_{bn_flag}_1/use_remain_{use_remain}/ResNet18_{method_short}_{bn_name}_"}


                    for lr in lr_list:
                        for step in step_list:
                            command = f"python batch_job_submit.py --job_type {MIA_method}  --method {method_short} --mode {bn_flag} --seed 1  --dataset cifar --adv_images /<path to directory>/amun/logs/correct/scratch/{dataset_name}/vanilla_{method_complete}_{bn_flag}_1/adv_data/seed_1/adv_tensor{attack_name}.pt --adv_delta /<path to directory>/amun/logs/correct/scratch/{dataset_name}/vanilla_{method_complete}_{bn_flag}_1/adv_data/seed_1/smallest_eps{attack_name}.csv --unlearn_indices /<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/unlearn_idx/{count}/seed_{unlearn_seed}.csv --model_path {addresses[method]} --epoch {epochs}  --model_count 3 --LRsteps {step} --lr {lr} --trials 1  --norm_cond {norm_cond} "

                            if method in ['amun_sa', 'amunra_sa', 'salun', 'advonly_sa']:
                                command += f" --mask_path /<path to directory>/amun/logs/correct/scratch/{dataset_name}/unlearn/genmask/{count}/unl_idx_seed_{unlearn_seed}/vanilla_{method_complete}_{bn_flag}_1/salun_mask/with_{mask_val}.pt"

                            if adaptive:
                                command += ' --adaptive --req_index ' + str(req_idx)

                            print(command)
                            os.system(command)


        print(lr_list)
