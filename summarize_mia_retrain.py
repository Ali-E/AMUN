import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


if __name__ == '__main__':

    MIA_summary_file = sys.argv[1]
    part_1 = MIA_summary_file.split('unl_idx_seed_')[0]
    seed_val = MIA_summary_file.split('unl_idx_seed_')[1].split('/')[0]
    part_2 = MIA_summary_file.split('unl_idx_seed_')[1].split('/')[1]
    part_3 = MIA_summary_file.split('unl_idx_seed_')[1].split('/')[2]

    model_num = part_2.split('_')[-1]
    model_name = part_2.split('_' + model_num)[0]

    column_names = ['forget_acc', 'remain_acc', 'test_acc', 'confidence', 'avg_diff', 'adv_acc']

    seed_list = [1, 10, 100]
    model_num_list = [1, 10, 100]  
    mia_df_list = []
    for seed in seed_list:
        for num in model_num_list:
            part_2 = model_name + '_' + str(num) 
            mia_file = part_1 + 'unl_idx_seed_' + str(seed) + '/' + part_2 + '/' + part_3
            try:
                mia_df = pd.read_csv(mia_file, index_col=0)#.T
                mia_df = mia_df[column_names]
                if mia_df['remain_acc'].mean() < 50:
                    continue
                print(mia_df)
                mia_df_list.append(mia_df)
            except:
                print('File not found:', mia_file)
                continue

    mean_results = pd.concat(mia_df_list, axis=0).mean()
    print(mean_results)
    std_results = pd.concat(mia_df_list, axis=0).std()
    print(std_results)

    results = pd.concat([mean_results, std_results], axis=1)
    results.columns = ['mean', 'std']
    print(results)
    results.to_csv(part_1 + '_mia_summary.csv', index=False)
