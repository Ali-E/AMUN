import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


if __name__ == '__main__':

    MIA_summary_file = sys.argv[1]

    if 'unl_idx_seed_' not in MIA_summary_file:
        print("Error: The provided file does not contain 'unl_idx_seed_' in its name.")
        sys.exit(1)
    
    seed_val = MIA_summary_file.split('unl_idx_seed_')[1].split('/')[0]
    part_1 = MIA_summary_file.split(f'unl_idx_seed_{seed_val}')[0]
    part_2 = MIA_summary_file.split(f'unl_idx_seed_{seed_val}')[1]

    column_names = ['forget_acc', 'remain_acc', 'test_acc', 'auc_ft', 'auc_fr', 'auc_rt']

    seed_list = [1, 10, 100]
    mia_df_list = []
    for model_seed in seed_list:
        for seed in seed_list:
            rmia_file = part_1 + 'unl_idx_seed_' + str(seed) + part_2
            try:
                mia_df = pd.read_csv(rmia_file, index_col=0)
                mia_df = mia_df[column_names]
                print(mia_df)
                mia_df_list.append(mia_df)
            except Exception as e:
                print(f'File not found: {rmia_file}, Error: {e}')
                continue

    mean_results = pd.concat(mia_df_list, axis=0).mean() *100.
    std_results = pd.concat(mia_df_list, axis=0).std() *100.

    results = pd.concat([mean_results, std_results], axis=1)
    results.columns = ['mean', 'std']
    print(results)
