import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


if __name__ == '__main__':

    MIA_summary_file = sys.argv[1]

    part_1 = MIA_summary_file.split('unl_idx_seed_')[0]
    part_2 = MIA_summary_file.split('unl_idx_seed_')[1]

    column_names = ['forget_acc', 'remain_acc', 'test_acc', 'confidence', 'avg_diff', 'adv_acc']

    use_new_reference = False
    if use_new_reference: # for noBN:
        retain_res = {'forget_acc': 89.74, 'remain_acc': 96.59, 'test_acc': 88.89, 'confidence': 14.64} ## 5000 clip_noBN

    seed_list = [1, 10, 100]
    mia_df_list = []
    for model_seed in seed_list:
        for seed in seed_list:
            source_model_name = part_2.split('/')[1]
            part_2_parts = part_2.split(source_model_name)
            mia_file = part_1 + 'unl_idx_seed_' + str(seed) + '/' + source_model_name[:-1] + str(model_seed) + part_2_parts[1]
            # print(mia_file)
            try:
                mia_df = pd.read_csv(mia_file).T
                mia_df.columns = column_names
                mia_df = mia_df[1:]
                if use_new_reference:
                    mia_df['avg_diff'] = np.abs(mia_df['test_acc'] - retain_res['test_acc']) + np.abs(mia_df['forget_acc'] - retain_res['forget_acc']) + np.abs(mia_df['remain_acc'] - retain_res['remain_acc']) + np.abs(mia_df['confidence'] - retain_res['confidence'])
                    mia_df['avg_diff'] = mia_df['avg_diff'] / 4
                # print(mia_df)
                mia_df_list.append(mia_df)
            except:
                # print('File not found!')
                continue


    mean_results = pd.concat(mia_df_list, axis=0).mean()
    # print(mean_results)
    std_results = pd.concat(mia_df_list, axis=0).std()
    # print(std_results)

    results = pd.concat([mean_results, std_results], axis=1)
    results.columns = ['mean', 'std']
    print(results)
    # results.to_csv(part_1 + '_mia_summary.csv', index=False)
