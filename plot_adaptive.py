import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import seaborn as sb
from matplotlib import rc




if __name__ == '__main__':

    file_name_remTrue = 'RMIA_results/cifar_unnorm_adaptive_FT_only_all_gap_list_remTrue_ep9_5000.csv'
    file_name_remFalse = 'RMIA_results/cifar_unnorm_adaptive_FT_only_all_gap_list_remFalse_ep9_5000.csv'


    sb.set_context(
        "talk",
        font_scale=1,
    )
    sb.set_style('white')
    sb.set_style('ticks')

    linestyles = ['solid', 'dashed', 'dashdot', 'dotted', 'dashdot', 'dotted','dashdot', 'dotted', 'dotted']

    fig, ax = sb.mpl.pyplot.subplots(1, 1, figsize=(16.5, 4))
    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    colors = sb.mpl_palette('tab10', n_colors=9)

    x = list(range(5))


    filenames =  [file_name_remTrue, file_name_remFalse]

    labels = {'adv': 'Amun-A', 'amun': 'Amun', 'salun': 'SalUn', 'RL': 'RL', 'FT': 'FT', 'GA': 'GA', 'BS': 'BS', 'l1': 'L1'}

    keys = ['auc_ft', 'auc_fr', 'auc_ft', 'auc_fr']

    for idx, filename in enumerate(filenames[:1]):
        df_main = pd.read_csv(filename)
        print(df_main.head())

        methods = df_main['method'].unique()

        df_dict = {}
        for method in methods:
            if method not in labels.keys():
                continue
            df = df_main[df_main['method'] == method]
            df_dict[method] = df

        counter = 0
        for key, df in df_dict.items():
            ax[idx].plot(
            df['req_index'].values + 1,
            # df['auc_fr'] - df['auc_ft'],
            df['forget_acc'],
            label=labels[key],
            color=colors[counter],
            linestyle=linestyles[counter])
            counter += 1


        z = ax[idx]
        if idx == 0:
            z.set_ylabel(r"AUC Gap", fontsize=14)
            # z.legend_.remove()
            # z.get_legend().remove()
        z.set_xlabel(r"Request counts", fontsize=14)

        z.tick_params(direction='in', length=6, width=2, colors='k', which='major')
        z.tick_params(direction='in', length=4, width=1, colors='gray', which='minor')
        z.set_xticks([1, 2, 3, 4, 5], ['1', '2', '3', '4', '5'], minor=False)
        z.set_aspect(aspect='auto', adjustable='datalim')
        # z.minorticks_on()


    z.legend(loc=0, fontsize=11)
    sb.despine(left=True, offset=10, trim=False)
    sb.mpl.pyplot.tight_layout()
    plt.savefig(f'ablation_plots/fancy_forget_adaptive.png')


