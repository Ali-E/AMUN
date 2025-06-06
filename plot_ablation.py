import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import seaborn as sb
from matplotlib import rc




if __name__ == '__main__':

    step_val = sys.argv[1]
    count = sys.argv[2]

    sb.set_context(
        "talk",
        font_scale=1,
    )
    sb.set_style('white')
    sb.set_style('ticks')

    linestyles = ['solid', 'dashed', 'dashdot', 'dashdot', 'dotted', 'dotted','dotted']

    fig, ax = sb.mpl.pyplot.subplots(1, 3, figsize=(14.5, 4))
    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    colors = sb.mpl_palette('tab10', n_colors=9)

    div = 2
    x = list(range(21))
    x = [i for i in x if i % div == 0]


    # unlearn_methods = ['adv', 'advonly']
    unlearn_tuples = [('adv', 'True'), ('adv', 'False'), ('advonly', 'False')]

    for idx, unlearn_method_tuple in enumerate(unlearn_tuples):
        unlearn_method, use_remain = unlearn_method_tuple
        filename = f'/<path to directory>/amun/logs/correct/scratch/cifar_unnorm/unlearn/{unlearn_method}/{count}/unl_idx_seed_1/vanilla_orig_wBN_1/use_remain_{use_remain}/ResNet18_orig__1/LRs_{step_val}_lr_0.01/{step_val}_loss_acc_results.csv'
        df_main = pd.read_csv(filename, index_col=0)
        print(df_main.head())

        methods = [unlearn_method] 
        other_methods = ['forgetset', 'RS', 'RL', 'forgetset_RL', 'forgetset_AdvL']
        methods += [unlearn_method + '_' + method for method in other_methods]

        labels = ['Adv', 'Orig', 'Adv-RS', 'Adv-RL', 'Orig-RL', 'Orig-AdvL']

        filenames = {method: filename.replace(unlearn_method, method) for method in methods}
        df_dict = {}

        for method in methods:
            df = pd.read_csv(filenames[method])#, index_col=0)
            df = df[df.iloc[:, 0] % div == 0]
            df_dict[method] = df

        counter = 0
        for key, df in df_dict.items():
            ax[idx].plot(
            x,
            df['ts_acc'],
            label=labels[counter],
            color=colors[counter],
            linestyle=linestyles[counter])
            counter += 1

        z = ax[idx]
        if idx == 0:
            z.set_ylabel(r"Test Accuracy", fontsize=14)
            # z.legend_.remove()
            # z.get_legend().remove()
        z.set_xlabel(r"Epochs", fontsize=14)

        z.tick_params(direction='in', length=6, width=2, colors='k', which='major')
        z.tick_params(direction='in', length=4, width=1, colors='gray', which='minor')
        z.set_xticks([5, 10, 15, 20], ['5', '10', '15', '20'], minor=True)
        # z.set_yticks([20*i for i in range(6)], [str(20*i) for i in range(6)], minor=True)
        z.set_aspect(aspect='auto', adjustable='datalim')
        z.minorticks_on()


    z.legend(loc='center left', fontsize=14, bbox_to_anchor=(1, 0.5))
    sb.despine(left=True, offset=10, trim=False)
    sb.mpl.pyplot.tight_layout()

    plt.savefig(f'ablation_plots/fancy_ablation_acc_remAll_ep{step_val}_{count}.png')


