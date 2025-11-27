# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Analysis functions for S-GAIN:

Todo properly name the graphs, legends, labels, etc

Helper functions:
(1) get_Gsm_Dsm: helper function to get the unique Generator and Discriminator settings (initialization and sparsity)
(2) prepare_plot: helper function to prepare plots
(3) prepare_text: helper function to prepare the text for printing
(4) get_experiments_from_info: helper function to group the experiments by dataset, miss_rate, miss_modality and seed
(5) prepare_subplot_params: helper function to prepare the subplot parameters
(6) prepare_data_params: helper function to prepare the data parameters
(7) plot_legend: helper function to plot the legend

Compile metrics and plot graphs:
(8) extract_log_info: extract information from the experiment logs
(9) compile_metrics: compile the metrics of the provided experiments
(10) plot_rmse: plot the RMSE of the provided experiments
(11) plot_success_rate: plot the success rate of the provided experiments
(12) plot_imputation_time: plot the imputation time of the provided experiments
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from datetime import timedelta
from matplotlib import ticker, container
from os import mkdir
from os.path import isdir
from pandas import DataFrame

from utils.graphs2 import get_sizing, plot_info
from utils.load_store import parse_experiment

# Groupings
exp = 'Experiments'
d_mr_mm_s = ['dataset', 'miss_rate', 'miss_modality', 'seed']
bs_hr_a_i = ['batch_size', 'hint_rate', 'alpha', 'iterations']
gi = ['generator_initialization']
gs = ['generator_sparsity']
di = ['discriminator_initialization']
ds = ['discriminator_sparsity']


# -- Helper functions -------------------------------------------------------------------------------------------------

def get_Gsm_Dsm(d_mr_mm_s_group):
    """Get the unique Generator and Discriminator settings (initialization and sparsity).

    :param d_mr_mm_s_group: the experiments grouped by dataset, miss_rate, miss_modality and seed

    :return:
    - Gsm: the unique Generator settings (initialization and sparsity)
    - Dsm: the unique Discriminator settings (initialization and sparsity)
    - nGsm: the number of unique Generator settings (initialization and sparsity)
    - nDsm: the number of unique Discriminator settings (initialization and sparsity)
    """

    if type(d_mr_mm_s_group) == DataFrame:
        Gsm = d_mr_mm_s_group.drop_duplicates(subset=gi + gs)
        Gsm = Gsm[gi + gs].values.tolist()

        Dsm = d_mr_mm_s_group.drop_duplicates(subset=di + ds)
        Dsm = Dsm[di + ds].values.tolist()

    else:  # type(d_mr_mm_s_group) == dict
        Gsm = list({(key[4], key[5]) for key in d_mr_mm_s_group.keys()})
        Dsm = list({(key[6], key[7]) for key in d_mr_mm_s_group.keys()})

    Gsm.sort()
    Dsm.sort()
    nGsm, nDsm = len(Gsm), len(Dsm)

    return Gsm, Dsm, nGsm, nDsm


def prepare_plot(nGsm, nDsm, ax_width=6.4, ax_height=4.8, share_axis=False):
    """Prepare a plot:
    (1) Set the correct number of rows and columns
    (2) Turn off the unused subplots
    (3) Determine the locations of the experiment, system information and legend.

    :param nGsm: the number of unique Generator settings (initialization and sparsity)
    :param nDsm: the number of unique Discriminator settings (initialization and sparsity)
    :param ax_width: the width of the subplots
    :param ax_height: the height of the subplots
    :param share_axis: plot the experiment, system information and legend to the same subplot

    :return:
    - fig: the figure containing the plots
    - axs: the subplots of the figure
    - info_ax: the subplot to print the experiment and system information to
    - legend_ax: the subplot to print the legend to
    - legend_loc: the location of the legend
    - y_title: the (relative) position of the title in the figure
    """

    if nDsm == 1 or nGsm == 1:
        # Plot parameters
        nrows = 1 if share_axis else 2
        width, height, left, right, top, bottom, wspace, _, y_title = get_sizing(2, nrows, ax_width, ax_height)
        fig, axs = plt.subplots(nrows, 2, figsize=(width, height))
        plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace)

        # Determine location of the experiment, system information and legend
        info_ax = axs[1] if share_axis else axs[1, 0]
        legend_ax = axs[1] if share_axis else axs[0, 1]
        legend_loc = 'upper left'

        # Set extra axes off
        info_ax.set_axis_off()
        legend_ax.set_axis_off()
        if not share_axis: axs[1, 1].set_axis_off()

    else:
        # Plot parameters
        nrows = max(nGsm, nDsm) + 1
        width, height, left, right, top, bottom, wspace, hspace, y_title = get_sizing(2, nrows, ax_width, ax_height)
        fig, axs = plt.subplots(nrows, 2, figsize=(width, height))
        plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace)

        # Remove unused subplots and use the first and second empty subplot for the legend and system information
        info_ax, legend_ax, axs_set_axis_off = None, None, []
        legend_loc = f'upper {"right" if share_axis else "left"}'

        for i in range(nGsm):
            if i >= nDsm:
                axs_set_axis_off.append((i + 1, 0))
                if not info_ax:
                    info_ax = (i + 1, 0)
                    if share_axis: legend_ax = (i + 1, 0)
                elif not legend_ax:
                    legend_ax = (i + 1, 0)

        for i in range(nDsm):
            if i >= nGsm:
                axs_set_axis_off.append((i + 1, 1))
                if not info_ax:
                    info_ax = (i + 1, 1)
                    if share_axis: legend_ax = (i + 1, 0)
                elif not legend_ax:
                    legend_ax = (i + 1, 1)

        if not info_ax or not legend_ax:
            # Plot parameters
            width, height, left, right, top, bottom, wspace, hspace = get_sizing(3, nrows, ax_width, ax_height)
            fig, axs = plt.subplots(nrows, 3, figsize=(width, height))
            plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace)

            for i in range(nrows): axs_set_axis_off.append((i, 2))
            if not info_ax:
                info_ax, legend_ax = (0, 2), (1, 2)
                legend_loc = 'upper left'
            else:  # not legend_ax
                legend_ax = (0, 2)

        # Turn off the unused subplots
        for coord in axs_set_axis_off: axs[coord[0], coord[1]].set_axis_off()
        info_ax = axs[info_ax[0], info_ax[1]]
        legend_ax = axs[legend_ax[0], legend_ax[1]]

    return fig, axs, info_ax, legend_ax, legend_loc, y_title


def prepare_text(dataset, miss_rate, miss_modality, seed, sys_info=None):
    """Prepare the text for printing.

    :param dataset: the dataset
    :param miss_rate: the miss rate
    :param miss_modality: the miss modality
    :param seed: the seed
    :param sys_info: the system information

    :return:
    - title: the title of the plot
    - text: the experiment (and system information)
    """

    title = f'{dataset}_{miss_rate}_{miss_modality}_{hex(seed)}'
    dataset = f'Dataset: {dataset}'
    miss_rate = f'Miss rate: {int(miss_rate * 100)}%'
    miss_modality = f'Miss modality: {miss_modality}'
    seed = f'Seed: {hex(seed)}'
    text = [exp, dataset, miss_rate, miss_modality, seed, ' ']
    if sys_info: text += sys_info

    return title, text


def get_experiments_from_info(experiments_info):
    """Get the experiments grouped by dataset, miss_rate, miss_modality and seed.

    :param experiments_info: a dictionary with the experiments information

    :return:
    - experiments: a dictionary with the experiments
    """

    experiments = {}
    for key, val in experiments_info.items():
        d_mr_mm_s_key, d_mr_mm_s_group = key[:4], key[4:]
        if d_mr_mm_s_key not in experiments:
            experiments.update({
                d_mr_mm_s_key: {
                    d_mr_mm_s_group: val
                }
            })
        else:  # Experiment already in dictionary
            experiments[d_mr_mm_s_key].update({
                d_mr_mm_s_group: val
            })

    return experiments


def prepare_subplot_params(ax, M, initialization, sparsity):
    """Prepare the subplot:
    (1) Set the title.
    (2) Get the subtitle.
    (3) Get the correct initialization and sparsity (Generator or Discriminator).

    :param ax: the subplot to write to
    :param M: the model this belongs to (Generator or Discriminator)
    :param initialization: the initialization of the other model (optional)
    :param sparsity: the sparsity of the other model (optional)

    :return:
    - subtitle: The label of the x-axis
    - init: the initialization to group by
    - sparsity: the sparsities to plot on the x-axis
    """

    if sparsity != 'all': sparsity = f'{int(sparsity * 100)}%'
    if M in ('G', 'generator'):
        ax.title.set_text(f'Discriminator: {initialization} {sparsity}')
        subtitle = 'Generator'
        init = ['generator_initialization']
        sparsity = 'generator_sparsity'
    else:  # ('D', 'discriminator')
        ax.title.set_text(f'Generator: {initialization} {sparsity}')
        subtitle = 'Discriminator'
        init = ['discriminator_initialization']
        sparsity = 'discriminator_sparsity'

    return subtitle, init, sparsity


def prepare_data_params(initialization, x=None):
    """Prepare the data parameters:
    (1) Recapitalize the initialization (after sorting).
    (2) Distinguish the labels (optional).
    (3) Get the primary and secondary colors.

    :param initialization: the initialization
    :param x: the labels

    :return:
    - initialization: the (recapitalized) initialization
    - x: the (distinct) labels
    - primary_color: the primary color
    - secondary_color: the secondary color
    """

    # Todo expand for different settings (not only initialization)
    if initialization == 'dense':
        primary_color = 'black'
        secondary_color = 'dimgray'
    elif initialization == 'random':
        if x is not None: x = [f' {x} ' for x in x]
        primary_color = 'tab:orange'
        secondary_color = 'orange'
    elif initialization == 'er':
        initialization = 'ER'
        if x is not None: x = [f'  {x}  ' for x in x]
        primary_color = 'tab:red'
        secondary_color = 'pink'
    elif initialization == 'errw':
        initialization = 'ERRW'
        if x is not None: x = [f'   {x}   ' for x in x]
        primary_color = 'tab:purple'
        secondary_color = 'mediumorchid'
    else:
        if x is not None: x = [f'    {x}    ' for x in x]
        primary_color = 'tab:green'
        secondary_color = 'limegreen'

    return initialization, x, primary_color, secondary_color


def plot_legend(ax, legend_ax, legend_loc, show_bs_hr_a_i, subtitle):
    """Plot the legend (once).

    :param ax: the current subplot (to get the legend from)
    :param legend_ax: the subplot to plot the legend to
    :param legend_loc: the legend location
    :param show_bs_hr_a_i: whether to show the batch_size, hint_rate, alpha and iterations
    :param subtitle: the x label
    """

    if not legend_ax.get_legend():
        handles, labels = ax.get_legend_handles_labels()
        handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]  # Remove error bars
        lgnd = legend_ax.legend(handles, labels, fontsize=12, loc=legend_loc)
        lgnd_title = f'Batch size, hint rate, alpha,\niterations and {subtitle.lower()} initialization' \
            if show_bs_hr_a_i else f'{subtitle} initialization'
        lgnd.set_title(lgnd_title, prop={'size': 13})


# -- Compile metrics and plot graphs ----------------------------------------------------------------------------------

def extract_log_info(logs, folder='output'):
    """Extract information from the experiment logs.

    :param logs: a list of logs
    :param folder: the folder containing the experiment logs

    :return: a dictionary with the experiment log information
    """

    exps = {}
    for log in logs:
        # Parse the experiment Todo update with the new params
        d, mr, mm, s, bs, hr, a, i, _, gi_, gs_, _, _, _, _, _, _, di_, ds_, _, _, _, _, _, _, _, _, _ \
            = parse_experiment(log, file=True)
        experiment = (d, mr, mm, s, bs, hr, a, i, gi_, gs_, di_, ds_)

        # Read the log
        with open(f'{folder}/{log}', 'r') as f:
            data = json.load(f)

        # Get imputation time
        it = data['imputation_time']
        it_total = it['total']
        it_preparation = it['preparation']
        it_s_gain = it['s_gain']
        it_finalization = it['finalization']

        # Add experiment to dictionary
        if experiment not in exps:
            exps.update({
                experiment: {
                    'imputation_time': {
                        'total': [it_total],
                        'preparation': [it_preparation],
                        's_gain': [it_s_gain],
                        'finalization': [it_finalization]
                    }
                }
            })
        else:  # Experiment already in dictionary (append)
            exps[experiment]['imputation_time']['total'].append(it_total)
            exps[experiment]['imputation_time']['preparation'].append(it_preparation)
            exps[experiment]['imputation_time']['s_gain'].append(it_s_gain)
            exps[experiment]['imputation_time']['finalization'].append(it_finalization)

    return exps


def compile_metrics(experiments, experiments_info, folder='analysis', verbose=False):
    """Compile the metrics of the provided experiments.
    Metrics: RMSE mean, std and improvement, and successes, total runs and success rate

    :param experiments: a Pandas DataFrame with the experiments to compile
    :param experiments_info: a dictionary with the experiments information
    :param folder: the folder to save the compiled metrics to
    :param verbose: enable verbose output to console

    :return:
    - exps: a Pandas DataFrame of the compiled metrics
    """

    if verbose: print('Compiling metrics...')

    # Calculate mean, std, successes, total_runs and success_rate
    exps = experiments.drop(['index', 'filetype'], axis='columns')
    exps = exps.groupby(d_mr_mm_s + bs_hr_a_i + gi + gs + di + ds, as_index=False).agg(['mean', 'std', 'count', 'size'])
    exps.columns = exps.columns.get_level_values(0) + exps.columns.get_level_values(1)
    exps.rename(columns={'rmsemean': 'rmse_mean', 'rmsestd': 'rmse_std', 'rmsecount': 'successes',
                         'rmsesize': 'total_runs'}, inplace=True)
    exps['success_rate'] = exps['successes'] / exps['total_runs']

    # Calculate RMSE improvement compared to dense
    dense = exps.loc[(exps['generator_sparsity'] == 0) & (exps['discriminator_sparsity'] == 0)]
    if not dense.empty:
        for d, mr, mm, s, bs, hr, a, i, _, _, _, _, dense_rmse_mean, _, _, _, _ in dense.values:
            match = exps.loc[(exps['dataset'] == d) & (exps['miss_rate'] == mr) & (exps['miss_modality'] == mm)
                             & (exps['seed'] == s) & (exps['batch_size'] == bs) & (exps['hint_rate'] == hr)
                             & (exps['alpha'] == a) & (exps['iterations'] == i)]

            exps.loc[match.index, 'rmse_improvement'] = 1 / (match['rmse_mean'] / dense_rmse_mean) - 1

        exps.insert(len(exps.columns) - 4, 'rmse_improvement', exps.pop('rmse_improvement'))  # Move column
        exps['rmse_improvement'] = exps['rmse_improvement'].round(3)  # Rounding

    # Add information from the logs
    for (d, mr, mm, s, bs, hr, a, i, gi_, gs_, di_, ds_), values in experiments_info.items():
        match = exps.loc[(exps['dataset'] == d) & (exps['miss_rate'] == mr) & (exps['miss_modality'] == mm)
                         & (exps['seed'] == s) & (exps['batch_size'] == bs) & (exps['hint_rate'] == hr)
                         & (exps['alpha'] == a) & (exps['iterations'] == i) & (exps['generator_initialization'] == gi_)
                         & (exps['generator_sparsity'] == gs_) & (exps['discriminator_initialization'] == di_)
                         & (exps['discriminator_sparsity'] == ds_)]

        # Imputation time
        it_total = values['imputation_time']['total']
        it_preparation = values['imputation_time']['preparation']
        it_s_gain = values['imputation_time']['s_gain']
        it_finalization = values['imputation_time']['finalization']

        cols = [
            'imputation_time_mean_total', 'imputation_time_mean_preparation', 'imputation_time_mean_s_gain',
            'imputation_time_mean_finalization', 'imputation_time_std_total', 'imputation_time_std_preparation',
            'imputation_time_std_s_gain', 'imputation_time_std_finalization', 'imputation_time_improvement_total',
            'imputation_time_improvement_preparation', 'imputation_time_improvement_s_gain',
            'imputation_time_improvement_finalization', 'imputation_time_min_total', 'imputation_time_min_preparation',
            'imputation_time_min_s_gain', 'imputation_time_min_finalization', 'imputation_time_max_total',
            'imputation_time_max_preparation', 'imputation_time_max_s_gain', 'imputation_time_max_finalization'
        ]
        vals = [
            timedelta(seconds=np.mean(it_total)), timedelta(seconds=np.mean(it_preparation)),
            timedelta(seconds=np.mean(it_s_gain)), timedelta(seconds=np.mean(it_finalization)),
            timedelta(seconds=np.std(it_total)), timedelta(seconds=np.std(it_preparation)),
            timedelta(seconds=np.std(it_s_gain)), timedelta(seconds=np.std(it_finalization)),
            np.nan, np.nan, np.nan, np.nan,
            timedelta(seconds=min(it_total)), timedelta(seconds=min(it_preparation)),
            timedelta(seconds=min(it_s_gain)), timedelta(seconds=min(it_finalization)),
            timedelta(seconds=max(it_total)), timedelta(seconds=max(it_preparation)),
            timedelta(seconds=max(it_s_gain)), timedelta(seconds=max(it_finalization))
        ]

        exps.loc[match.index, cols] = vals

    # Calculate imputation time improvements compared to dense
    dense = exps.loc[(exps['generator_sparsity'] == 0) & (exps['discriminator_sparsity'] == 0)]
    if not dense.empty:
        for (d, mr, mm, s, bs, hr, a, i, _, _, _, _, _, _, _, _, _, _, dense_itit, dense_itip, dense_itis, dense_itif,
             _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _) in dense.values:
            match = exps.loc[(exps['dataset'] == d) & (exps['miss_rate'] == mr) & (exps['miss_modality'] == mm)
                             & (exps['seed'] == s) & (exps['batch_size'] == bs) & (exps['hint_rate'] == hr)
                             & (exps['alpha'] == a) & (exps['iterations'] == i)]

            exps.loc[match.index, 'imputation_time_improvement_total'] \
                = 1 / (match['imputation_time_mean_total'] / dense_itit) - 1
            exps.loc[match.index, 'imputation_time_improvement_preparation'] \
                = 1 / (match['imputation_time_mean_preparation'] / dense_itip) - 1
            exps.loc[match.index, 'imputation_time_improvement_s_gain'] \
                = 1 / (match['imputation_time_mean_s_gain'] / dense_itis) - 1
            exps.loc[match.index, 'imputation_time_improvement_finalization'] \
                = 1 / (match['imputation_time_mean_finalization'] / dense_itif) - 1

        # Rounding
        exps['imputation_time_improvement_total'] = exps['imputation_time_improvement_total'].round(3)
        exps['imputation_time_improvement_preparation'] = exps['imputation_time_improvement_preparation'].round(3)
        exps['imputation_time_improvement_s_gain'] = exps['imputation_time_improvement_s_gain'].round(3)
        exps['imputation_time_improvement_finalization'] = exps['imputation_time_improvement_finalization'].round(3)

    # Rounding
    exps['rmse_mean'] = exps['rmse_mean'].round(4)
    exps['rmse_std'] = exps['rmse_std'].round(7)
    exps['success_rate'] = exps['success_rate'].round(2)
    exps['imputation_time_std_total'] = exps['imputation_time_std_total'].round(7)
    exps['imputation_time_std_preparation'] = exps['imputation_time_std_preparation'].round(7)
    exps['imputation_time_std_s_gain'] = exps['imputation_time_std_s_gain'].round(7)
    exps['imputation_time_std_finalization'] = exps['imputation_time_std_finalization'].round(7)

    # Save the metrics
    if verbose: print('Saving compiled metrics...')
    if not isdir(folder): mkdir(folder)
    exps.to_csv(f'{folder}/metrics.csv', index=False)

    return exps


def plot_rmse(experiments, sys_info=None, folder='analysis', verbose=False):
    """Plot the RMSE of the provided experiments.

    :param experiments: a Pandas DataFrame with the (non-compiled) experiments
    :param sys_info: the system info (in print ready format)
    :param folder: the folder to save the plots to
    :param verbose: enable verbose output to console
    """

    def subplot(ax, M_rmse_mean_std, M, legend_ax, initialization, sparsity):
        """Create a subplot per model setting.

        :param ax: the subplot to write to
        :param M_rmse_mean_std: the rmse mean and std group
        :param M: the model this belongs to (Generator or Discriminator)
        :param legend_ax: the subplot to write the legend to
        :param initialization: the initialization of the other model (optional)
        :param sparsity: the sparsity of the other model (optional)
        """

        # Subplot parameters
        subtitle, init, sparsity = prepare_subplot_params(ax, M, initialization, sparsity)

        # Only show batch_size, hint_rate, alpha and iterations if different settings were tested
        show_bs_hr_a_i = False if M_rmse_mean_std.groupby(bs_hr_a_i).ngroups == 1 else True

        # Group by batch_size, hint_rate, alpha, iterations, initialization
        for (batch_size, hint_rate, alpha, iterations, initialization), bs_hr_a_i_m_group \
                in M_rmse_mean_std.groupby(bs_hr_a_i + init):

            # Get datapoints
            x = bs_hr_a_i_m_group[sparsity]
            y = bs_hr_a_i_m_group['rmse']['mean']
            e = bs_hr_a_i_m_group['rmse']['std']

            # Data parameters
            initialization, _, primary_color, _ = prepare_data_params(initialization.lower())

            # Set label
            label = f'{batch_size}_{hint_rate}_{alpha}_{iterations}_{initialization}' if show_bs_hr_a_i \
                else f'{initialization[0].upper()}{initialization[1:]}'

            # Add experiment to plot
            if initialization == 'dense':
                y_mu = y.iloc[0]
                y_sigma = y_mu + e.iloc[0], y_mu - e.iloc[0]

                ax.axhline(y=y_mu, color=primary_color, linewidth=1, label=label)
                ax.axhline(y=y_sigma[0], color=primary_color, linewidth=1, linestyle='--')
                ax.axhline(y=y_sigma[1], color=primary_color, linewidth=1, linestyle='--')
                ax.axhspan(y_sigma[0], y_sigma[1], facecolor=primary_color, alpha=0.2)

            else:
                ax.errorbar(x, y, e, color=primary_color, marker='o', markersize=3, capsize=5, elinewidth=1,
                            label=label)
                ax.fill_between(x, y - e, y + e, color=primary_color, alpha=0.2)

            # Subplot parameters
            ax.title.set_size(16)
            ax.grid(True)
            ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1., decimals=0))  # Format sparsity as percentage
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))  # Format RMSE to 4 decimals
            ax.set_xlabel(f'{subtitle} sparsity', size=13)
            ax.set_ylabel('RMSE ± 1 SD', size=13)  # Todo CI
            ax.tick_params(labelsize=12)

        # Update the legend only on the first plot
        plot_legend(ax, legend_ax, 'upper left', show_bs_hr_a_i, subtitle)

    #  -- Plot RMSE ---------------------------------------------------------------------------------------------------

    if verbose: print(f'Plotting RMSE graphs...')

    # Group by dataset, miss_rate, miss_modality and seed
    exps = experiments.drop(['index', 'filetype'], axis='columns')
    for (dataset, miss_rate, miss_modality, seed), d_mr_mm_s_group in exps.groupby(d_mr_mm_s):

        # Prepare plot
        d_mr_mm_s_group.drop(d_mr_mm_s, axis='columns', inplace=True)
        Gsm, Dsm, nGsm, nDsm = get_Gsm_Dsm(d_mr_mm_s_group)
        fig, axs, info_ax, legend_ax, legend_loc, y_title = prepare_plot(nGsm, nDsm, 6.4, 4.8)
        title, text = prepare_text(dataset, miss_rate, miss_modality, seed, sys_info)
        if verbose: print(title)

        # Get G_group
        G_group = d_mr_mm_s_group.drop(ds + di, axis='columns')
        G_rmse_mean_std = G_group.groupby(bs_hr_a_i + gs + gi, as_index=False).agg(['mean', 'std'])

        # Get D_group
        D_group = d_mr_mm_s_group.drop(gs + gi, axis='columns')
        D_rmse_mean_std = D_group.groupby(bs_hr_a_i + ds + di, as_index=False).agg(['mean', 'std'])

        # Plot subplots
        if nDsm == 1 or nGsm == 1:
            # Show the influence of different settings for the Generator
            if nDsm == 1:
                subplot(axs[0, 0], G_rmse_mean_std, 'G', legend_ax, Dsm[0][0], Dsm[0][1])

            # Show the influence of different settings for the Discriminator
            else:  # nGsm == 1
                subplot(axs[0, 0], D_rmse_mean_std, 'D', legend_ax, Gsm[0][0], Gsm[0][1])

        else:  # Multiple settings used for both the Generator and Discriminator
            # Show the influence of different settings for the Generator (ignore Discriminator settings)
            subplot(axs[0, 0], G_rmse_mean_std, 'G', legend_ax, 'all', 'settings')

            # Show the influence of different settings for the Discriminator (ignore Generator settings)
            subplot(axs[0, 1], D_rmse_mean_std, 'D', legend_ax, 'all', 'settings')

            # Show the influence of different settings for the Generator for different Discriminator settings
            for i in range(nDsm):
                G_group = d_mr_mm_s_group.where(
                    (d_mr_mm_s_group['discriminator_initialization'] == Dsm[i][1])
                    & (d_mr_mm_s_group['discriminator_sparsity'] == Dsm[i][0])
                )
                G_group.drop(ds + di, axis='columns', inplace=True)
                G_rmse_mean_std = G_group.groupby(bs_hr_a_i + gs + gi, as_index=False).agg(['mean', 'std'])
                subplot(axs[i + 1, 0], G_rmse_mean_std, 'G', legend_ax, Dsm[i][0], Dsm[i][1])

            # Show the influence of different settings for the Discriminator for different Generator settings
            for i in range(nGsm):
                D_group = d_mr_mm_s_group.where(
                    (d_mr_mm_s_group['generator_initialization'] == Gsm[i][1])
                    & (d_mr_mm_s_group['generator_sparsity'] == Gsm[i][0])
                )
                D_group.drop(gs + gi, axis='columns', inplace=True)
                D_rmse_mean_std = D_group.groupby(bs_hr_a_i + ds + di, as_index=False).agg(['mean', 'std'])
                subplot(axs[i + 1, 1], D_rmse_mean_std, 'D', legend_ax, Gsm[i][0], Gsm[i][1])

        # Plot system information
        plot_info(info_ax, text)

        # Plot parameters
        plt.suptitle(title, size=24, y=y_title)

        if verbose: print(f'Saving plot...')
        path = f'{folder}/{title}_RMSE.pdf'
        plt.savefig(path, format='pdf', dpi=1200)

        plt.close()


def plot_success_rate(experiments, sys_info=None, folder='analysis', verbose=False):
    """Plot the success rate of the provided experiments.

    :param experiments: a Pandas DataFrame with the (non-compiled) experiments
    :param sys_info: the system info (in print ready format)
    :param folder: the folder to save the plots to
    :param verbose: enable verbose output to console
    """

    def subplot(ax, M_success_rate, M, legend_ax, legend_loc, initialization, sparsity):
        """Create a subplot per model setting.

        :param ax: the subplot to write to
        :param M_success_rate: the success rate group
        :param M: the model this belongs to (Generator or Discriminator)
        :param legend_ax: the subplot to write the legend to
        :param legend_loc: the location of the legend
        :param initialization: the initialization of the other model (optional)
        :param sparsity: the sparsity of the other model (optional)
        """

        # Subplot parameters
        subtitle, init, sparsity = prepare_subplot_params(ax, M, initialization, sparsity)

        # Only show batch_size, hint_rate, alpha and iterations if different settings were tested
        show_bs_hr_a_i = False if M_success_rate.groupby(bs_hr_a_i).ngroups == 1 else True

        # Group by batch_size, hint_rate, alpha, iterations, initialization
        M_success_rate[init[0]] = M_success_rate[init[0]].str.lower()  # Prevent ER(K)(RW) before dense
        for (batch_size, hint_rate, alpha, iterations, initialization), bs_hr_a_i_m_group \
                in M_success_rate.groupby(bs_hr_a_i + init):
            # Get datapoints
            x = bs_hr_a_i_m_group[sparsity].map('{:.0%}'.format)
            y = bs_hr_a_i_m_group['success_rate']

            # Data parameters
            initialization, x, primary_color, _ = prepare_data_params(initialization, x)

            # Set label
            label = f'{batch_size}_{hint_rate}_{alpha}_{iterations}_{initialization}' if show_bs_hr_a_i \
                else f'{initialization[0].upper()}{initialization[1:]}'

            # Add experiment to plot
            ax.bar(x, y, color=primary_color, label=label, zorder=3)

            # Plot parameters
            ax.title.set_size(16)
            ax.grid(True, axis='y', zorder=0)
            ax.set_xlabel(f'{subtitle} sparsity', size=13)
            ax.set_ylabel('Success rate', size=13)
            ax.set_ylim((0, 1.05))
            ax.tick_params(labelsize=12)

        # Update the legend only on the first plot
        plot_legend(ax, legend_ax, legend_loc, show_bs_hr_a_i, subtitle)

    #  -- Plot success rate -------------------------------------------------------------------------------------------

    if verbose: print(f'Plotting success rate graphs...')

    # Group by dataset, miss_rate, miss_modality and seed
    exps = experiments.drop(['index', 'filetype'], axis='columns')
    for (dataset, miss_rate, miss_modality, seed), d_mr_mm_s_group in exps.groupby(d_mr_mm_s):

        # Prepare plot
        d_mr_mm_s_group.drop(d_mr_mm_s, axis='columns', inplace=True)
        Gsm, Dsm, nGsm, nDsm = get_Gsm_Dsm(d_mr_mm_s_group)
        fig, axs, info_ax, legend_ax, legend_loc, y_title = prepare_plot(nGsm, nDsm, 12.8, 4.8,
                                                                         share_axis=True)
        title, text = prepare_text(dataset, miss_rate, miss_modality, seed, sys_info)
        if verbose: print(title)

        # Get G_group
        G_group = d_mr_mm_s_group.drop(ds + di, axis='columns')
        G_success_rate = G_group.groupby(bs_hr_a_i + gs + gi, as_index=False).agg(['count', 'size'])
        G_success_rate['success_rate'] = G_success_rate['rmse']['count'] / G_success_rate['rmse']['size']

        # Get D_group
        D_group = d_mr_mm_s_group.drop(gs + gi, axis='columns')
        D_success_rate = D_group.groupby(bs_hr_a_i + ds + di, as_index=False).agg(['count', 'size'])
        D_success_rate['success_rate'] = D_success_rate['rmse']['count'] / D_success_rate['rmse']['size']

        # Plot subplots
        if nDsm == 1 or nGsm == 1:  # Only one setting used for the Discriminator
            # Show the influence of different settings for the Generator
            if nDsm == 1:
                subplot(axs[0], G_success_rate, 'G', legend_ax, legend_loc, Dsm[0][0], Dsm[0][1])

            # Show the influence of different settings for the Discriminator
            else:  # nGsm == 1
                subplot(axs[0], D_success_rate, 'D', legend_ax, legend_loc, Gsm[0][0], Gsm[0][1])

            # Plot system information
            plot_info(axs[1], text, x=0.5)

        else:  # Multiple settings used for both the Generator and Discriminator
            # Show the influence of different settings for the Generator (ignore Discriminator settings)
            subplot(axs[0, 0], G_success_rate, 'G', legend_ax, legend_loc, 'all', 'settings')

            # Show the influence of different settings for the Discriminator (ignore Generator settings)
            subplot(axs[0, 1], D_success_rate, 'D', legend_ax, legend_loc, 'all', 'settings')

            # Show the influence of different settings for the Generator for different Discriminator settings
            for i in range(nDsm):
                G_group = d_mr_mm_s_group.where(
                    (d_mr_mm_s_group['discriminator_initialization'] == Dsm[i][1])
                    & (d_mr_mm_s_group['discriminator_sparsity'] == Dsm[i][0])
                )
                G_group.drop(ds + di, axis='columns', inplace=True)
                G_success_rate = G_group.groupby(bs_hr_a_i + gs + gi, as_index=False).agg(['count', 'size'])
                G_success_rate['success_rate'] = G_success_rate['rmse']['count'] / G_success_rate['rmse']['size']
                subplot(axs[i + 1, 0], G_success_rate, 'G', legend_ax, legend_loc, Dsm[i][0], Dsm[i][1])

            # Show the influence of different settings for the Discriminator for different Generator settings
            for i in range(nGsm):
                D_group = d_mr_mm_s_group.where(
                    (d_mr_mm_s_group['generator_initialization'] == Gsm[i][1])
                    & (d_mr_mm_s_group['generator_sparsity'] == Gsm[i][0])
                )
                D_group.drop(gs + gi, axis='columns', inplace=True)
                D_success_rate = D_group.groupby(bs_hr_a_i + ds + di, as_index=False).agg(['count', 'size'])
                D_success_rate['success_rate'] = D_success_rate['rmse']['count'] / D_success_rate['rmse']['size']
                subplot(axs[i + 1, 1], D_success_rate, 'D', legend_ax, legend_loc, Gsm[i][0], Gsm[i][1])

            # Plot system information
            plot_info(info_ax, text)

        # Plot parameters
        plt.suptitle(title, size=24, y=y_title)

        if verbose: print(f'Saving plot...')
        path = f'{folder}/{title}_success_rate.pdf'
        plt.savefig(path, format='pdf', dpi=1200)

        plt.close()


def plot_imputation_time(experiments_info, sys_info=None, folder='analysis', verbose=False):
    """Plot the imputation time of the provided experiments.

    :param experiments_info: a dictionary with the experiments information
    :param sys_info: the system info (in print ready format)
    :param folder: the folder to save the plots to
    :param verbose: enable verbose output to console
    """

    def update_group(group, key, val):
        """Prepare a group for plotting.

        :param group: the group dictionary to prepare
        :param key: the experiment(s) to plot
        :param val: the values associated with this experiment
        """

        if key not in group:
            group.update({
                key: {'imputation_time': val['imputation_time']}
            })
        else:  # Experiment already in dictionary
            group[key]['imputation_time']['total'] += val['imputation_time']['total']
            group[key]['imputation_time']['preparation'] += val['imputation_time']['preparation']
            group[key]['imputation_time']['s_gain'] += val['imputation_time']['s_gain']
            group[key]['imputation_time']['finalization'] += val['imputation_time']['finalization']

    def subplot(ax, M_imputation_time, M, legend_ax, legend_loc, sparsity, initialization):
        """Create a subplot per model setting.

        :param ax: the subplot to write to
        :param M_imputation_time: the imputation time group
        :param M: the model this belongs to (Generator or Discriminator)
        :param legend_ax: the subplot to write the legend to
        :param legend_loc: the location of the legend
        :param initialization: the initialization of the other model
        :param sparsity: the sparsity of the other model

        :return:
        - y_max: the maximum y value in the plot
        """

        # Subplot parameters
        subtitle, _, sparsity = prepare_subplot_params(ax, M, initialization, sparsity)

        # Only show batch_size, hint_rate, alpha and iterations if different settings were tested
        show_bs_hr_a_i = False if len({k[:4] for k in M_imputation_time.keys()}) == 1 else True

        # Group by batch_size, hint_rate, alpha, iterations, initialization
        groups = {}
        for s, v in M_imputation_time.items():
            k = s[:4] + (s[5].lower(),)  # Prevent ER(K)(RW) before dense and sort
            if k not in groups:
                groups.update({k: {s[4]: v}})
            else:  # Experiment already in dictionary
                groups[k].update({s[4]: v})
        groups = dict(sorted(groups.items()))

        y_max = 0
        for (batch_size, hint_rate, alpha, iterations, initialization), bs_hr_a_i_m_group in groups.items():
            bs_hr_a_i_m_group = dict(sorted(bs_hr_a_i_m_group.items()))

            # Get datapoints
            x = []
            y_total, y_preparation, y_s_gain, y_finalization = [], [], [], []
            e_total, e_preparation, e_s_gain, e_finalization = [], [], [], []
            for s, v in bs_hr_a_i_m_group.items():
                # X (labels)
                x.append(f'{int(s * 100)}%')

                # Get values
                total = v['imputation_time']['total']
                preparation = v['imputation_time']['preparation']
                s_gain = v['imputation_time']['s_gain']
                finalization = v['imputation_time']['finalization']

                # Y (heights)
                y_total.append(np.mean(total))
                y_preparation.append(np.mean(preparation))
                y_s_gain.append(np.mean(s_gain))
                y_finalization.append(np.mean(finalization))

                # Error bars
                e_total.append(np.std(total))
                e_preparation.append(np.std(preparation))
                e_s_gain.append(np.std(s_gain))
                e_finalization.append(np.std(finalization))

                # Chart height
                y_max = max([y_total[j] + e_total[j] for j in range(len(y_total))] + [y_max])

            # Data parameters
            initialization, x, primary_color, secondary_color = prepare_data_params(initialization, x)

            # Set label
            label = f'{batch_size}_{hint_rate}_{alpha}_{iterations}_{initialization}' if show_bs_hr_a_i \
                else f'{initialization[0].upper()}{initialization[1:]}'

            # Add experiment to plot Todo outliers
            ax.bar(x, y_preparation, color=secondary_color, zorder=3)

            ax.bar(x, y_s_gain, bottom=y_preparation, color=primary_color, label=label, zorder=3)

            bottom = [y_preparation[j] + y_s_gain[j] for j in range(len(y_preparation))]
            ax.bar(x, y_finalization, bottom=bottom, color=secondary_color, yerr=e_total, capsize=10,
                   ecolor='tab:gray', zorder=3)

            # Plot parameters
            ax.title.set_size(16)
            ax.grid(True, axis='y', zorder=0)
            # ax.yaxis.set_major_formatter(DateFormatter('%H:%M:%S'))  Todo time format
            ax.set_xlabel(f'{subtitle} sparsity', size=13)
            ax.set_ylabel('Imputation time (in seconds) ± 1 SD', size=13)  # Todo CI
            ax.tick_params(labelsize=12)

        # Update the legend only on the first plot
        plot_legend(ax, legend_ax, legend_loc, show_bs_hr_a_i, subtitle)

        return y_max

    #  -- Plot imputation time ----------------------------------------------------------------------------------------

    if verbose: print(f'Plotting imputation time graphs...')

    # Group by dataset, miss_rate, miss_modality and seed
    exps = get_experiments_from_info(experiments_info)
    for (dataset, miss_rate, miss_modality, seed), d_mr_mm_s_group in exps.items():

        # Prepare plot
        Gsm, Dsm, nGsm, nDsm = get_Gsm_Dsm(d_mr_mm_s_group)
        fig, axs, info_ax, legend_ax, legend_loc, y_title = prepare_plot(nGsm, nDsm, 12.8, 4.8,
                                                                         share_axis=True)
        title, text = prepare_text(dataset, miss_rate, miss_modality, seed, sys_info)
        if verbose: print(title)

        # Get G_group and D_group
        G_group, D_group = {}, {}
        for key, val in d_mr_mm_s_group.items(): update_group(G_group, key[:6], val)
        for key, val in d_mr_mm_s_group.items(): update_group(D_group, key[:4] + key[6:], val)

        # Plot subplots
        if nDsm == 1 or nGsm == 1:
            # Show the influence of different settings for the Generator (ignore Discriminator settings)
            if nDsm == 1:
                y_max = subplot(axs[0], G_group, 'G', legend_ax, legend_loc, Dsm[0][0], Dsm[0][1])

            # Show the influence of different settings for the Discriminator (ignore Generator settings)
            else:  # nGsm == 1
                y_max = subplot(axs[0], D_group, 'D', legend_ax, legend_loc, Gsm[0][0], Gsm[0][1])

            # Plot parameters
            axs[0].set_ylim(0, y_max * 1.05)

            # Plot system information
            plot_info(info_ax, text, x=0.5)

        else:  # Multiple settings used for both the Generator and Discriminator
            # Show the influence of different settings for the Generator (ignore Discriminator settings)
            y_max = subplot(axs[0, 0], G_group, 'G', legend_ax, legend_loc, 'all', 'settings')

            # Show the influence of different settings for the Discriminator (ignore Generator settings)
            y_max = max(y_max, subplot(axs[0, 1], D_group, 'D', legend_ax, legend_loc, 'all', 'settings'))

            # Show the influence of different settings for the Generator for different Discriminator settings
            for i in range(nDsm):
                G_group = {}
                for key, val in d_mr_mm_s_group.items():
                    if key[6] == Dsm[i][0] and key[7] == Dsm[i][1]: update_group(G_group, key[:6], val)
                y_max = max(y_max, subplot(axs[i + 1, 0], G_group, 'G', legend_ax, legend_loc, Dsm[i][0], Dsm[i][1]))

            # Show the influence of different settings for the Discriminator for different Generator settings
            for i in range(nGsm):
                D_group = {}
                for key, val in d_mr_mm_s_group.items():
                    if key[4] == Gsm[i][0] and key[5] == Gsm[i][1]: update_group(D_group, key[:4] + key[6:], val)
                y_max = max(y_max, subplot(axs[i + 1, 1], D_group, 'D', legend_ax, legend_loc, Gsm[i][0], Gsm[i][1]))

            # Plot parameters
            for i in range(nDsm + 1): axs[i, 0].set_ylim(0, y_max * 1.05)
            for i in range(nGsm + 1): axs[i, 1].set_ylim(0, y_max * 1.05)

            # Plot system information
            plot_info(info_ax, text)

        # Plot parameters
        plt.suptitle(title, size=24, y=y_title)

        if verbose: print(f'Saving plot...')
        path = f'{folder}/{title}_imputation_time.pdf'
        plt.savefig(path, format='pdf', dpi=1200)

        plt.close()
