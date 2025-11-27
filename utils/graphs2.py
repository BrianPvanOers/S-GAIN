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

"""Graphing functions for S-GAIN:

Helper functions:
(1) get_sizing: helper function to calculate the different sizes for the plot
(2) plot_info: helper function to plot the experiment and system information

Plot graphs:
(3) plot_graphs: load and plot the graphs (RMSE, imputation time, memory usage, energy_consumption, loss and FLOPs)
"""

import matplotlib.pyplot as plt

from datetime import timedelta
from matplotlib import ticker


# -- Helper functions -------------------------------------------------------------------------------------------------

def get_sizing(ncols, nrows, ax_width, ax_height, w_space=1.28, h_space=1.2):
    """Calculates the different sizes for the plot.

    :param ncols: the number of columns in the plot
    :param nrows: the number of rows in the plot
    :param ax_width: the width of the subplots
    :param ax_height: the height of the subplots
    :param w_space: the width of the whitespace
    :param h_space: the height of the whitespace

    :return:
    - fig_width: total width of the figure
    - fig_height: total height of the figure
    - left: left margin of the figure
    - right: right margin of the figure
    - top: top margin of the figure
    - bottom: bottom margin of the figure
    - wspace: the horizontal padding between two subplots
    - hspace: the vertical padding between two subplots
    - title: the (relative) position of the title in the figure
    """

    # Margins (absolute)
    left_abs = w_space
    right_abs = 0.52
    top_abs = 2.96
    bottom_abs = 1
    title_abs = 0.76

    # Subplots (absolute)
    ax_width_total = ax_width * ncols
    ax_height_total = ax_height * nrows

    # Padding (absolute)
    wspace_abs = w_space
    hspace_abs = h_space
    wspace_total = wspace_abs * (ncols - 1)
    hspace_total = hspace_abs * (nrows - 1)

    # Figure (absolute)
    fig_width = left_abs + ax_width_total + wspace_total + right_abs
    fig_height = top_abs + ax_height_total + hspace_total + bottom_abs

    # Margins (relative)
    left = left_abs / fig_width
    right = 1 - right_abs / fig_width
    top = 1 - top_abs / fig_height
    bottom = bottom_abs / fig_height
    title = (fig_height - title_abs) / fig_height

    # Padding (relative)
    wspace = wspace_abs / ax_width
    hspace = hspace_abs / ax_height

    return fig_width, fig_height, left, right, top, bottom, wspace, hspace, title


def plot_info(ax, text, x=0.0, y=0.97):
    """Plot the experiment and system information.

    :param ax: the subplot to print the experiment and/or experiment to
    :param text: a list of strings to print
    :param x: the x coordinate to start printing from
    :param y: the y coordinate to start printing from
    """

    h1 = 0.08
    h2 = 0.05

    y_ = y
    for txt in text:
        if txt in ['Experiment', 'Experiments']:
            ax.text(x, y, txt, fontsize=16, weight='bold')
            y -= h1
        elif txt in ['Data settings', 'S-GAIN settings']:
            ax.text(x, y, txt, fontsize=13, weight='bold')
            y -= h2
        elif txt == 'Generator settings':
            ax.text(x, y, txt, fontsize=12, weight='bold')
            y -= h2
        elif 'Generator prune rate' in txt:
            x += 0.25
            y = y_ - h1
            ax.text(x, y, txt, fontsize=12)
            y -= h2
        elif txt == 'Discriminator settings':
            ax.text(x, y, txt, fontsize=12, weight='bold')
            y -= h2
        elif txt == 'System information':
            x += 0.3
            y = y_
            ax.text(x, y, txt, fontsize=16, weight='bold')
            y -= h1
        else:
            ax.text(x, y, txt, fontsize=12)
            y -= h2


# -- Plot graphs ------------------------------------------------------------------------------------------------------

def plot_graphs(filepath, logs, experiment=None, sys_info=None, title=None):
    """Load and plot the graphs.

    Todo implement strategy

    :param filepath: the filepath for the graphs
    :param logs: a dictionary of the logs
    :param experiment: the experiment
    :param sys_info: the system info (in print ready format)
    :param title: the title (optional)
    """

    # Get nrows required
    nrows = (1 if experiment or sys_info else 0) + (1 if logs.get('rmse') else 0) \
            + (1 if logs.get('imputation_time') else 0) + (1 if logs.get('memory_usage') else 0) \
            + (1 if logs.get('energy_consumption') else 0) + (2 if logs.get('sparsity') else 0) \
            + (1 if logs.get('FLOPs') else 0) + (2 if logs.get('loss') else 0)

    # Stop if no logs are provided
    if nrows == 0: return

    # New plot
    width, height, left, right, top, bottom, wspace, hspace, y_title = get_sizing(1, nrows, 12.8, 4.8)
    fig, axs = plt.subplots(nrows, figsize=(width, height))

    index = 0
    if experiment or sys_info:
        text = []
        if experiment:  # Todo strategy
            exp = 'Experiment'
            data_settings = 'Data settings'
            dataset = f'Dataset: {experiment["dataset"]}'
            miss_rate = f'Miss rate: {int(experiment["miss_rate"] * 100)}%'
            miss_modality = f'Miss modality: {experiment["miss_modality"]}'
            seed = f'Seed: {hex(experiment["seed"])}'
            s_gain_settings = 'S-GAIN settings'
            version = f'Version: {experiment["version"]}'
            batch_size = f'Batch size: {experiment["batch_size"]}'
            hint_rate = f'Hint rate: {experiment["hint_rate"]}'
            alpha = f'Alpha: {experiment["alpha"]}'
            iterations = f'Iterations: {experiment["iterations"]}'
            clipping = f'Clipping: {experiment["clipping"]}'
            generator_settings = 'Generator settings'
            generator_initialization = f'Generator initialization: {experiment["generator_initialization"]}'
            generator_sparsity = f'Generator sparsity: {experiment["generator_sparsity"]}'
            generator_pruner = f'Generator pruner: {experiment["generator_pruner"]}'
            generator_prune_rate = f'Generator prune rate: {experiment["generator_prune_rate"]}'
            generator_prune_period = f'Generator prune period: {experiment["generator_prune_period"]}'
            generator_regrower = f'Generator regrower: {experiment["generator_regrower"]}'
            generator_regrow_rate = f'Generator regrow rate: {experiment["generator_regrow_rate"]}'
            generator_regrow_period = f'Generator regrow period: {experiment["generator_regrow_period"]}'
            discriminator_settings = 'Discriminator settings'
            discriminator_initialization = f'Discriminator initialization: {experiment["discriminator_initialization"]}'
            discriminator_sparsity = f'Discriminator sparsity: {experiment["discriminator_sparsity"]}'
            discriminator_pruner = f'Discriminator pruner: {experiment["discriminator_pruner"]}'
            discriminator_prune_rate = f'Discriminator prune rate: {experiment["discriminator_prune_rate"]}'
            discriminator_prune_period = f'Discriminator prune period: {experiment["discriminator_prune_period"]}'
            discriminator_regrower = f'Discriminator regrower: {experiment["discriminator_regrower"]}'
            discriminator_regrow_rate = f'Discriminator regrow rate: {experiment["discriminator_regrow_rate"]}'
            discriminator_regrow_period = f'Discriminator regrow period: {experiment["discriminator_regrow_period"]}'

            text += [
                exp, data_settings, dataset, miss_rate, miss_modality, seed, '',
                s_gain_settings, version, batch_size, hint_rate, alpha, iterations, clipping, '',
                generator_settings, generator_initialization, generator_sparsity, generator_pruner,
                generator_prune_rate, generator_prune_period, generator_regrower, generator_regrow_rate,
                generator_regrow_period, '',
                discriminator_settings, discriminator_initialization, discriminator_sparsity, discriminator_pruner,
                discriminator_prune_rate, discriminator_prune_period, discriminator_regrower, discriminator_regrow_rate,
                discriminator_regrow_period, ''
            ]

        if sys_info: text += sys_info

        # Plot info
        plot_info(axs[index], text)
        axs[index].set_axis_off()

        # Increase index
        index += 1

    if logs.get('rmse'):  # Plot RMSE
        axs[index].plot(logs['rmse']['log'], label=f'{logs["rmse"]["final"]:.4f}')

        len_log = len(logs['rmse']['log'])

        # RMSE parameters
        axs[index].set_title('RMSE per iteration')
        axs[index].title.set_size(16)
        axs[index].set_ylabel('RMSE', size=13)
        axs[index].set_xlabel('Iterations', size=13)
        axs[index].set_xlim(-len_log * 0.01, len_log * 1.01)
        axs[index].tick_params(labelsize=12)
        lgnd = axs[index].legend(fontsize=12)
        lgnd.set_title(title='Final RMSE', prop={'size': 13})
        axs[index].grid(True)

        # Increase index
        index += 1

    if logs.get('imputation_time'):  # Plot imputation time
        label = f'{timedelta(seconds=round(logs["imputation_time"]["total"]))}'
        axs[index].plot(logs['imputation_time']['log'], label=label, color='black')

        len_log = len(logs['imputation_time']['log'])

        # Plot parameters
        axs[index].set_title('Imputation time per iteration')
        axs[index].title.set_size(16)
        axs[index].set_ylabel('Time (in seconds)', size=13)
        axs[index].set_xlabel('Iterations', size=13)
        axs[index].set_xlim(-len_log * 0.01, len_log * 1.01)
        axs[index].tick_params(labelsize=12)
        lgnd = axs[index].legend(fontsize=12)
        lgnd.set_title(title='Total imputation time', prop={'size': 13})
        axs[index].grid(True)

        # Increase index
        index += 1

    if logs.get('memory_usage'):  # Plot memory usage
        axs[index].plot(logs['memory_usage']['log'])
        # Todo: plot the graph (use the legend to display the maximum)

        len_log = len(logs['memory_usage']['log'])

        # Plot parameters
        axs[index].set_title('Memory usage per iteration')
        axs[index].title.set_size(16)
        axs[index].set_ylabel('Memory usage (in MB)', size=13)
        axs[index].set_xlabel('Iterations', size=13)
        axs[index].set_xlim(-len_log * 0.01, len_log * 1.01)
        axs[index].tick_params(labelsize=12)
        lgnd = axs[index].legend(fontsize=12)
        lgnd.set_title(title='Total memory usage', prop={'size': 13})
        axs[index].grid(True)

        # Increase index
        index += 1

    if logs.get('energy_consumption'):  # Plot energy consumption
        axs[index].plot(logs['energy_consumption']['log'])
        # Todo: plot the graph (use the legend to display the total)

        len_log = len(logs['energy_consumption']['log'])

        # Plot parameters
        axs[index].set_title('Energy consumption per iteration')
        axs[index].title.set_size(16)
        axs[index].set_ylabel('Energy consumption (in joule)', size=13)
        axs[index].set_xlabel('Iterations', size=13)
        axs[index].set_xlim(-len_log * 0.01, len_log * 1.01)
        axs[index].tick_params(labelsize=12)
        lgnd = axs[index].legend(fontsize=12)
        lgnd.set_title(title='Total energy consumption', prop={'size': 13})
        axs[index].grid(True)

        # Increase index
        index += 1

    if logs.get('sparsity'):  # Plot sparsity

        # Labels Todo add average and maximum?
        label_S_GAIN = f'S-GAIN: {logs["sparsity"]["initial"] * 100:.1f}%' \
                       f' | {logs["sparsity"]["final"] * 100:.1f}%' \
                       f' | {logs["sparsity"]["minimum"] * 100:.1f}%'

        label_G = f'Overall: {logs["sparsity"]["generator"]["initial"] * 100:.1f}%' \
                  f' | {logs["sparsity"]["generator"]["final"] * 100:.1f}%' \
                  f' | {logs["sparsity"]["generator"]["minimum"] * 100:.1f}%'
        label_G_W1 = f'Layer 1: {logs["sparsity"]["generator"]["G_W1"]["initial"] * 100:.1f}%' \
                     f' | {logs["sparsity"]["generator"]["G_W1"]["final"] * 100:.1f}%' \
                     f' | {logs["sparsity"]["generator"]["G_W1"]["minimum"] * 100:.1f}%'
        label_G_W2 = f'Layer 2: {logs["sparsity"]["generator"]["G_W2"]["initial"] * 100:.1f}%' \
                     f' | {logs["sparsity"]["generator"]["G_W2"]["final"] * 100:.1f}%' \
                     f' | {logs["sparsity"]["generator"]["G_W2"]["minimum"] * 100:.1f}%'
        label_G_W3 = f'Layer 3: {logs["sparsity"]["generator"]["G_W1"]["initial"] * 100:.1f}%' \
                     f' | {logs["sparsity"]["generator"]["G_W1"]["final"] * 100:.1f}%' \
                     f' | {logs["sparsity"]["generator"]["G_W1"]["minimum"] * 100:.1f}%'

        label_D = f'Overall: {logs["sparsity"]["discriminator"]["initial"] * 100:.1f}%' \
                  f' | {logs["sparsity"]["discriminator"]["final"] * 100:.1f}%' \
                  f' | {logs["sparsity"]["discriminator"]["minimum"] * 100:.1f}%'
        label_D_W1 = f'Layer 1: {logs["sparsity"]["discriminator"]["D_W1"]["initial"] * 100:.1f}%' \
                     f' | {logs["sparsity"]["discriminator"]["D_W1"]["final"] * 100:.1f}%' \
                     f' | {logs["sparsity"]["discriminator"]["D_W1"]["minimum"] * 100:.1f}%'
        label_D_W2 = f'Layer 2: {logs["sparsity"]["discriminator"]["D_W2"]["initial"] * 100:.1f}%' \
                     f' | {logs["sparsity"]["discriminator"]["D_W2"]["final"] * 100:.1f}%' \
                     f' | {logs["sparsity"]["discriminator"]["D_W2"]["minimum"] * 100:.1f}%'
        label_D_W3 = f'Layer 3: {logs["sparsity"]["discriminator"]["D_W3"]["initial"] * 100:.1f}%' \
                     f' | {logs["sparsity"]["discriminator"]["D_W3"]["final"] * 100:.1f}%' \
                     f' | {logs["sparsity"]["discriminator"]["D_W3"]["minimum"] * 100:.1f}%'

        # Plots
        axs[index].plot(logs['sparsity']['log'], label=label_S_GAIN, color='black')
        axs[index].plot(logs['sparsity']['generator']['log'], label=label_G, color='navy')
        axs[index].plot(logs['sparsity']['generator']['G_W1']['log'], label=label_G_W1, color='blue')
        axs[index].plot(logs['sparsity']['generator']['G_W2']['log'], label=label_G_W2, color='dodgerblue')
        axs[index].plot(logs['sparsity']['generator']['G_W3']['log'], label=label_G_W3, color='deepskyblue')

        axs[index + 1].plot(logs['sparsity']['log'], label=label_S_GAIN, color='black')
        axs[index + 1].plot(logs['sparsity']['discriminator']['log'], label=label_D, color='darkred')
        axs[index + 1].plot(logs['sparsity']['discriminator']['D_W1']['log'], label=label_D_W1, color='tab:red')
        axs[index + 1].plot(logs['sparsity']['discriminator']['D_W2']['log'], label=label_D_W2, color='lightcoral')
        axs[index + 1].plot(logs['sparsity']['discriminator']['D_W3']['log'], label=label_D_W3, color='pink')

        len_log = len(logs['sparsity']['log'])

        # Generator parameters
        axs[index].set_title('Generator sparsity per iteration')
        axs[index].title.set_size(16)
        axs[index].set_ylabel('Sparsity', size=13)
        axs[index].set_ylim(0, 1)
        axs[index].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1., decimals=0))
        axs[index].set_xlabel('Iterations', size=13)
        axs[index].set_xlim(-len_log * 0.01, len_log * 1.01)
        axs[index].tick_params(labelsize=12)
        lgnd = axs[index].legend(fontsize=12)
        lgnd.set_title(title='Sparsity: Initial | Final | Minimum', prop={'size': 13})
        axs[index].grid(True)

        # Discriminator parameters
        axs[index + 1].set_title('Discriminator sparsity per iteration')
        axs[index + 1].title.set_size(16)
        axs[index + 1].set_ylabel('Sparsity', size=13)
        axs[index + 1].set_ylim(0, 1)
        axs[index + 1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1., decimals=0))
        axs[index + 1].set_xlabel('Iterations', size=13)
        axs[index + 1].set_xlim(-len_log * 0.01, len_log * 1.01)
        axs[index + 1].tick_params(labelsize=12)
        lgnd = axs[index + 1].legend(fontsize=12)
        lgnd.set_title(title='Sparsity: Initial | Final | Minimum', prop={'size': 13})
        axs[index + 1].grid(True)

        # Increase index
        index += 2

    if logs.get('FLOPs'):  # Plot FLOPs
        axs[index].plot(logs['FLOPs']['log'], label=f'S-GAIN: {logs["FLOPs"]["total"]}', color='black')
        axs[index].plot(logs['FLOPs']['generator']['log'],
                        label=f'Generator: {logs["FLOPs"]["generator"]["total"]}', color='tab:blue')
        axs[index].plot(logs['FLOPs']['discriminator']['log'],
                        label=f'Discriminator: {logs["FLOPs"]["discriminator"]["total"]}', color='tab:red')

        len_log = len(logs['FLOPs']['log'])

        # Plot parameters
        axs[index].set_title('FLOPs per iteration')
        axs[index].title.set_size(16)
        axs[index].set_ylabel('FLOPs', size=13)
        axs[index].set_xlabel('Iterations', size=13)
        axs[index].set_xlim(-len_log * 0.01, len_log * 1.01)
        axs[index].tick_params(labelsize=12)
        lgnd = axs[index].legend(fontsize=12)
        lgnd.set_title(title='Total FLOPs', prop={'size': 13})
        axs[index].grid(True)

        # Increase index
        index += 1

    if logs.get('loss'):  # Plot losses (Cross Entropy and MSE)
        axs[index].plot(logs['loss']['cross_entropy']['generator']['log'], label='Generator loss', color='tab:blue')
        axs[index].plot(logs['loss']['cross_entropy']['discriminator']['log'], label='Discriminator loss',
                        color='tab:red')

        axs[index + 1].plot(logs['loss']['MSE']['log'], label='MSE loss', color='black')

        len_log = len(logs['loss']['MSE']['log'])

        # Cross Entropy parameters
        axs[index].title.set_text('Learning curves')
        axs[index].title.set_size(16)
        axs[index].set_ylabel('Cross Entropy', size=13)
        axs[index].set_xlabel('Iterations', size=13)
        axs[index].set_xlim(-len_log * 0.01, len_log * 1.01)
        axs[index].tick_params(labelsize=12)
        axs[index].legend(fontsize=12)
        axs[index].grid(True)

        # MSE parameters
        axs[index + 1].title.set_text('Learning curves')
        axs[index + 1].title.set_size(16)
        axs[index + 1].set_ylabel('MSE', size=13)
        axs[index + 1].set_xlabel('Iterations', size=13)
        axs[index + 1].set_xlim(-len_log * 0.01, len_log * 1.01)
        axs[index + 1].tick_params(labelsize=12)
        axs[index + 1].legend(fontsize=12)
        axs[index + 1].grid(True)

        # Increase index
        index += 2

    # Plot parameters
    quarter = int(len(title) / 4)
    title = title[:quarter] \
            + title[quarter:quarter * 2].replace('_', '\n_', 1) \
            + title[quarter * 2:quarter * 3].replace('_', '\n_', 1) \
            + title[quarter * 3:].replace('_', '\n_', 1)
    plt.suptitle(title, size=22, y=y_title)
    plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace)

    # Save plot
    plt.savefig(filepath, format='png')
