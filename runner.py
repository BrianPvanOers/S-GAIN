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

"""Main function for S-GAIN."""

import argparse
import json
import os

import numpy as np

from models.s_gain_TFv1_FP32 import s_gain as s_gain_TFv1_FP32
from models.s_gain_TFv2_INT8 import s_gain as s_gain_TFv2_INT8
from monitors.monitor import Monitor
from utils.data_loader import data_loader
from utils.graphs2 import plot_graphs
from utils.load_store import get_filepaths, save_imputation, save_logs, system_information
from utils.metrics import get_rmse


# ---------------------------------------------------------------------------------------------------------------------

def main(args):
    """Main function for S-GAIN:
    1. Load the parameters from the run_config
    2. Load and introduce missing elements in the data according to the provided miss rate and modality
    3. Call S-GAIN
    4. Save the imputed data, logs and trained model and plot the graphs

    :param args:
    - verbose: enable verbose output to the console
    - no_system_information: don't log system information

    :return:
    - imputed_data_x: the imputed data
    - rmse: the root mean squared error
    """

    # Load run config Todo implement strategies
    with open('temp/run_config.json', 'r') as f:
        config = json.load(f)

    dataset = config['dataset']
    miss_rate = config['miss_rate']
    miss_modality = config['miss_modality']
    seed = config['seed']
    store_prepared_dataset = config['store_prepared_dataset']
    version = config['version']
    batch_size = config['batch_size']
    hint_rate = config['hint_rate']
    alpha = config['alpha']
    iterations = config['iterations']
    clipping = config['clipping']
    generator_sparsity = config['generator_sparsity']
    generator_initialization = config['generator_initialization']
    generator_pruner = config['generator_pruner']
    generator_prune_rate = config['generator_prune_rate']
    generator_prune_period = config['generator_prune_period']
    generator_regrower = config['generator_regrower']
    generator_regrow_rate = config['generator_regrow_rate']
    generator_regrow_period = config['generator_regrow_period']
    discriminator_sparsity = config['discriminator_sparsity']
    discriminator_initialization = config['discriminator_initialization']
    discriminator_pruner = config['discriminator_pruner']
    discriminator_prune_rate = config['discriminator_prune_rate']
    discriminator_prune_period = config['discriminator_prune_period']
    discriminator_regrower = config['discriminator_regrower']
    discriminator_regrow_rate = config['discriminator_regrow_rate']
    discriminator_regrow_period = config['discriminator_regrow_period']
    output_folder = config['output_folder']
    no_imputation = config['no_imputation']
    no_log = config['no_log']
    no_graphs = config['no_graphs']
    no_model = config['no_model']
    enable_rmse_monitor = config['enable_rmse_monitor']
    enable_imputation_time_monitor = config['enable_imputation_time_monitor']
    enable_memory_usage_monitor = config['enable_memory_usage_monitor']
    enable_energy_consumption_monitor = config['enable_energy_consumption_monitor']
    enable_sparsity_monitor = config['enable_sparsity_monitor']
    enable_FLOPs_monitor = config['enable_FLOPs_monitor']
    enable_loss_monitor = config['enable_loss_monitor']
    verbose = args.verbose
    no_system_information = args.no_system_information

    if seed is None: seed = np.random.randint(2 ** 31)

    # Exit program if a modality is not implemented yet Todo: implement the modalities
    not_implemented = ['MAR', 'MNAR', 'ERK', 'ERKRW', 'SNIP', 'GraSP', 'RSensitivity']
    if miss_modality in not_implemented:
        print(f'Miss modality {miss_modality} is not implemented.\nExiting program...')
        return [None] * 2
    if generator_initialization in not_implemented:
        print(f'Generator modality {discriminator_initialization} is not implemented.\nExiting program...')
        return [None] * 2
    if discriminator_initialization in not_implemented:
        print(f'Discriminator modality {discriminator_initialization} is not implemented.\nExiting program...')
        return [None] * 2

    # Name the experiment Todo implement new params
    experiment = f'S-GAIN_{dataset}_MR_{miss_rate}_MM_{miss_modality}_S_0x{seed:08x}_V_{version}_BS_{batch_size}' \
                 f'_HR_{hint_rate}_a_{alpha}_i_{iterations}_C_{1 if clipping else 0}_GI_{generator_initialization}' \
                 f'_GS_{generator_sparsity}_GP_{generator_pruner}_GPR_{generator_prune_rate}' \
                 f'_GPP_{generator_prune_period}_GR_{generator_regrower}_GRR_{generator_regrow_rate}' \
                 f'_GRP_{generator_regrow_period}_DI_{discriminator_initialization}_DS_{discriminator_sparsity}' \
                 f'_DP_{discriminator_pruner}_DPR_{discriminator_prune_rate}_DPP_{discriminator_prune_period}' \
                 f'_DR_{discriminator_regrower}_DRR_{discriminator_regrow_rate}_DRP_{discriminator_regrow_period}'
    if verbose: print(f'\n{experiment}')

    # Load the data with missing elements
    data_x, miss_data_x, data_mask = data_loader(dataset, miss_rate, miss_modality, seed,
                                                 store_prepared_dataset=store_prepared_dataset,
                                                 verbose='main' if verbose else False)

    # Initialize monitor
    monitor = None if no_log and no_model else Monitor(
        data_x, data_mask, enable_rmse_monitor=enable_rmse_monitor,
        enable_imputation_time_monitor=enable_imputation_time_monitor,
        enable_memory_usage_monitor=enable_memory_usage_monitor,
        enable_energy_consumption_monitor=enable_energy_consumption_monitor,
        enable_sparsity_monitor=enable_sparsity_monitor, enable_FLOPs_monitor=enable_FLOPs_monitor,
        enable_loss_monitor=enable_loss_monitor, experiment=experiment, verbose=verbose
    )

    # S-GAIN Todo implement new params
    if version == 'TFv1_FP32':
        imputed_data_x = s_gain_TFv1_FP32(
            miss_data_x, batch_size=batch_size, hint_rate=hint_rate, alpha=alpha, iterations=iterations,
            generator_initialization=generator_initialization, generator_sparsity=generator_sparsity,
            discriminator_initialization=discriminator_initialization, discriminator_sparsity=discriminator_sparsity,
            verbose=verbose, no_model=no_model, monitor=monitor
        )
    elif version == 'TFv2_INT8':
        imputed_data_x = s_gain_TFv2_INT8(
            miss_data_x, batch_size=batch_size, hint_rate=hint_rate, alpha=alpha, iterations=iterations,
            generator_initialization=generator_initialization, generator_sparsity=generator_sparsity,
            discriminator_initialization=discriminator_initialization, discriminator_sparsity=discriminator_sparsity,
            verbose=verbose, no_model=no_model, monitor=monitor
        )
    else:  # This should not happen
        print(f'Invalid version: {version}.\nExiting program...')
        return [None] * 2

    # Calculate the RMSE
    rmse = get_rmse(data_x, imputed_data_x, data_mask, rounding=True)
    if verbose: print(f'RMSE: {rmse}')

    # Get filepaths and store to run log_and_graphs.py later
    filepath_imputation, filepath_log, filepath_model, filepath_graphs = get_filepaths(output_folder, experiment, rmse)

    # Save imputation
    if not no_imputation and 'nan' not in filepath_imputation:
        if verbose: print('Saving imputation...')
        save_imputation(filepath_imputation, imputed_data_x)

    # Save (trained) model
    if monitor and not no_model and 'nan' not in filepath_model:
        if verbose: print('Saving (trained) model...')
        monitor.save_model(filepath_model)

    # Save log and graphs
    if monitor and not no_log:
        if verbose: print('Saving logs...')
        sys_info = system_information() if not no_system_information else None
        logs, exp = save_logs(filepath_log, experiment, sys_info)

        # Save graphs
        if not no_graphs:
            if verbose: print('Plotting graphs...')
            sys_info = system_information(print_ready=True) if not no_system_information else None
            plot_graphs(filepath_graphs, logs, experiment=exp, sys_info=sys_info, title=experiment)

    if verbose: print(f'Finished.')

    return imputed_data_x, rmse


if __name__ == '__main__':
    # Parsers
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--verbose', '-v',
        help='enable verbose output to the console',
        action='store_true')
    parser.add_argument(
        '--no_system_information', '-nsi',
        help="don't log system information",
        action='store_true')
    args = parser.parse_args()

    # Call main
    main(args)
