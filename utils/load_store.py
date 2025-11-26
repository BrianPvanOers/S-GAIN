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

"""Load and store operations for S-GAIN:

Parsers:
(1) parse_experiment: parse the experiment
(2) parse_files: parse the output files to a pandas dataframe

Load operations:
(3) load_config: load the config into a dictionary
(4) get_experiments_from_config: get a pandas DataFrame with the config of each experiment
(5) get completed_experiments: get a Pandas DataFrame with the completed experiments
(6) read_bin: read a (temporary) binary file

Store operations:
(7) get_filepaths: create the necessary directory and return the appropriate filepaths
(8) save_imputation: save the imputed data to a csv file
(9) save_logs: compile and save the logs to a json file

Other functions:
(10) system_info: get the system information
"""

import cpuinfo
import json
import platform
import psutil
import struct
import subprocess
import wmi

import pandas as pd

from os import listdir, makedirs
from os.path import isdir, isfile

from utils.standardizers import standardize, standardize_dataset, standardize_miss_modality, standardize_version, \
    standardize_init, standardize_pruner, standardize_regrower


# -- Parsers ----------------------------------------------------------------------------------------------------------

def parse_experiment(experiment, file=False):
    """Parse the experiment.

    Todo implement two types, init prune regrow and strategy

    :param experiment: the name of the experiment
    :param file: whether parsing a file or not

    :return:
    - False: if the experiment is not in S-GAIN format
    - dataset: the dataset used
    - miss_rate: the probability of missing elements in the data
    - miss_modality: the modality of missing data (MCAR, MAR, MNAR)
    - seed: the seed used to introduce missing elements in the data
    - batch_size: the number of samples in mini-batch
    - hint_rate: the hint probability
    - alpha: the hyperparameter
    - iterations (epochs): the number of training iterations
    - generator_initialization: the initialization and pruning and regrowth strategy of the generator
    - generator_sparsity: the probability of sparsity in the generator
    - discriminator_initialization: the initialization and pruning and regrowth strategy of the discriminator
    - discriminator_sparsity: the probability of sparsity in the discriminator
    - rmse: the RMSE (if parsing a file)
    - index: the index of the experiment (if parsing a file)
    - filetype: the type of file (imputed_data, log, model or graphs)
    """

    # Check if the experiment belongs to S-GAIN
    if not experiment.startswith('S-GAIN'): return False

    # Remove the file extension
    if file: experiment = experiment.rsplit('.', 1)[0]

    # Parse experiment
    _, rest = experiment.split('S-GAIN_')
    dataset, rest = rest.split('_MR_')
    miss_rate, rest = rest.split('_MM_')
    miss_rate = float(miss_rate)
    miss_modality, rest = rest.split('_S_')
    seed, rest = rest.split('_V_')
    seed = int(seed, 16)
    version, rest = rest.split('_BS_')
    batch_size, rest = rest.split('_HR_')
    batch_size = int(batch_size)
    hint_rate, rest = rest.split('_a_')
    hint_rate = float(hint_rate)
    alpha, rest = rest.split('_i_')
    alpha = float(alpha)
    iterations, rest = rest.split('_C_')
    iterations = int(iterations)
    clipping, rest = rest.split('_GI_')
    clipping = True if int(clipping) else False
    generator_initialization, rest = rest.split('_GS_')
    generator_sparsity, rest = rest.split('_GP_')
    generator_sparsity = float(generator_sparsity)
    generator_pruner, rest = rest.split('_GPR_')
    generator_prune_rate, rest = rest.split('_GPP_')
    generator_prune_rate = float(generator_prune_rate)
    generator_prune_period, rest = rest.split('_GR_')
    generator_prune_period = int(generator_prune_period)
    generator_regrower, rest = rest.split('_GRR_')
    generator_regrow_rate, rest = rest.split('_GRP_')
    generator_regrow_rate = float(generator_regrow_rate)
    generator_regrow_period, rest = rest.split('_DI_')
    generator_regrow_period = int(generator_regrow_period)
    discriminator_initialization, rest = rest.split('_DS_')
    discriminator_sparsity, rest = rest.split('_DP_')
    discriminator_sparsity = float(discriminator_sparsity)
    discriminator_pruner, rest = rest.split('_DPR_')
    discriminator_prune_rate, rest = rest.split('_DPP_')
    discriminator_prune_rate = float(discriminator_prune_rate)
    discriminator_prune_period, rest = rest.split('_DR_')
    discriminator_prune_period = int(discriminator_prune_period)
    discriminator_regrower, rest = rest.split('_DRR_')
    discriminator_regrow_rate, rest = rest.split('_DRP_')
    discriminator_regrow_rate = float(discriminator_regrow_rate)

    if file:  # rmse, index, filetype
        discriminator_regrow_period, rest = rest.split('_DEC_')
        discriminator_regrow_period = int(discriminator_regrow_period)

        rests = rest.split('_', 1)
        rmse = float(rests[0])

        if len(rests) == 1:
            index = 0
            filetype = 'imputed_data'
        else:  # index, filetype
            index_filetype = rests[1].split('_', 1)
            if len(index_filetype) == 1:  # index or filetype
                if index_filetype[0].isdigit():  # index
                    index = int(index_filetype[0])
                    filetype = 'imputed_data'
                else:  # filetype
                    index = 0
                    filetype = index_filetype[0]
            elif index_filetype[0].isdigit():  # index and filetype
                index = int(index_filetype[0])
                filetype = index_filetype[1]
            else:  # filetype
                index = 0
                filetype = rests[1]

        return dataset, miss_rate, miss_modality, seed, version, batch_size, hint_rate, alpha, iterations, clipping, \
            generator_initialization, generator_sparsity, generator_pruner, generator_prune_rate, \
            generator_prune_period, generator_regrower, generator_regrow_rate, generator_regrow_period, \
            discriminator_initialization, discriminator_sparsity, discriminator_pruner, discriminator_prune_rate, \
            discriminator_prune_period, discriminator_regrower, discriminator_regrow_rate, \
            discriminator_regrow_period, rmse, index, filetype

    else:
        discriminator_regrow_period = int(rest)

        return dataset, miss_rate, miss_modality, seed, version, batch_size, hint_rate, alpha, iterations, clipping, \
            generator_initialization, generator_sparsity, generator_pruner, generator_prune_rate, \
            generator_prune_period, generator_regrower, generator_regrow_rate, generator_regrow_period, \
            discriminator_initialization, discriminator_sparsity, discriminator_pruner, discriminator_prune_rate, \
            discriminator_prune_period, discriminator_regrower, discriminator_regrow_rate, discriminator_regrow_period


def parse_files(filepath='output', filetype=None, files=None):
    """Parse the output files to a pandas dataframe.

    Todo update with the new parameters

    :param filepath: the output filepath
    :param filetype: only return files of this type (imputed_data, log, model, etc.)
    :param files: a list of files to parse (optional)

    :return:
    - df_files: a Pandas DataFrame with all the experiments
    """

    files = [parse_experiment(file, file=True) for file in files] if files \
        else [parse_experiment(file, file=True) for file in listdir(filepath)] if isdir(filepath) \
        else []

    # Remove files that don't belong to S-GAIN (parse_experiment returns False for non S-GAIN files)
    files = [file for file in files if file]

    header = [
        'dataset', 'miss_rate', 'miss_modality', 'seed', 'version', 'batch_size', 'hint_rate', 'alpha', 'iterations',
        'clipping', 'generator_initialization', 'generator_sparsity', 'generator_pruner', 'generator_prune_rate',
        'generator_prune_period', 'generator_regrower', 'generator_regrow_rate', 'generator_regrow_period',
        'discriminator_initialization', 'discriminator_sparsity', 'discriminator_pruner', 'discriminator_prune_rate',
        'discriminator_prune_period', 'discriminator_regrower', 'discriminator_regrow_rate',
        'discriminator_regrow_period', 'rmse', 'index', 'filetype'
    ]

    # Only keep selected filetype
    if filetype: files = [file for file in files if file[-1] == filetype]

    df_files = pd.DataFrame(files, columns=header)
    return df_files


# -- Load operations --------------------------------------------------------------------------------------------------

def load_config(config):
    """Load the config into a dictionary.

    :param config: the config file

    :return: the config loaded into a dictionary
    """

    return {
        # Data preparation settings
        'dataset': config.dataset,
        'miss_rate': config.miss_rate,
        'miss_modality': config.miss_modality,
        'seed': config.seed,
        'store_prepared_dataset': config.store_prepared_dataset,

        # S-GAIN settings
        'version': config.version,
        'batch_size': config.batch_size,
        'hint_rate': config.hint_rate,
        'alpha': config.alpha,
        'iterations': config.iterations,
        'clipping': config.clipping,

        # Generator settings
        'generator_sparsity': config.generator_sparsity,
        'generator_initialization': config.generator_initialization,
        'generator_pruner': config.generator_pruner,
        'generator_prune_rate': config.generator_prune_rate,
        'generator_prune_period': config.generator_prune_period,
        'generator_regrower': config.generator_regrower,
        'generator_regrow_rate': config.generator_regrow_rate,
        'generator_regrow_period': config.generator_regrow_period,
        'generator_strategy': config.generator_strategy,
        'generator_use_strategy': config.generator_use_strategy,

        # Discriminator settings
        'discriminator_sparsity': config.discriminator_sparsity,
        'discriminator_initialization': config.discriminator_initialization,
        'discriminator_pruner': config.discriminator_pruner,
        'discriminator_prune_rate': config.discriminator_prune_rate,
        'discriminator_prune_period': config.discriminator_prune_period,
        'discriminator_regrower': config.discriminator_regrower,
        'discriminator_regrow_rate': config.discriminator_regrow_rate,
        'discriminator_regrow_period': config.discriminator_regrow_period,
        'discriminator_strategy': config.discriminator_strategy,
        'discriminator_use_strategy': config.discriminator_use_strategy,

        # Output settings
        'output_folder': config.output_folder,
        'no_imputation': config.no_imputation,
        'no_log': config.no_log,
        'no_graphs': config.no_graphs,
        'no_model': config.no_model,

        # Monitor settings
        'enable_rmse_monitor': config.enable_rmse_monitor,
        'enable_imputation_time_monitor': config.enable_imputation_time_monitor,
        'enable_memory_usage_monitor': config.enable_memory_usage_monitor,
        'enable_energy_consumption_monitor': config.enable_energy_consumption_monitor,
        'enable_sparsity_monitor': config.enable_sparsity_monitor,
        'enable_FLOPs_monitor': config.enable_FLOPs_monitor,
        'enable_loss_monitor': config.enable_loss_monitor,

        # Run settings
        'n_runs': config.n_runs,
        'retry_failed_experiments': config.retry_failed_experiments,
        'max_failed_experiments': config.max_failed_experiments,
        'ignore_existing_files': config.ignore_existing_files,
        'perform_analysis': config.perform_analysis,

        # Analysis settings
        'analysis_folder': config.analysis_folder,
        'compile_metrics': config.compile_metrics,
        'plot_rmse': config.plot_rmse,
        'plot_success_rate': config.plot_success_rate,
        'plot_imputation_time': config.plot_imputation_time,
        'plot_memory_usage': config.plot_memory_usage,
        'plot_energy_consumption': config.plot_energy_consumption
    }


def get_experiments_from_config(config):
    """Get a Pandas DataFrame with the config of each experiment.

    :param config: the configuration file

    :return:
    - experiments: a Pandas DataFrame with the config of each experiment
    """

    def update_experiments(exps):
        """Update the experiments dictionary.

        Todo implement strategy and use_strategy

        :param exps: a dictionary of the experiments to run
        """

        exps.update({
            (
                # Data preparation
                standardize_dataset(D), MR, standardize_miss_modality(MM), S, cfg['store_prepared_dataset'],

                # S-GAIN
                standardize_version(V), BS, HR, a, i, cfg['clipping'],

                # Generator
                standardize_init(GI, GSp)[0], standardize_init(GI, GSp)[1], standardize_pruner(GP), GPR, GPP,
                standardize_regrower(GR), GRR, GRP,

                # Discriminator
                standardize_init(DI, DSp)[0], standardize_init(DI, DSp)[1], standardize_pruner(DP), DPR, DPP,
                standardize_regrower(DR), DRR, DRP,

                # Output
                cfg['output_folder'], cfg['no_imputation'], cfg['no_log'], cfg['no_graphs'], cfg['no_model'],

                # Monitor
                cfg['enable_rmse_monitor'], cfg['enable_imputation_time_monitor'], cfg['enable_memory_usage_monitor'],
                cfg['enable_energy_consumption_monitor'], cfg['enable_sparsity_monitor'], cfg['enable_FLOPs_monitor'],
                cfg['enable_loss_monitor']
            ): (
                # Run
                cfg['n_runs'], cfg['retry_failed_experiments'], cfg['max_failed_experiments'],
                cfg['ignore_existing_files'],

                # Analysis
                cfg['analysis_folder'], cfg['perform_analysis'], cfg['compile_metrics'], cfg['plot_rmse'],
                cfg['plot_success_rate'], cfg['plot_imputation_time'], cfg['plot_memory_usage'],
                cfg['plot_energy_consumption']
            )

            # Data preparation
            for D in cfg['dataset'] for MR in cfg['miss_rate'] for MM in cfg['miss_modality'] for S in cfg['seed']

            # S-GAIN
            for V in cfg['version'] for BS in cfg['batch_size'] for HR in cfg['hint_rate'] for a in cfg['alpha']
            for i in cfg['iterations']

            # Generator
            for GI in cfg['generator_initialization'] for GSp in cfg['generator_sparsity']
            for GP in cfg['generator_pruner'] for GPR in cfg['generator_prune_rate']
            for GPP in cfg['generator_prune_period'] for GR in cfg['generator_regrower']
            for GRR in cfg['generator_regrow_rate'] for GRP in cfg['generator_regrow_period']

            # Discriminator
            for DI in cfg['discriminator_initialization'] for DSp in cfg['discriminator_sparsity']
            for DP in cfg['discriminator_pruner'] for DPR in cfg['discriminator_prune_rate']
            for DPP in cfg['discriminator_prune_period'] for DR in cfg['discriminator_regrower']
            for DRR in cfg['discriminator_regrow_rate'] for DRP in cfg['discriminator_regrow_period']
        })

    # Get the experiments
    experiments = {}
    cfg = load_config(config)
    update_experiments(experiments)

    # Inclusions
    for inclusion in config.inclusions:
        cfg = load_config(config)
        for key, value in inclusion.items():
            cfg[key] = value
        update_experiments(experiments)

    # Convert to DataFrame
    columns = [
        # Data preparation
        'dataset', 'miss_rate', 'miss_modality', 'seed', 'store_prepared_dataset',

        # S-GAIN
        'version', 'batch_size', 'hint_rate', 'alpha', 'iterations', 'clipping',

        # Generator
        'generator_initialization', 'generator_sparsity', 'generator_pruner', 'generator_prune_rate',
        'generator_prune_period', 'generator_regrower', 'generator_regrow_rate', 'generator_regrow_period',

        # Discriminator
        'discriminator_initialization', 'discriminator_sparsity', 'discriminator_pruner', 'discriminator_prune_rate',
        'discriminator_prune_period', 'discriminator_regrower', 'discriminator_regrow_rate',
        'discriminator_regrow_period',

        # Output
        'output_folder', 'no_imputation', 'no_log', 'no_graphs', 'no_model',

        # Monitor
        'enable_rmse_monitor', 'enable_imputation_time_monitor', 'enable_memory_usage_monitor',
        'enable_energy_consumption_monitor', 'enable_sparsity_monitor', 'enable_FLOPs_monitor', 'enable_loss_monitor',

        # Run
        'n_runs', 'retry_failed_experiments', 'max_failed_experiments', 'ignore_existing_files',

        # Analysis
        'analysis_folder', 'perform_analysis', 'compile_metrics', 'plot_rmse', 'plot_success_rate',
        'plot_imputation_time', 'plot_memory_usage', 'plot_energy_consumption'
    ]
    lst = [k + v for k, v in experiments.items()]
    experiments = pd.DataFrame(lst, columns=columns)

    # Exclusions
    for exclusion in config.exclusions:
        excls = experiments
        for key, value in exclusion.items():
            if isinstance(value, list):
                excls = excls.loc[excls[key].isin(standardize(key, value))]
            else:
                excls = excls.loc[excls[key] == standardize(key, value)]
        experiments.drop(index=excls.index, inplace=True)

    return experiments


def get_completed_experiments(folder):
    """Get a Pandas DataFrame with the completed experiments

    :param folder: the folder the experiments are saved in

    :return:
    - exps: a Pandas DataFrame with the completed experiments
    """

    # Get completed experiments
    exps = parse_files(filepath=folder)
    exps.drop('filetype', axis=1, inplace=True)  # ignore filetype
    exps.drop_duplicates(inplace=True)  # remove duplicates
    exps.drop(['index'], axis=1, inplace=True)  # ignore index

    # Get successes and failures
    exps = exps.groupby(exps.columns.values.tolist()[:-1], as_index=False).agg(['count', 'size'])
    exps.columns = exps.columns.get_level_values(0) + exps.columns.get_level_values(1)
    exps.rename(columns={'rmsecount': 'successes', 'rmsesize': 'failures'}, inplace=True)
    exps['failures'] = exps['failures'] - exps['successes']

    return exps


def read_bin(filepath):
    """Read a (temporary) binary file.

    :param filepath: the filepath

    :return:
    - data: the unpacked data from the file
    """

    # Read binary data
    with open(filepath, 'rb') as f:
        data = f.read()

    # Unpack the data
    fmt = '<%df' % (len(data) // 4)
    data = list(struct.unpack(fmt, data))

    return data


# -- Store operations -------------------------------------------------------------------------------------------------

def get_filepaths(directory, experiment, rmse):
    """Create the necessary directory and return the appropriate filepaths.

    :param directory: the directory to save to
    :param experiment: the name of the experiment
    :param rmse: the Root Mean Squared Error

    :return:
    - filepath_imputed_data: the filepath for the imputed data
    - filepath_log: the filepath for the log
    - filepath_model: the filepath for the (trained) model
    - filepath_graphs: the filepath for the graphs
    """

    if not isdir(directory): makedirs(directory)
    temp_filepath = f'{directory}/{experiment}_RMSE_{rmse}'

    # Avoid overwriting if RMSE is the same
    if (isfile(f'{temp_filepath}.csv')
            or isfile(f'{temp_filepath}_log.json')
            or isfile(f'{temp_filepath}_model.json')
            or isfile(f'{temp_filepath}_graphs.png')
    ):
        i = 1
        while (isfile(f'{temp_filepath}_{i}.csv')
               or isfile(f'{temp_filepath}_{i}_log.json')
               or isfile(f'{temp_filepath}_{i}_model.json')
               or isfile(f'{temp_filepath}_{i}_graphs.png')
        ): i += 1
        temp_filepath = f'{temp_filepath}_{i}'

    filepath_imputed_data = f'{temp_filepath}.csv'
    filepath_log = f'{temp_filepath}_log.json'
    filepath_model = f'{temp_filepath}_model.json'
    filepath_graphs = f'{temp_filepath}_graphs.png'

    return filepath_imputed_data, filepath_log, filepath_model, filepath_graphs


def save_imputation(filepath, imputed_data_x):
    """Save the imputed data to a CSV file.

    :param filepath: the filepath to save to
    :param imputed_data_x: the imputed data
    """

    # Save the imputation
    imputed_data_x = pd.DataFrame(imputed_data_x)
    imputed_data_x.to_csv(filepath, header=False, index=False)


def save_logs(filepath, experiment=None, sys_info=None):
    """Compile and save the logs to a json file.

    :param filepath: the filepath to save the logs to
    :param experiment: the name of the experiment
    :param sys_info: the system information

    Todo check if file exists

    :return:
    - RMSE: the RMSE log
    - imputation_time: the imputation time log
    - memory_usage: the memory usage log
    - energy_consumption: the energy consumption log
    - sparsity: the sparsity log (total)
    - sparsity_G: the sparsity log for the generator
    - sparsity_G_W1: the sparsity log for the first layer of the generator
    - sparsity_G_W2: the sparsity log for the second layer of the generator
    - sparsity_G_W3: the sparsity log for the third layer of the generator
    - sparsity_D: the sparsity log for the discriminator
    - sparsity_D_W1: the sparsity log for the first layer of the discriminator
    - sparsity_D_W2: the sparsity log for the second layer of the discriminator
    - sparsity_D_W3: the sparsity log for the third layer of the discriminator
    - FLOPs: the FLOPs log (total)
    - FLOPs_G: the FLOPs log for the generator
    - FLOPs_D: the FLOPs log for the discriminator
    - loss_G: the loss log for the generator (cross entropy)
    - loss_D: the loss log for the discriminator (cross entropy)
    - loss_MSE: the loss log (MSE)
    - exp: a dictionary containing the experiment
    """

    # Filepaths
    fp_RMSE = 'temp/exp_bins/rmse.bin'
    fp_impution_time = 'temp/exp_bins/imputation_time.bin'
    fp_memory_usage = 'temp/exp_bins/memory_usage.bin'
    fp_energy_consumption = 'temp/exp_bins/energy_consumption.bin'
    fp_sparsity_G = 'temp/exp_bins/sparsity_G.bin'
    fp_sparsity_G_W1 = 'temp/exp_bins/sparsity_G_W1.bin'
    fp_sparsity_G_W2 = 'temp/exp_bins/sparsity_G_W2.bin'
    fp_sparsity_G_W3 = 'temp/exp_bins/sparsity_G_W3.bin'
    fp_sparsity_D = 'temp/exp_bins/sparsity_D.bin'
    fp_sparsity_D_W1 = 'temp/exp_bins/sparsity_D_W1.bin'
    fp_sparsity_D_W2 = 'temp/exp_bins/sparsity_D_W2.bin'
    fp_sparsity_D_W3 = 'temp/exp_bins/sparsity_D_W3.bin'
    fp_FLOPs_G = 'temp/exp_bins/FLOPs_G.bin'
    fp_FLOPs_D = 'temp/exp_bins/FLOPs_D.bin'
    fp_loss_G = 'temp/exp_bins/loss_G.bin'
    fp_loss_D = 'temp/exp_bins/loss_D.bin'
    fp_loss_MSE = 'temp/exp_bins/loss_MSE.bin'

    # Read the log files
    RMSE = read_bin(fp_RMSE) if isfile(fp_RMSE) else None
    imputation_time = read_bin(fp_impution_time) if isfile(fp_impution_time) else None
    memory_usage = read_bin(fp_memory_usage) if isfile(fp_memory_usage) else None
    energy_consumption = read_bin(fp_energy_consumption) if isfile(fp_energy_consumption) else None
    sparsity_G = read_bin(fp_sparsity_G) if isfile(fp_sparsity_G) else None
    sparsity_G_W1 = read_bin(fp_sparsity_G_W1) if isfile(fp_sparsity_G_W1) else None
    sparsity_G_W2 = read_bin(fp_sparsity_G_W2) if isfile(fp_sparsity_G_W2) else None
    sparsity_G_W3 = read_bin(fp_sparsity_G_W3) if isfile(fp_sparsity_G_W3) else None
    sparsity_D = read_bin(fp_sparsity_D) if isfile(fp_sparsity_D) else None
    sparsity_D_W1 = read_bin(fp_sparsity_D_W1) if isfile(fp_sparsity_D_W1) else None
    sparsity_D_W2 = read_bin(fp_sparsity_D_W2) if isfile(fp_sparsity_D_W2) else None
    sparsity_D_W3 = read_bin(fp_sparsity_D_W3) if isfile(fp_sparsity_D_W3) else None
    FLOPs_G = read_bin(fp_FLOPs_G) if isfile(fp_FLOPs_G) else None
    FLOPs_D = read_bin(fp_FLOPs_D) if isfile(fp_FLOPs_D) else None
    loss_G = read_bin(fp_loss_G) if isfile(fp_loss_G) else None
    loss_D = read_bin(fp_loss_D) if isfile(fp_loss_D) else None
    loss_MSE = read_bin(fp_loss_MSE) if isfile(fp_loss_MSE) else None

    # Totals
    sparsity = [(sparsity_G[i] + sparsity_D[i]) / 2 for i in range(len(sparsity_G))] if sparsity_G else None
    FLOPs = [FLOPs_G[i] + FLOPs_D[i] for i in range(len(FLOPs_G))] if FLOPs_G else None

    logs, exp = {}, None
    if experiment is not None:
        dataset, miss_rate, miss_modality, seed, version, batch_size, hint_rate, alpha, iterations, clipping, \
            generator_initialization, generator_sparsity, generator_pruner, generator_prune_rate, \
            generator_prune_period, generator_regrower, generator_regrow_rate, generator_regrow_period, \
            discriminator_initialization, discriminator_sparsity, discriminator_pruner, discriminator_prune_rate, \
            discriminator_prune_period, discriminator_regrower, discriminator_regrow_rate, discriminator_regrow_period \
            = parse_experiment(experiment, file=False)

        exp = {
            'dataset': dataset,
            'miss_rate': miss_rate,
            'miss_modality': miss_modality,
            'seed': seed,
            'version': version,
            'batch_size': batch_size,
            'hint_rate': hint_rate,
            'alpha': alpha,
            'iterations': iterations,
            'clipping': clipping,
            'generator_initialization': generator_initialization,
            'generator_sparsity': generator_sparsity,
            'generator_pruner': generator_pruner,
            'generator_prune_rate': generator_prune_rate,
            'generator_prune_period': generator_prune_period,
            'generator_regrower': generator_regrower,
            'generator_regrow_rate': generator_regrow_rate,
            'generator_regrow_period': generator_regrow_period,
            'discriminator_initialization': discriminator_initialization,
            'discriminator_sparsity': discriminator_sparsity,
            'discriminator_pruner': discriminator_pruner,
            'discriminator_prune_rate': discriminator_prune_rate,
            'discriminator_prune_period': discriminator_prune_period,
            'discriminator_regrower': discriminator_regrower,
            'discriminator_regrow_rate': discriminator_regrow_rate,
            'discriminator_regrow_period': discriminator_regrow_period
        }
        logs.update({'experiment': exp})

    if sys_info: logs.update({'system_information': sys_info})

    # Log variables
    it_total = sum(imputation_time)
    it_preparation = imputation_time[0]
    it_finalization = imputation_time[-1]
    it_s_gain = it_total - it_preparation - it_finalization

    if RMSE: logs.update({
        'rmse': {
            'final': RMSE[-1],
            'log': RMSE,
        }
    })
    if imputation_time: logs.update({
        'imputation_time': {
            'total': it_total,
            'preparation': it_preparation,
            's_gain': it_s_gain,
            'finalization': it_finalization,
            'log': imputation_time
        }
    })
    if memory_usage: logs.update({
        'memory_usage': {
            'maximum': max(memory_usage),
            'average': sum(memory_usage) / len(memory_usage),
            'log': memory_usage
        }
    })
    if energy_consumption: logs.update({
        'energy_consumption': {
            'total': sum(energy_consumption),
            'log': energy_consumption
        }
    })
    if sparsity_G: logs.update({
        'sparsity': {
            'initial': sparsity[0],
            'final': sparsity[-1],
            'minimum': min(sparsity),
            'log': sparsity,
            'generator': {
                'initial': sparsity_G[0],
                'final': sparsity_G[-1],
                'minimum': min(sparsity_G),
                'log': sparsity_G,
                'G_W1': {
                    'initial': sparsity_G_W1[0],
                    'final': sparsity_G_W1[-1],
                    'minimum': min(sparsity_G_W1),
                    'log': sparsity_G_W1
                },
                'G_W2': {
                    'initial': sparsity_G_W2[0],
                    'final': sparsity_G_W2[-1],
                    'minimum': min(sparsity_G_W2),
                    'log': sparsity_G_W2
                },
                'G_W3': {
                    'initial': sparsity_G_W3[0],
                    'final': sparsity_G_W3[-1],
                    'minimum': min(sparsity_G_W3),
                    'log': sparsity_G_W3
                }
            },
            'discriminator': {
                'initial': sparsity_D[0],
                'final': sparsity_D[-1],
                'minimum': min(sparsity_D),
                'log': sparsity_D,
                'D_W1': {
                    'initial': sparsity_D_W1[0],
                    'final': sparsity_D_W1[-1],
                    'minimum': min(sparsity_D_W1),
                    'log': sparsity_D_W1
                },
                'D_W2': {
                    'initial': sparsity_D_W2[0],
                    'final': sparsity_D_W2[-1],
                    'minimum': min(sparsity_D_W2),
                    'log': sparsity_D_W2
                },
                'D_W3': {
                    'initial': sparsity_D_W3[0],
                    'final': sparsity_D_W3[-1],
                    'minimum': min(sparsity_D_W3),
                    'log': sparsity_D_W3
                }
            }
        }
    })
    if FLOPs: logs.update({
        'FLOPs': {
            'total': sum(FLOPs),
            'log': FLOPs,
            'generator': {
                'total': sum(FLOPs_G),
                'log': FLOPs_G
            },
            'discriminator': {
                'total': sum(FLOPs_D),
                'log': FLOPs_D
            }
        }
    })
    if loss_G: logs.update({
        'loss': {
            'cross_entropy': {
                'generator': {
                    'initial': loss_G[0],
                    'total': loss_G[-1],
                    'log': loss_G
                },
                'discriminator': {
                    'initial': loss_D[0],
                    'final': loss_D[-1],
                    'log': loss_D
                }
            },
            'MSE': {
                'initial': loss_MSE[0],
                'final': loss_MSE[-1],
                'log': loss_MSE
            }
        }
    })

    with open(filepath, 'w') as f:
        f.write(json.dumps(logs))

    return logs, exp


# -- Other functions --------------------------------------------------------------------------------------------------

def system_information(print_ready=False):
    """Get the system information.

    :param print_ready: return a list of print ready strings instead of a dictionary

    :return:
    - sys_info: the system information
    """

    # Parameters
    filepath = 'temp/sys_info.json'

    # Load system information
    if isfile(filepath):
        with open(filepath, 'r') as f:
            sys_info = json.load(f)

    # Get and store system information
    else:
        sys_info = {
            'platform': platform.system(),
            'version': platform.version(),
            'cpu': cpuinfo.get_cpu_info()['brand_raw'],
            'memory': f'{psutil.virtual_memory().total / (1024 ** 3):.1f} GB'
        }

        # Todo multiple gpu support (log the one used for running the experiment only)
        #  and more memory detail (speed, timings or identifier)
        if sys_info['platform'] == 'Linux':
            # Todo Get GPU
            sys_info['gpu'] = 'unable to identify GPU (no support for Linux yet)'

            # Todo Get disk info
            sys_info['disk'] = 'unable to identify disk (no support for Linux yet)'

            # Todo Update OS and log motherboard
            sys_info['motherboard'] = 'unable to identify motherboard (no support for Linux yet)'

        elif sys_info['platform'] == 'Windows':
            # Get GPU
            sys_info['gpu'] = wmi.WMI().Win32_VideoController()[0].name

            # Todo Get disk info
            sys_info['disk'] = 'unable to identify disk (no support for Windows yet)'

            # Update OS and log motherboard
            for x in subprocess.check_output(['systeminfo']).decode('utf-8').split('\n'):
                if x.startswith('OS Name'):
                    sys_info['version'] = x.split(':')[1].strip()
                elif x.startswith('OS Version'):
                    sys_info['version'] += f' {x.split(":")[1].strip()}'
                elif x.startswith('System Manufacturer'):
                    sys_info['motherboard'] = x.split(':')[1].strip()
                elif x.startswith('System Model'):
                    sys_info['motherboard'] += f' {x.split(":")[1].strip()}'
                    break

        elif sys_info['platform'] == 'Darwin':
            # Todo Get GPU
            sys_info['gpu'] = 'unable to identify GPU (no support for Mac OS yet)'

            # Todo Get disk info
            sys_info['disk'] = 'unable to identify disk (no support for Mac OS yet)'

            # Todo Update OS and log motherboard
            sys_info['motherboard'] = 'unable to identify motherboard (no support for Mac OS yet)'

        else:
            # Log GPU, disk and motherboard
            sys_info['gpu'] = f'unable to identify GPU ({sys_info["platform"]} unsupported)'
            sys_info['disk'] = f'unable to identify disk ({sys_info["platform"]} unsupported)'
            sys_info['motherboard'] = f'unable to identify motherboard ({sys_info["platform"]} unsupported)'

        # Store the system information
        if not isdir('temp'): makedirs('temp')
        with open(filepath, 'w') as f:
            f.write(json.dumps(sys_info))

    # Convert dictionary to print ready strings
    if print_ready:
        info = 'System information'
        version = f'OS: {sys_info["version"]}'
        cpu = f'CPU: {sys_info["cpu"]}'
        memory = f'Memory: {sys_info["memory"]}'
        gpu = f'GPU: {sys_info["gpu"]}'
        disk = f'Disk: {sys_info["disk"]}'
        motherboard = f'Motherboard: {sys_info["motherboard"]}'
        sys_info = [info, version, cpu, memory, gpu, disk, motherboard]

    return sys_info
