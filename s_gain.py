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

"""The main interface to interact with the S-GAIN testing framework."""

import argparse

import config

import utils.subroutines as subroutines


# -- Main -------------------------------------------------------------------------------------------------------------

def main(args):
    """The main interface to interact with S-GAIN."""

    # Select subroutine
    subroutine = args.subroutine
    if subroutine == 'settings':
        subroutines.settings(settings, args.operation, args.filename, args.information)
    elif subroutine == 'prepare':
        subroutines.prepare_datasets(config, args.dataset, args.miss_rate, args.miss_modality, args.seed)
    elif subroutine == 'run':
        subroutines.run_experiments(config, args.show)
    elif subroutine == 'analyze':
        subroutines.analyze(config, args.input, args.output)
    else:
        parser.print_help()


if __name__ == '__main__':
    # Parser and subparsers
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest='subroutine',
        title='subroutine',
        help='change settings, run or analyze experiments'
    )

    # Settings Todo overwrite config
    settings = subparsers.add_parser(
        'settings',
        help='show, load, store or delete settings'
    )
    settings.add_argument(
        'operation',
        choices=['show', 'list', 'ls', 'load', 'l', 'store', 'save', 's', 'delete', 'del', 'remove', 'rm'],
        help='show settings, load settings (default, IDEAL2025, showcase, ...), store current settings or delete settings',
        type=str
    )
    settings.add_argument(
        'filename',
        help='the name of the settings file to show, load, store or delete (shows the current settings if left blank)',
        nargs='?',
        type=str
    )
    settings.add_argument(
        '--information', '-info',
        help='show additional information about the settings',
        action='store_true'
    )

    # Prepare datasets
    prepare = subparsers.add_parser(
        'prepare',
        help='prepare the datasets specified in config.py'
    )
    prepare.add_argument(
        'dataset',
        help='the dataset to prepare (overwrites config)',
        nargs='?',
        type=str
    )
    prepare.add_argument(
        '--datasets', '-ds',
        help='the datasets to prepare (overwrites config)',
        nargs='+',
        type=str
    )
    prepare.add_argument(
        'miss_rate',
        help='the miss rate to prepare the datasets with (overwrites config)',
        nargs='?',
        type=float
    )
    prepare.add_argument(
        '--miss_rate', '--miss_rates', '-mr',
        help='the miss rate to prepare the datasets with (overwrites config)',
        nargs='+',
        type=float
    )
    prepare.add_argument(
        'miss_modality',
        help='the miss modality to prepare the datasets with (overwrites config)',
        choices=['MCAR', 'MAR', 'MNAR', 'AI_upscaler', 'square'],
        nargs='?',
        type=str
    )
    prepare.add_argument(
        '--miss_modality', '--miss_modalities', '-mm',
        help='the miss modality to prepare the datasets with (overwrites config)',
        choices=['MCAR', 'MAR', 'MNAR', 'AI_upscaler', 'square'],
        nargs='+',
        type=str
    )
    prepare.add_argument(
        'seed',
        help='the seeds to prepare the datasets with (overwrites config)',
        nargs='?',
        type=int
    )
    prepare.add_argument(
        '--seed', '--seeds', '-s',
        help='the seeds to prepare the datasets with (overwrites config). random seed if left blank',
        default=[],
        nargs='*',
        type=int
    )

    # Run experiments Todo overwrite config
    run = subparsers.add_parser(
        'run',
        help='run the experiments specified in config.py'
    )
    run.add_argument(
        '--show',
        help='show the experiments to run',
        action='store_true'
    )

    # Analysis Todo overwrite config
    analysis = subparsers.add_parser(
        'analyze',
        help='analyze the completed experiments'
    )
    analysis.add_argument(
        'input',
        help='the folder where the completed experiments are located (use default: output, if not specified)',
        nargs='?',
        type=str
    )
    analysis.add_argument(
        '-in', '--input',
        help='the folder where the completed experiments are located (use default: output, if not specified)',
        type=str
    )
    analysis.add_argument(
        'output',
        help='the folder to save the analysis to (use default: analysis, if not specified)',
        nargs='?',
        type=str
    )
    analysis.add_argument(
        '-out', '--output',
        help='the folder to save the analysis to (use default: analysis, if not specified)',
        type=str
    )

    # Call main
    args = parser.parse_args()
    main(args)
