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

"""Data preparation functions.

Data loader:
(1) dataloader: load a dataset and the header.

Data preparation:
(2) dataprep: load a dataset and introduce missing values.
"""

import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo
from os import makedirs
from os.path import isfile, isdir

from dataprep.distributions import MCAR
from utils.standardizers import standardize_dataset, standardize_distribution


# -- Data loader ------------------------------------------------------------------------------------------------------

def dataloader(dataset, labels=False, store=False, verbose=False):
    """Load a dataset and the header.

    Args:
        dataset: the dataset to use.
        labels: whether to include the labels for imputation or not.
        store: whether to store the downloaded dataset.
        verbose: enable verbose output to the console.

    Returns:
        x_data: the data.
        header: the header of the dataset.
    """

    # Standardization
    dataset = standardize_dataset(dataset)
    filepath = f'datasets/{dataset}.csv'

    if verbose: print(f'\nLoading the dataset...')

    # -- Tabular datasets (UCI repository) ----------------------------------------------------------------------------

    if dataset in ['MHR', 'BCWD', 'DCCC', 'ONP', 'LR', 'SB']:

        def uci_id(d):
            """Returns the id of the dataset in the UCI repository, the index of the classes,
            and the indexes of non-predictive features.
            """
            if d == 'MHR': return 863, -1, None
            if d == 'BCWD': return 17, -1, 0
            if d == 'DCCC': return 350, -1, 0
            if d == 'ONP': return 332, -1, [0, 1]
            if d == 'LR': return 59, 0, None
            if d == 'SB': return 94, -1, None
            return None  # This should not happen

        # Load the dataset
        if isfile(filepath):
            if verbose: print('Loading from disk...')
            x_data = pd.read_csv(filepath, delimiter=',')
        else:
            if verbose: print('Downloading from the UCI repository...')
            uci_dataset = fetch_ucirepo(id=uci_id(dataset)[0])
            x_data = uci_dataset.data.original

            if store:  # Store the dataset
                if verbose: print('Storing the dataset on the disk...')
                x_data.to_csv(filepath, index=False)

        # Exclude non-predictive features
        non_predictive = uci_id(dataset)[2]
        if non_predictive is not None: x_data.drop(columns=x_data.columns[non_predictive], inplace=True)

        # Exclude the labels
        if not labels: x_data.drop(columns=x_data.columns[uci_id(dataset)[1]], inplace=True)

        # Get the header and convert to numpy array
        header = x_data.columns.tolist()
        x_data = x_data.values

    # -- Image datasets (Keras imports) -------------------------------------------------------------------------------

    elif dataset in ['MNIST', 'Fashion_MNIST', 'CIFAR10']:

        if isfile(filepath):
            # Load the dataset
            if verbose: print('Loading from disk...')
            x_data = pd.read_csv(filepath, delimiter=',')

            # Exclude the labels
            if not labels: x_data.drop(columns=x_data.columns[-1], inplace=True)

            # Get the header and convert to numpy array
            header = x_data.columns.tolist()
            x_data = x_data.values

        else:
            # Import the dataset from Keras
            if verbose: print('Importing from Keras...')
            if dataset == 'MNIST':
                from keras.datasets import mnist as kd
                shape = [60000, 28 * 28 * 1]
            elif dataset == 'Fashion_MNIST':
                from keras.datasets import fashion_mnist as kd
                shape = [60000, 28 * 28 * 1]
            else:  # CIFAR10
                from keras.datasets import cifar10 as kd
                shape = [50000, 32 * 32 * 3]

            # Load the dataset
            (x_data, x_labels), _ = kd.load_data()
            x_data = np.reshape(np.asarray(x_data), shape).astype(float)
            header = list(range(shape[1]))

            # Include labels
            if labels:
                x_labels = np.reshape(np.asarray(x_labels), [shape[0], 1]).astype(float)
                x_data = np.append(x_data, x_labels, axis=1)
                header += ['class']

            # Store the dataset
            if store:
                if verbose: print('Storing the dataset on the disk...')

                if labels:
                    temp_x_data = x_data
                    temp_header = header
                else:
                    x_labels = np.reshape(np.asarray(x_labels), [shape[0], 1]).astype(float)
                    temp_x_data = np.append(x_data, x_labels, axis=1)
                    temp_header = header + ['class']

                temp_x_data = pd.DataFrame(temp_x_data, columns=temp_header)
                temp_x_data.to_csv(filepath, index=False)

    # -- Invalid datasets ---------------------------------------------------------------------------------------------

    else:  # This should not happen
        print(f'Invalid dataset: {dataset}.\nExiting the program...')
        return None

    # -----------------------------------------------------------------------------------------------------------------

    if verbose: print('Finished loading the dataset.')
    return x_data, header


# -- Data preparation -------------------------------------------------------------------------------------------------

def dataprep(dataset, prob, dist, seed=None, labels=False, prepdata_folder='datasets/prepared', store=False,
             dictionary=False, verbose=False):
    """Load a dataset and introduce missing values:
    (1) Load a dataset.
    (2) Introduce missing values according to the specified distribution and probability (optional).
    (3) Remove header.
    (4) Remove labels (optional).
    (5) Return list or dictionary.

    Args:
        dataset: the dataset to use.
        prob: the probability of the missing values in the dataset.
        dist: the distribution of the missing values in the dataset [MCAR, MAR, MNAR].
        seed: the random seed used introduce the missing values in the dataset.
        labels: whether to include the labels for imputation or not.
        prepdata_folder: the folder to store the prepared datasets in.
        store: whether to store the downloaded/prepared datasets.
        dictionary: the format of the dataset [DataFrame, dictionary, list].
        verbose: enable verbose output to the console.

    Returns:
        x_data: the original dataset.
        x_miss: the data with missing values
        mask: the indicator matrix for missing values.
    """

    # Standardization
    dataset = standardize_dataset(dataset)
    dist = standardize_distribution(dist)
    if seed is None: seed = np.random.randint(2 ** 31)
    filename = f'{dataset}#{prob}#{dist}#{seed:08x}#{int(labels)}'

    # Output to console
    if verbose:
        if verbose != 'main': print(f'\n{filename}')

    # Load the dataset TODO (immediately return if it already has missing values)
    x_data, header = dataloader(dataset, labels)
    if dataset == 'PSCD': return x_data, header

    # Try to load a prepared dataset
    prepared_dataset = f'{prepdata_folder}/{filename}.csv'
    if isfile(prepared_dataset):
        if verbose: print('Loading the prepared dataset from disk...')
        x_miss = np.loadtxt(prepared_dataset, delimiter=',')
        mask = np.ones(x_miss.shape, dtype=int)
        mask[np.isnan(x_miss)] = 0
        if verbose: print('Finished loading the prepared dataset.')

    elif dist == 'MCAR':
        if verbose: print(f'Introducing {prob} missing values MCAR...')
        x_miss, mask = MCAR(x_data, prob, seed)

    # TODO MAR MNAR
    elif dist in ['MAR', 'MNAR']:
        # if verbose: print(f'Introducing {prob} missing values {dist}...')
        print(f'Distribution: {dist} is not implemented yet.\nExiting the program...')
        return None

    else:  # This should not happen
        print(f'Invalid distribution: {dist}.\nExiting the program...')
        return None

    # Store prepared dataset
    if store:
        if verbose: print('Storing the prepared dataset...')
        if not isdir(prepdata_folder): makedirs(prepdata_folder)
        np.savetxt(prepared_dataset, x_miss, delimiter=',')

    # Return dictionary
    if dictionary:
        return {
            'x_data': x_data,
            'x_miss': x_miss,
            'mask': mask,
            'header': header,
            'labels': labels
        }

    return x_data, x_miss, mask, header
