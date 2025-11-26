"""Dataset loader for S-GAIN:

Data loaders:
(1) data_loader: load a dataset and introduce missing elements
"""

import numpy as np

from os import makedirs
from os.path import isfile, isdir

from utils.standardizers import standardize_dataset, standardize_miss_modality
from utils.utils import binary_sampler


# -- Data loaders -----------------------------------------------------------------------------------------------------

def data_loader(dataset, miss_rate, miss_modality, seed=None, prepared_dataset_folder='datasets/prepared',
                store_prepared_dataset=False, verbose=False):
    """Load a dataset and introduce missing elements.

    Todo: other miss modalities [MAR, MNAR, upscaler, square]

    :param dataset: the dataset to use
    :param miss_rate: the probability of missing elements in the data
    :param miss_modality: the modality of missing data [MCAR, MAR, MNAR, upscaler, square]
    :param seed: the seed used to introduce missing elements in the data
    :param prepared_dataset_folder: the folder to store the prepared datasets in
    :param store_prepared_dataset: whether to store the prepared dataset
    :param verbose: enable verbose output to the console

    :return:
    - data_x: the original data (without missing values)
    - miss_data_x: the data with missing values
    - data_mask: the indicator matrix for missing elements
    """

    # Standardization
    dataset = standardize_dataset(dataset)
    miss_modality = standardize_miss_modality(miss_modality)
    if seed is None: seed = np.random.randint(2 ** 31)
    filename = f'{dataset}_MR_{miss_rate}_MM_{miss_modality}_{seed:08x}'

    if verbose:
        if verbose != 'main': print(f'\n{filename}')
        print(f'\nLoading data...')

    # Load the data
    if dataset in ['health', 'letter', 'spam']:
        file_name = f'datasets/{dataset}.csv'
        data_x = np.loadtxt(file_name, delimiter=',', skiprows=1)
    elif dataset == 'MNIST':
        from keras.datasets import mnist
        (data_x, _), _ = mnist.load_data()
        data_x = np.reshape(np.asarray(data_x), [60000, 28 * 28]).astype(float)
    elif dataset == 'Fashion_MNIST':
        from keras.datasets import fashion_mnist
        (data_x, _), _ = fashion_mnist.load_data()
        data_x = np.reshape(np.asarray(data_x), [60000, 28 * 28]).astype(float)
    elif dataset == 'CIFAR10':
        from keras.datasets import cifar10
        (data_x, _), _ = cifar10.load_data()
        data_x = np.reshape(np.asarray(data_x), [50000, 32 * 32 * 3]).astype(float)
    else:  # This should not happen
        print(f'Invalid dataset: {dataset}.\nExiting the program...')
        return [None] * 3

    # Try to load a prepared dataset
    prepared_dataset = f'{prepared_dataset_folder}/{filename}.csv'
    if isfile(prepared_dataset):
        miss_data_x = np.loadtxt(prepared_dataset, delimiter=',')
        data_mask = np.ones(miss_data_x.shape, dtype=int)
        data_mask[np.isnan(miss_data_x)] = 0

    # Introduce missing elements in the data
    elif miss_modality == 'MCAR':
        no, dim = data_x.shape
        data_mask = binary_sampler(1 - miss_rate, no, dim, seed)
        miss_data_x = data_x.copy()
        miss_data_x[data_mask == 0] = np.nan

    elif miss_modality in ['MAR', 'MNAR', 'upscaler', 'square']:
        print(f'Miss modality: {miss_modality} is not implemented yet.\nExiting the program...')
        return [None] * 3

    else:  # This should not happen
        print(f'Invalid miss modality: {miss_modality}.\nExiting the program...')
        return [None] * 3

    # Store prepared dataset
    if store_prepared_dataset:
        if verbose: print('Storing the prepared dataset...')
        if not isdir(prepared_dataset_folder): makedirs(prepared_dataset_folder)
        np.savetxt(prepared_dataset, miss_data_x, delimiter=',')

    return data_x, miss_data_x, data_mask
