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

"""Utility functions for S-GAIN:

Samplers:
(1) uniform_sampler: sample uniform random variables
(2) binary_sampler: sample binary random variables

Other functions:
(3) sample_batch_index: sample index of the mini-batch
(4) normalization: normalize the data in [0, 1] range
(5) renormalization: re-normalize data from [0, 1] range to the original range
(6) rounding: round the imputed data for categorical variables
"""

import numpy as np


# -- Samplers ---------------------------------------------------------------------------------------------------------

def uniform_sampler(low, high, rows, cols, seed=None):
    """Sample uniform random variables.

    :param low: the low limit
    :param high: the high limit
    :param rows: the number of rows
    :param cols: the number of columns
    :param seed: the random seed

    :return:
    - uniform_random_matrix: a uniform random matrix
    """

    # Fix seed for run-to-run consistency
    if seed is not None: np.random.seed(seed)

    uniform_random_matrix = np.random.uniform(low, high, size=(rows, cols))
    return uniform_random_matrix


def binary_sampler(p, rows, cols, seed=None):
    """Sample binary random variables.

    :param p: the probability of 1
    :param rows: the number of rows
    :param cols: the number of columns
    :param seed: the random seed

    :return:
    - binary_random_matrix: a binary random matrix
    """

    uniform_random_matrix = uniform_sampler(0., 1., rows, cols, seed)
    binary_random_matrix = 1 * (uniform_random_matrix < p)
    return binary_random_matrix


def MAR3(data_x, prob, rows, cols, seed=None):
    """Sample variables distributed Missing at Random (MAR).

    This method uses the formula from the supplementary materials of:
    J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data Imputation using Generative Adversarial Nets", ICML,
    2018. https://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf.
    And was implemented by: Adam Bosch, Roman Ladus, and Vlad Negara.

    :param data_x: the original dataset.
    :param prob: the probability of the missing values.
    :param rows: the number of rows (entries).
    :param cols: the number of columns (features).
    :param seed: the random seed.

    :return:
    - miss_data_x: the data with missing values distributed MAR.
    - data_mask: the indicator matrix for missing values distributed MAR.
    """

    # Fix the seed for run-to-run consistency
    if seed: np.random.seed(seed)

    # Uniform p_m
    p_m = np.full((cols,), prob)

    # Array to memoize sums in exponents in the formula. Cell [n][i] holds the sum over j<i
    exponent_terms = np.zeros(shape=(rows, cols + 1))

    # Array to memoize the denominator in the formula
    denominators = np.zeros(shape=(cols + 1,))

    # The first denominator is always equal to rows
    denominators[0] = rows

    # Initialize random weights with the U(0,1) distribution
    w = np.random.uniform(0., 1., size=cols)

    # Initialize random biases with the U(0,1) distribution
    b = np.random.uniform(0., 1., size=cols)

    # Initialize the mask and the data with missingness
    data_mask = np.ones(shape=(rows, cols))
    miss_data_x = data_x.copy()

    # Normalize data using min-max scaling
    data_x_min = data_x.min(axis=0)
    data_x_max = data_x.max(axis=0)
    data_x_normalized = (data_x.copy() - data_x_min) / (data_x_max - data_x_min)

    # Iterate over the features, then the rows
    for i in range(cols):
        for n in range(rows):
            # Extract the memoized exponent in the numerator of the formula
            numerator_exponent = exponent_terms[n][i]

            # Extract the memoized denominator of the formula
            denominator = denominators[i]

            # Compute the probability of missingness using the formula
            P = p_m[i] * rows * np.exp(-numerator_exponent) / denominator

            # Generate a random value between 0 and 1 to check against the probability
            uniform_random_value = np.random.uniform()
            if uniform_random_value < P:
                # The value is missing
                data_mask[n][i] = 0
                miss_data_x[n][i] = np.nan

                # Add the bias of this feature to the memorized numerator exponent for the next feature
                exponent_terms[n][i + 1] = exponent_terms[n][i] + b[i]
            else:
                # Add the weighted value of this feature to the memorized numerator exponent for the next feature
                exponent_terms[n][i + 1] = exponent_terms[n][i] + w[i] * data_x_normalized[n][i]

            # Add the numerator exponent for the next feature to its memorized denominator
            denominators[i + 1] += np.exp(-exponent_terms[n][i + 1])

    return miss_data_x, data_mask


def MNAR3(data_x, prob, rows, cols, seed=None):
    """Sample variables distributed Missing not at Random (MNAR).

    This method uses the formula from the supplementary materials of:
    J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data Imputation using Generative Adversarial Nets", ICML,
    2018. https://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
    And was implemented by: Adam Bosch, Roman Ladus, and Vlad Negara.

    :param data_x: the original dataset.
    :param prob: the probability of the missing values.
    :param rows: the number of rows (entries).
    :param cols: the number of columns (features).
    :param seed: the random seed.

    :return:
    - miss_data_x: the data with missing values distributed MAR.
    - data_mask: the indicator matrix for missing values distributed MAR.
    """

    # Fix the seed for run-to-run consistency
    if seed: np.random.seed(seed)

    # Uniform p_m
    p_m = np.full((cols,), prob)

    # Initialize random weights with the U(0,1) distribution
    w = np.random.uniform(0., 1., size=cols)

    # Normalize data using min-max scaling
    data_x_min = data_x.min(axis=0)
    data_x_max = data_x.max(axis=0)
    data_x_normalized = (data_x.copy() - data_x_min) / (data_x_max - data_x_min)

    # Array to memoize the denominator in the formula
    denominators = np.zeros(shape=(cols,))
    for i in range(cols):
        for n in range(rows):
            denominators[i] += np.exp(-w[i] * data_x_normalized[n][i])

    # Initialize the mask and the data with missingness
    data_mask = np.ones(shape=(rows, cols))
    miss_data_x = data_x.copy()

    # Iterate over the features, then the rows
    for i in range(cols):
        for n in range(rows):
            # Extract the memoized denominator of the formula
            denominator = denominators[i]

            # Compute the probability of missingness using the formula
            P = p_m[i] * rows * np.exp(-w[i] * data_x_normalized[n][i]) / denominator

            # Generate a random value between 0 and 1 to check against the
            # probability
            uniform_random_value = np.random.uniform()
            if uniform_random_value < P:
                # The value is missing
                data_mask[n][i] = 0
                miss_data_x[n][i] = np.nan

    return miss_data_x, data_mask


# -- Other functions --------------------------------------------------------------------------------------------------

def sample_batch_index(total, batch_size):
    """Sample index of the mini-batch.

    :param total: the total number of samples
    :param batch_size: the batch size

    Returns:
    - batch_idx: the batch index
    """

    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx


def normalization(data_x, norm_parameters=None):
    """Normalize the data in [0, 1] range.

    :param data_x: the original data

    :return:
    - norm_data_x: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
    """

    # Parameters
    _, dim = data_x.shape
    norm_data_x = data_x.copy()

    if norm_parameters is None:
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        for i in range(dim):  # Todo: run on GPU?
            min_val[i] = np.nanmin(norm_data_x[:, i])
            norm_data_x[:, i] = norm_data_x[:, i] - np.nanmin(norm_data_x[:, i])
            max_val[i] = np.nanmax(norm_data_x[:, i])
            norm_data_x[:, i] = norm_data_x[:, i] / (np.nanmax(norm_data_x[:, i]) + 1e-7)

        norm_parameters = {'min_val': min_val, 'max_val': max_val}

    else:
        min_val = norm_parameters['min_val']
        max_val = norm_parameters['max_val']

        for i in range(dim):  # Todo: run on GPU?
            norm_data_x[:, i] = norm_data_x[:, i] - min_val[i]
            norm_data_x[:, i] = norm_data_x[:, i] / (max_val[i] + 1e-7)

    return norm_data_x, norm_parameters


def renormalization(norm_data_x, norm_parameters):
    """Re-normalize data from [0, 1] range to the original range.

    :param norm_data_x: the normalized data
    :param norm_parameters: the min_val and max_val for each feature for renormalization

    :returns:
    - renorm_data_x: the re-normalized data
    """

    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']

    _, dim = norm_data_x.shape
    renorm_data_x = norm_data_x.copy()

    for i in range(dim):  # Todo: run on GPU?
        renorm_data_x[:, i] = renorm_data_x[:, i] * (max_val[i] + 1e-7)
        renorm_data_x[:, i] = renorm_data_x[:, i] + min_val[i]

    return renorm_data_x


def rounding(imputed_data_x, miss_data_x):
    """Round the imputed data for categorical variables.

    :param imputed_data_x: the imputed data
    :param miss_data_x: the data with missing values

    Returns:
    - rounded_data_x: the rounded data
    """

    _, dim = miss_data_x.shape
    rounded_data_x = imputed_data_x.copy()

    for i in range(dim):  # Todo: run on GPU?
        temp = miss_data_x[~np.isnan(miss_data_x[:, i]), i]

        # Only for the categorical variables
        if len(np.unique(temp)) < 20:
            rounded_data_x[:, i] = np.round(rounded_data_x[:, i])

    return rounded_data_x
