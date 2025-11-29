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

"""The distributions for the dataloader.

TODO unify team 1 and team 3's implementations of MAR and MNAR.

Distributions:
(1) MCAR: Missing Completely at random.
(2) MAR1: Missing at Random (team 1).
(3) MAR3: Missing at Random (team 3).
(4) MNAR1: Missing not at Random (team 1).
(5) MNAR3: Missing not at Random (team 3).
"""

import numpy as np


# -- Distributions ----------------------------------------------------------------------------------------------------

def MCAR(data, prob, seed=None):
    """Sample variables distributed Missing Completely at Random (MCAR).

    Args:
        data: the original dataset.
        prob: the probability of the missing values.

    Returns:
        miss_data: the data with missing values.
        mask: the indicator matrix for missing values.

    """

    # Fix seed for run-to-run consistency
    if seed is not None: np.random.seed(seed)

    # Calculate the MCAR mask
    uniform_random_matrix = np.random.uniform(0., 1., size=data.shape)
    mask = 1 * (uniform_random_matrix >= prob)

    # Introduce missing values
    miss_data = data.copy()
    miss_data[mask == 0] = np.nan

    return miss_data, mask


def MAR1(X_data, prob, rows, cols, seed=None):
    """Sample variables distributed Missing at Random (MAR).

    This method uses the formula from the supplementary materials of:
    J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data Imputation using Generative Adversarial Nets", ICML,
    2018. https://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf.
    And was implemented by: Lars van Soest, Rune Ebbers, and Ryan Bartelds.

    :param X_data: the original dataset.
    :param prob: the probability of the missing values.
    :param rows: the number of rows (entries).
    :param cols: the number of columns (features).
    :param seed: the random seed.

    :return: a MAR distributed matrix.
    """

    # Fix the seed for run-to-run consistency
    if seed: np.random.seed(seed)

    # Initialize W (weight) and b (bias) matrix for every j (the first value is W, the second is b) and the MAR matrix
    Wb_matrix = np.random.uniform(0., 1., size=(2, cols))
    MAR_matrix = np.random.uniform(0., 1., size=(rows, cols))

    # Calculate the MAR matrix
    for i in range(cols):
        vectors = []
        for l in range(rows):
            vector = Wb_matrix[0] * MAR_matrix[l] * X_data[l] + Wb_matrix[1] * (1 - MAR_matrix[l])
            vectors.append(np.sum(vector[0:i]))
        vectors = np.array(vectors)
        divisor = np.sum(np.exp(-vectors))

        for n in range(rows):
            result = prob * rows * np.exp(-vectors[n]) / divisor
            MAR_matrix[n][i] = 1 * (MAR_matrix[n][i] < result)

    MAR_matrix = 1 - MAR_matrix

    return MAR_matrix


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


def MNAR1(X_data, prob, rows, cols, seed=None):
    """Sample variables distributed Missing not at Random (MNAR).

    This method uses the formula from the supplementary materials of:
    J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data Imputation using Generative Adversarial Nets", ICML,
    2018. https://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
    And was implemented by: Lars van Soest, Rune Ebbers, and Ryan Bartelds.

    :param X_data: the original dataset.
    :param prob: the probability of the missing values.
    :param rows: the number of rows (entries).
    :param cols: the number of columns (features).
    :param seed: the random seed.

    :return: a MAR distributed matrix.
    """

    # Fix the seed for run-to-run consistency
    if seed: np.random.seed(seed)

    # Initialize W (weight) and b (bias) matrix for every j (the first value is W, the second is b) and the MNAR matrix
    W_array = np.random.uniform(0., 1., size=(cols,))
    MNAR_matrix = np.random.uniform(0., 1., size=(rows, cols))

    # Calculate the MNAR matrix
    numerators = np.exp(-W_array[:, None] * X_data.T)
    denominators = np.sum(numerators, axis=1)
    numerators = prob * rows * numerators
    MNAR_matrix = 1 * (MNAR_matrix >= (numerators.T / denominators))

    return MNAR_matrix


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
