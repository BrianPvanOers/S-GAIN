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

"""Metrics calculations for S-GAIN:

Metrics:
(1) get_rmse: evaluate the imputed data in terms of RMSE
(2) get_sparsity: compute the sparsity of the model
"""

import numpy as np

from utils.utils import normalization


# -- Metrics ----------------------------------------------------------------------------------------------------------

def get_rmse(data_x, imputed_data_x, data_mask, rounding=False):
    """Compute the RMSE between the original data and the imputed data.

    :param data_x: the original data (without missing values)
    :param imputed_data_x: the imputed data
    :param data_mask: the indicator matrix for missing elements
    :param rounding: whether to round or not

    :return: the Root Mean Squared Error (rounded to 4 decimals)
    """

    data_x, norm_parameters = normalization(data_x)
    imputed_data, _ = normalization(imputed_data_x, norm_parameters)

    nominator = np.sum(((1 - data_mask) * data_x - (1 - data_mask) * imputed_data) ** 2)
    denominator = np.sum(1 - data_mask)
    RMSE = np.sqrt(nominator / float(denominator))
    if rounding: RMSE = f'{RMSE:.4f}'

    return RMSE


def get_sparsity(theta):
    """Compute the actual sparsity of one of the models (generator or discriminator).

    :param theta: the layer weights and biases of the model

    :return:
    - M_sparsity: the total sparsity of the model
    - W1_sparsity: the sparsity of the first layer of the model
    - W2_sparsity: the sparsity of the second layer of the model
    - W3_sparsity: the sparsity of the third layer of the model
    """

    W1, W2, W3, b1, b2, b3 = theta

    W1_size = np.size(W1)
    W1_nzc = np.count_nonzero(W1)
    W1_sparsity = (W1_size - W1_nzc) / W1_size

    W2_size = np.size(W2)
    W2_nzc = np.count_nonzero(W2)
    W2_sparsity = (W2_size - W2_nzc) / W2_size

    W3_size = np.size(W3)
    W3_nzc = np.count_nonzero(W3)
    W3_sparsity = (W3_size - W3_nzc) / W3_size

    M_size = W1_size + W2_size + W3_size
    M_nzc = W1_nzc + W2_nzc + W3_nzc
    M_sparsity = (M_size - M_nzc) / M_size

    return M_sparsity, W1_sparsity, W2_sparsity, W3_sparsity
