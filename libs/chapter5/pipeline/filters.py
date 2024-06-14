# Copyright 2024 Miguel Matey Sanz
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


import pywt
import numpy as np

from scipy.ndimage import uniform_filter
from sklearn.cluster import DBSCAN


def dbscan_filtering(data):
    dbscan = DBSCAN().fit(data.reshape(-1, 1))
    for noisy_value in np.where(dbscan.labels_ == -1)[0]:
        if noisy_value <= 5:
            start, end = 0, 10
        elif noisy_value >= len(data) - 6:
            start, end = -10, -1
        else:
            start, end = noisy_value - 5, noisy_value + 5
        data[noisy_value] = np.mean(data[start: end])
    return data


def wavelet_filtering(data):
    cA2, cD2, cD1 = pywt.wavedec(data, 'db4', level=2)
    cD1[cD1 > 0.1] = 0.1
    cD1[cD1 < -0.1] = -0.1
    cD2[cD2 > 0.1] = 0.1
    cD2[cD2 < -0.1] = -0.1
    reconstructed = pywt.waverec([cA2, cD2, cD1], 'db4')
    return uniform_filter(reconstructed)