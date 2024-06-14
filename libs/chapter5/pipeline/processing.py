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


import numpy as np

from hampel import hampel
from scipy.signal import savgol_filter
from libs.chapter5.pipeline.filters import dbscan_filtering, wavelet_filtering


def proposed_method(window):
    return np.apply_along_axis(lambda x: wavelet_filtering(dbscan_filtering(x)),1, window)


def choi_method(window):
    window = np.apply_along_axis(lambda x: savgol_filter(hampel(x, window_size=3, n_sigma=3.0).filtered_data, window_length=5, polyorder=2),1, window)
    return _extract_features_Choi(window)


def _adjacent_amplitude_difference(window, N=2):
    adjacent_differences = []
    for i in range(N, window.shape[0] - N):
        diff = np.zeros(shape=(50,))
        for j in range(1, N + 1):
            diff += np.abs(window[i] - window[i-j]) + np.abs(window[i] - window[i+j])
        adjacent_difference = np.mean(diff)
        adjacent_differences.append(adjacent_difference)
    return np.array(adjacent_differences)


def _euclian_distance(window):
    distances = []
    for i in range(window.shape[1] - 1):
        distances.append(np.abs(window[:, i+1] - window[:, i]))
    return np.median(np.array(distances), axis=0)


def _extract_features_Choi(window):
    stds = np.std(window, axis=1)
    mins = np.min(window, axis=1)
    maxs = np.max(window, axis=1)
    qtls = np.quantile(window, 0.25, axis=1)
    qtus = np.quantile(window, 0.75, axis=1)
    avgs = np.mean(window, axis=1)

    iqr = qtus - qtls
    adjs = _adjacent_amplitude_difference(window)
    eucs = _euclian_distance(window)

    features = np.concatenate([stds, mins, maxs, qtls, qtus, avgs, iqr, adjs, eucs])
    return features