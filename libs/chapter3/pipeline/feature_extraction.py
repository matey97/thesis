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

from functools import reduce


def apply_feature_extraction(window):
    means = np.apply_along_axis(np.mean, 1, window)
    medians = np.apply_along_axis(np.median, 1, window)
    maxs = np.apply_along_axis(np.max, 1, window)
    mins = np.apply_along_axis(np.min, 1, window)
    stds = np.apply_along_axis(np.std, 1, window)
    ranges = maxs - mins
    rmss = np.apply_along_axis(rms, 1, window)
    pitch_rolls = pitch_and_roll(window)
    angles = compute_gyro_angle(window)

    return np.concatenate((means, medians, maxs, mins, stds, ranges, rmss, pitch_rolls, angles))

def rms(values):
    return np.sqrt(reduce(lambda prev, curr: prev + curr ** 2, values, 0) / len(values))

    
def pitch_and_roll(window):
    def angular_function(a, b):
        return np.arctan2(a, b) * 180/np.pi

    return np.array([
        np.mean(angular_function(window[1], window[2])),
        np.mean(angular_function(-window[0], np.sqrt(np.power(window[1], 2) + np.power(window[2], 2))))
    ])

    
def integrator(series):
    return np.trapz(series)


def compute_gyro_angle(window):
    return np.array([
        integrator(window[3]), integrator(window[4]), integrator(window[5])
    ])