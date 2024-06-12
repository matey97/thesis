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

"""Data preprocessing script for LODO dataset.

Processes the raw data by aranging samples in windows and process them using DBSCAN for outlier detection
and 2-level DWT for threshold based filtering.

**Example**:

    $ python 01.4_lodo-dataset-processing.py 
        --input_data_path <PATH_OF_RAW_DATA> 
        --windowed_data_path <PATH_TO_STORE_RESULTS>
        --window_size <WINDOW_SIZE>
"""


import argparse
import copy
import os
import sys

sys.path.append("../../..")

import numpy as np

from alive_progress import alive_bar
from libs.chapter5.pipeline.processing import proposed_method

WINDOW_SIZE = 100

def create_windows(dataset, labels, window_size=100):
    splits = np.arange(window_size, dataset.shape[1], window_size)
    return np.array(np.split(dataset, splits, axis=1)[:-1]), np.array(np.split(labels, splits, axis=0)[:-1])[:,0]


def process_windows(windows):
    windows_copy = copy.deepcopy(windows)
    with alive_bar(len(windows_copy), title=f'Processing windows', force_tty=True, refresh_secs=5) as progress_bar:
        for i, window in enumerate(i, windows_copy):
            windows_copy[i] = proposed_method(window)
            progress_bar()
    return windows_copy



def save_windowed_data(data, labels, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    np.save(os.path.join(directory, '{0}_x.npy'), data)
    np.save(os.path.join(directory, '{0}_y.npy'), labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path', help='Path of input data', type=str, required=True)
    parser.add_argument('--windowed_data_path', help='Path to store windowed data', type=str, required=True)
    parser.add_argument('--window_size', help='Window size', required=True, default=WINDOW_SIZE)
    args = parser.parse_args()

    amplitude_files = ['amplitudes_03_28.npy', 'amplitudes_03_29.npy', 'amplitudes_03_30.npy', 'amplitudes_03_31.npy', 'amplitudes_04_01.npy']
    labels_files = ['labels_03_28.npy', 'labels_03_29.npy', 'labels_03_30.npy', 'labels_03_31.npy', 'labels_04_01.npy']

    for amplitude_file, label_file in zip(amplitude_files, labels_files):
        print(f'Processing dataset {amplitude_file}')
        name = amplitude_file.split('_', 1)[1]
        amplitudes = np.load(os.path.join(args.input_data_path, amplitude_file))
        labels = np.load(os.path.join(args.input_data_path, label_file))

        windows, windows_labels = create_windows(amplitudes, labels, args.window_size)

        del amplitudes, labels

        windows_processed = process_windows(windows)
        save_windowed_data(windows_processed, windows_labels, os.path.join(args.windowed_data_path, name))