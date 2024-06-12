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

"""Data preprocessing script for preliminar dataset.

Processes the raw data by: arange samples in windows and process them using 1) DBSCAN for outlier detection
and 2-level DWT for threshold based filtering or 2) Choi et al. method.

**Example**:

    $ python 01_1_preliminar-dataset-processing.py 
        --input_data_path <PATH_OF_RAW_DATA> 
        --windowed_data_path <PATH_TO_STORE_RESULTS>
        --method <PROCESSING_METHOD>
        --window_size <WINDOW_SIZE>
        --window_overlap <WINDOW_OVERLAP>
"""


import argparse
import os
import sys

sys.path.append("../../..")

import numpy as np

from alive_progress import alive_bar
from libs.chapter5.pipeline.processing import proposed_method, choi_method
from libs.chapter5.pipeline.raw_data_loading import load_labelled_data

WINDOW_SIZE = 50
WINDOW_OVERLAP = 25


def create_windows(executions_amplitudes, executions_labels, window_size, window_overlap):
    win = {}
    win_labels = {}
    for execution_id in executions_amplitudes:
        amplitudes = executions_amplitudes[execution_id]
        exec_labels = executions_labels[execution_id]

        data = amplitudes
        n = data.shape[1] // window_overlap

        windows = []
        windows_labels = []
        for i in range(0, (n-1) * window_overlap, window_overlap):
            if i+window_size > data.shape[1]:
                break
            window_labels = exec_labels[i:i+window_size]
            values, counts = np.unique(window_labels, return_counts=True)
            if len(values) != 1:
                continue
            windows.append(data[:,i:i+window_size])
            windows_labels.append(values[counts.argmax()])

        windows = np.array(windows)
        windows_labels = np.array(windows_labels)

        win[execution_id] = windows
        win_labels[execution_id] = windows_labels
    return win, win_labels


def process_windows(executions_windows, processing_function):
    processed_windows = {}
    executions_ids = executions_windows.keys()
    with alive_bar(len(executions_ids), title=f'Processing windows', force_tty=True) as progress_bar:
        for execution_id in executions_ids:
            proc_windows = []
            windows = executions_windows[execution_id]
            for window in windows:
                proc_windows.append(processing_function(window))
            processed_windows[execution_id] = np.array(proc_windows)
            progress_bar()
    return processed_windows


def save_windowed_data(data, labels, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    x_file_path = os.path.join(directory, '{0}-x.npy')
    y_file_path = os.path.join(directory, '{0}-y.npy')

    for execution_id in data:
        x = data[execution_id]
        y = labels[execution_id]

        np.save(x_file_path.format(execution_id), x)
        np.save(y_file_path.format(execution_id), y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path', help='Path of input data', type=str, required=True)
    parser.add_argument('--windowed_data_path', help='Path to store windowed data', type=str, required=True)
    parser.add_argument('--method', help='Processing method', required=True, choices=['proposed', 'choi'])
    args = parser.parse_args()

    processing_function = proposed_method if args.method == 'proposed' else choi_method

    for dataset in ['D1', 'D2', 'D3', 'D4']:
        print(f'Processing dataset {dataset}')
        executions_amp, labels = load_labelled_data(os.path.join(args.input_data_path, dataset))
        windows, windows_labels = create_windows(executions_amp, labels, WINDOW_SIZE, WINDOW_OVERLAP)
        windows_processed = process_windows(windows, processing_function)
        save_windowed_data(windows_processed, windows_labels, os.path.join(args.windowed_data_path, dataset))
