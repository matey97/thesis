"""Data preprocessing script for Multi-environment dataset.

Processes the raw data by aranging samples in windows and processing them using DBSCAN for outlier detection
and 2-level DWT for threshold based filtering

**Example**:

    $ python 01_3_multienvironment-processing.py 
        --input_data_path <PATH_OF_RAW_DATA> 
        --windowed_data_path <PATH_TO_STORE_RESULTS>
"""

import argparse
import copy
import os
import sys
sys.path.append("../../..")

import numpy as np
import pandas as pd

from alive_progress import alive_bar
from libs.chapter5.pipeline.processing import proposed_method
from math import sqrt

ACTIVITY_MAPPING = {
    'A01': 'A1',
    'A02': 'A2',
    'A03': 'A1',
    'A04': 'A1',
    'A05': 'A2',
    'A06': 'A3',
    'A07': 'A5',
    'A08': 'A3',
    'A09': 'A5',
    'A10': 'A4',
    'A11': 'A4',
    'A12': 'A6',
}

def load_multienvironment_dataset(environment):
    data = {}
    subject_dirs = os.listdir(environment)
    subject_dirs = list(filter(lambda x: x.startswith('Subject'), subject_dirs))
    with alive_bar(len(subject_dirs), title=f'Loading data from subjects', force_tty=True) as progress_bar:
        for subject_dir in subject_dirs:
            subject = f'S{int(subject_dir.split(" ")[-1]):02d}'
            data[subject] = {}
            subject_dir_path = os.path.join(environment, subject_dir)
            for file in os.listdir(subject_dir_path):
                if not file.endswith('.csv'):
                    continue

                base_activity = file.split('_')[3]
                file_path = os.path.join(subject_dir_path, file)
                df = pd.read_csv(file_path)
                df = df.iloc[160:-160] #remove 0.5 sec after and before due to noise

                if base_activity not in data[subject]:
                    data[subject][base_activity] = df
                else:
                    data[subject][base_activity] = pd.concat([data[subject][base_activity], df])
            progress_bar()
    return data


def amplitude_from_raw_data(data):
    amplitudes = {}
    with alive_bar(len(data.keys()), title=f'Extracting amplitudes from subject\'s data', force_tty=True) as progress_bar:
        for subject in data:
            amplitudes[subject] = {}
            for activity in data[subject]:
                activity_data = data[subject][activity]
                activity_amplitudes = []
                for index, row in activity_data.iterrows():
                    instance_amplitudes = []
                    for antenna in range(1,4):
                        for subcarrier in range(1,31):
                            csi_data = row[f'csi_1_{antenna}_{subcarrier}']
                            real, imaginary = csi_data.split('+')
                            real = int(real)
                            imaginary = int(imaginary[:-1])

                            instance_amplitudes.append(sqrt(imaginary ** 2 + real ** 2))
                    activity_amplitudes.append(instance_amplitudes)
                amplitudes[subject][activity] = np.array(activity_amplitudes)
            progress_bar()
    return amplitudes


def create_windows(amplitudes, window_size=320, window_overlap=160):
    windows = {}
    windows_labels = {}
    for subject_id in amplitudes:
        subject_windows = []
        subject_windows_labels = []
        for activity_id in amplitudes[subject_id]:
            activity_amplitudes = amplitudes[subject_id][activity_id].T

            n = activity_amplitudes.shape[1] // window_overlap
            for i in range(0, (n-1) * window_overlap, window_overlap):
                if i+window_size > activity_amplitudes.shape[1]:
                    break
                subject_windows.append(activity_amplitudes[:,i:i+window_size])
                subject_windows_labels.append(ACTIVITY_MAPPING[activity_id])

        windows[subject_id] = np.array(subject_windows)
        windows_labels[subject_id] = np.array(subject_windows_labels)
    return windows, windows_labels


def process_windows(windows):
    proc_windows = {}
    with alive_bar(len(windows.keys()), title=f'Processing subject\'s windows', force_tty=True) as progress_bar:
        for subject_id in windows:
            windows_copy = copy.deepcopy(windows[subject_id])
            for i in range(len(windows_copy)):
                windows_copy[i] = proposed_method(windows_copy[i])
            proc_windows[subject_id] = windows_copy
            progress_bar()
    return proc_windows


def save_windowed_data(data, labels, directory):
    for subject_id, subject_data in data.items():
        np.save(os.path.join(directory, f'{subject_id}_x.npy'), subject_data)
        np.save(os.path.join(directory, f'{subject_id}_x.npy'), labels[subject_id])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path', help='Path of input data', type=str, required=True)
    parser.add_argument('--windowed_data_path', help='Path to store windowed data', type=str, required=True)
    args = parser.parse_args()

    for dataset in ['ENVIRONMENT 1', 'ENVIRONMENT 2']:
        print(f'Processing dataset {dataset}')
        data = load_multienvironment_dataset(os.path.join(args.input_data_path, dataset))
        amplitudes = amplitude_from_raw_data(data)
        windows, windows_labels = create_windows(amplitudes)

        del data, amplitudes
        
        proc_windows = process_windows(windows)
        save_windowed_data(proc_windows, windows_labels, args.windowed_data_path)
