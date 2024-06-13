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

"""Cross-validation script

Performs a cross-validation on the selected dataset.

**Example**:

    $ python 03_2_cross-validation.py 
        --data_dir <PATH_OF_DATA> 
        --reports_dir <PATH_TO_STORE_REPORTS>
        --dataset <stanwifi,multienvironment>
"""

import argparse
import os
import sys
sys.path.append("../../..")

import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

from libs.chapter5.pipeline.data_grouping import combine_windows
from libs.chapter5.pipeline.ml import cross_validation
from libs.common.data_loading import ground_truth_to_categorical
from libs.common.utils import save_json, set_seed

STANWIFI_LABELS = ['LIE DOWN', 'FALL', 'WALK', 'RUN', 'SITDOWN', 'STANDUP']
STANWIFI_BATCH_SIZE = 128

MULTI_ENV_LABELS = ['No movement', 'Falling', 'Walking', 'Sitting/Standing', 'Turning', 'Pick up pen']
MULTI_ENV_MAPPING = {'A1': 0, 'A2': 1, 'A3': 2, 'A4': 3, 'A5': 4, 'A6': 5}
MULTIENV_BATCH_SIZE = 256

FOLDS = 10
EPOCHS = 30


def stanwifi_model():
    set_seed()

    model = keras.Sequential([
        layers.Conv2D(filters=16, kernel_size=(5,25), input_shape=(90, 500, 1)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        
        layers.Flatten(),
        
        layers.Dense(512, activation='relu'),
        layers.Dense(6, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
    return model
    

def multienvironment_model():
    set_seed()

    model = keras.Sequential([
        layers.Conv2D(filters=8, kernel_size=(5,25), input_shape=(90, 320, 1)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        
        layers.Flatten(),
        
        layers.Dense(512, activation='relu'),
        layers.Dense(6, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
    return model


def load_multienv_data(path, dataset_dir):
    dataset_path = os.path.join(path, dataset_dir)
    subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10'] if dataset_dir == 'E1' else ['S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20'] 

    windows = {}
    windows_labels = {}
    for subject_id in subjects:
        windows[subject_id] = np.load(os.path.join(dataset_path, f'x_{subject_id}.npy'))
        windows_labels[subject_id] = np.load(os.path.join(dataset_path, f'y_{subject_id}.npy'))
    return windows, windows_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='data directory', type=str, required=True)
    parser.add_argument('--reports_dir', help='directory to store the generated classification reports', type=str, required=True)
    parser.add_argument('--dataset', help='optimize hyperparameters for selected model', type=str, choices=['stanwifi', 'multienvironment'])
    args = parser.parse_args()

    if args.dataset == 'stanwifi':
        x = np.load(os.path.join(args.data_dir, 'x.npy'))
        y = np.load(os.path.join(args.data_dir, 'x.npy'))
        
        model_builder = stanwifi_model
        batch_size = STANWIFI_BATCH_SIZE
        labels = STANWIFI_LABELS
        reports = cross_validation(x, y, stanwifi_model, FOLDS, STANWIFI_BATCH_SIZE, EPOCHS, STANWIFI_LABELS)
        save_json(reports, args.reports_dir, 'cv_report.json')
    else:
        for dataset in ['E1', 'E2']:
            windows, windows_labels = load_multienv_data(args.data_dir, dataset)
            windows_labels_cat = ground_truth_to_categorical(windows_labels, MULTI_ENV_MAPPING)  
            x, y = combine_windows(windows, windows_labels_cat)
            reports = cross_validation(x, y, multienvironment_model, FOLDS, MULTIENV_BATCH_SIZE, EPOCHS, MULTI_ENV_LABELS)
            save_json(reports, args.reports_dir, f'{dataset.lower()}-cv_report.json')
