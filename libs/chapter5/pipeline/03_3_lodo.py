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

"""Leaving-One-Day-Out validation script

Performs a Leaving-One-Day-Out evaluation on the LODO dataset.

**Example**:

    $ python 03_3_lodo.py 
        --data_dir <PATH_OF_DATA> 
        --reports_dir <PATH_TO_STORE_REPORTS>
"""

import argparse
import os
import sys
sys.path.append("../../..")

import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

from libs.chapter5.pipeline.ml import evaluate_model
from libs.common.utils import save_json, set_seed


MAPPING = {
    '03/28': 0,
    '03/29': 1,
    '03/30': 2,
    '03/31': 3,
    '04/01': 4,
}

LABELS = ['03/28', '03/29', '03/30', '03/31', '04/01']
NUM_CLASSES = len(LABELS)

BATCH_SIZE = 512
EPOCHS = 30


def build_model():
    set_seed()

    model = keras.Sequential([
        layers.Conv2D(filters=8, kernel_size=(5,25), input_shape=(56, 100, 1)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        
        layers.Flatten(),
        
        layers.Dense(512, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
    return model


def train_models(datasets, labels, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1):
    reports = []
    
    for i in range(len(datasets)):
        training_datasets = [datasets[j] for j in range(len(datasets)) if j != i]
        training_labels = [labels[j] for j in range(len(labels)) if j != i]
        print(f'Training with: {training_labels}')
        print(f'Testing with: {labels[i]}')
    
        x_train = np.vstack(training_datasets)
        y_train = one_hot_encoding(np.concatenate(training_labels), MAPPING)

        x_test = datasets[i]
        y_test = one_hot_encoding(labels[i], MAPPING)

        model = build_model()
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)
        report = evaluate_model(model, x_test, y_test, LABELS)
        reports.append(report)

        del x_train
        del y_train
        del x_test
        del y_test
        del training_datasets
        del training_labels
        del model
    
    return reports


def one_hot_encoding(y, mapping):
    return to_categorical(list(map(lambda i: mapping[i], y)), num_classes=len(mapping.keys()))    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='data directory', type=str, required=True)
    parser.add_argument('--reports_dir', help='directory to store the generated classification reports', type=str, required=True)
    args = parser.parse_args()

    windows = []
    labels = []

    for day in ['03_28', '03_29', '03_30', '03_31', '04_01']:
        windows.append(np.load(os.path.join(args.data_dir), f'{day}_x.npy'))
        labels.append(np.load(os.path.join(args.data_dir), f'{day}_y.npy'))

    reports = train_models(windows, labels)

    save_json(reports, args.reports_dir, 'reports.json')