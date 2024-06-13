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

"""Multiple evaluation script

Performs a cross-validation and an evaluation with different subsets collected at different time frames.

**Example**:

    $ python 03_1_multiple_evaluations.py 
        --data_dir <PATH_OF_DATA> 
        --reports_dir <PATH_TO_STORE_REPORTS>
        --model <MLP,CNN>
"""


import argparse
import os
import sys
sys.path.append("../../..")


from tensorflow import keras
from tensorflow.keras import layers

from libs.chapter5.pipeline.data_loading import load_data
from libs.chapter5.pipeline.data_grouping import combine_windows, split_train_test
from libs.chapter5.pipeline.ml import cross_validation, evaluate_model
from libs.common.data_loading import ground_truth_to_categorical
from libs.common.utils import save_json, set_seed

MAPPING = {
    'SEATED_RX': 0, 
    'STANDING_UP_RX': 1, 
    'WALKING_TX': 2, 
    'TURN_TX': 3, 
    'SITTING_DOWN_TX': 4, 
    'SEATED_TX': 5, 
    'STANDING_UP_TX': 6,
    'WALKING_RX': 7,
    'TURN_RX': 8,
    'SITTING_DOWN_RX': 9,
}
LABELS = ['SEATED_RX','STANDING_UP_RX','WALKING_TX','TURNING_TX','SITTING_DOWN_TX', 'SEATED_TX', 'STANDING_UP_TX','WALKING_RX','TURNING_RX','SITTING_DOWN_RX']
NUM_CLASSES = len(LABELS)

TRAIN_IDS = ['e01_rx_tx', 'e01_tx_rx', 'e02_rx_tx', 'e02_tx_rx', 'e03_rx_tx', 'e03_tx_rx', 'e04_rx_tx', 'e04_tx_rx',
             'e05_rx_tx', 'e05_tx_rx', 'e06_rx_tx', 'e06_tx_rx', 'e07_rx_tx', 'e07_tx_rx', 'e08_rx_tx', 'e08_tx_rx']
TEST_IDS = ['e09_rx_tx', 'e09_tx_rx', 'e10_rx_tx', 'e10_tx_rx']

BATCH_SIZE = 32
EPOCHS = 50
FOLDS = 10

def mlp_model():
    set_seed()

    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(500,)),
        layers.Dense(1024, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0005), metrics=['accuracy'])
    return model


def cnn_model():
    set_seed()

    model = keras.Sequential([
        layers.Conv2D(filters=128, kernel_size=(5,25), input_shape=(56, 50, 1)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        
        layers.Flatten(),
        
        layers.Dense(512, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
    return model


def model_builder(model_type):
    if model_type == 'cnn':
        return cnn_model
    return mlp_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='data directory', type=str, required=True)
    parser.add_argument('--reports_dir', help='directory to store the generated classification reports', type=str, required=True)
    parser.add_argument('--model', help='optimize hyperparameters for selected model', type=str, choices=['mlp', 'cnn'])
    args = parser.parse_args()

    d1_windows, d1_labels = load_data(os.path.join(args.data_dir, 'D1'))
    d1_labels_cat = ground_truth_to_categorical(d1_labels, MAPPING)    
    x, y = combine_windows(d1_windows, d1_labels_cat)
    
    print("Starting 10-fold cross-validation")
    cv_reports = cross_validation(x, y, model_builder(args.model), FOLDS, BATCH_SIZE, EPOCHS, LABELS)
    save_json(cv_reports, args.reports_dir, 'cv_report.json')

    print("Starting D1T training and D1E evaluation")
    (x_d1t, y_d1t), (x_d1e, y_d1e) = split_train_test(d1_windows, d1_labels_cat, TRAIN_IDS, TEST_IDS)
    model = model_builder(args.model)()
    model.fit(x_d1t, y_d1t, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0)
    report = evaluate_model(model, x_d1e, y_d1e, LABELS)
    save_json(report, args.reports_dir, 'd1_report.json')

    print("Starting D2, D3 and D4 evaluation")
    for eval_dataset in ['D2', 'D3', 'D4']:
        windows, labels = load_data(os.path.join(args.data_dir, eval_dataset))
        labels_cat = ground_truth_to_categorical(labels, MAPPING)    
        x, y = combine_windows(windows, labels_cat)
        report = evaluate_model(model, x, y, LABELS)
        save_json(report, args.reports_dir, f'{eval_dataset.lower()}_report.json')

