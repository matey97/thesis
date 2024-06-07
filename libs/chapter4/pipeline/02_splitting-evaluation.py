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


"""Splitting approach evaluation script

This script trains 100 models for each data source (smartphone, smartwatch) and splitting approach. For the training process,
a 80/20 train/test split is employed with a batch size of 20 windows during 50 epochs.

**Example**:

    $ python 02_splitting-evaluation.py 
        --ts_data_path <PATH_OF_TURNING_SITTING_DATA> 
        --tts_data_path <PATH_OF_TURN_TO_SIT_DATA>
        --reports_output_path <PATH_TO_STORE_REPORTS>
"""


import argparse
import json
import os
import sys
sys.path.append("../../..")

from alive_progress import alive_bar
from libs.chapter4.pipeline.training import create_trainer
from libs.common.data_loading import load_data
from libs.common.data_grouping import generate_training_and_test_sets
from libs.common.ml import generate_report
from libs.common.utils import set_seed
from sklearn.model_selection import train_test_split


TURNING_AND_SITTING_MAPPING = {"SEATED": 0, "STANDING_UP": 1, "WALKING": 2, "TURNING": 3, "SITTING_DOWN": 4}
TURN_TO_SIT_MAPPING = {"SEATED": 0, "STANDING_UP": 1, "WALKING": 2, "TURNING": 3, "TURN_TO_SIT": 4}

BATCH_SIZE = 20
EPOCHS = 50

def training_report_from_datasets(datasets, models_per_dataset=100):
    set_seed()
    trainer = create_trainer(BATCH_SIZE, EPOCHS)
    reports = {}

    for dataset_id, (x, y) in datasets.items():
        reports[dataset_id] = []
        activity_names = TURNING_AND_SITTING_MAPPING.keys() if 'turning_and_sitting' in dataset_id else TURN_TO_SIT_MAPPING.keys()
        with alive_bar(models_per_dataset, title=f'Training models for dataset {dataset_id}', force_tty=True) as progress:
            for i in range(models_per_dataset):
                train_subjects, test_subjects = train_test_split(list(x.keys()), test_size=0.2)

                x_train, y_train, x_test, y_test = generate_training_and_test_sets(x, y, train_subjects, test_subjects)

                model = trainer(x_train, y_train, verbose=0)
                y_pred = model.predict(x_test)

                reports[dataset_id].append(generate_report(y_test, y_pred, activity_names))
                progress()

    return reports


def store_reports(reports, path):
    with open(os.path.join(path, 'reports.json'), 'w') as file:
        json.dump(reports, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ts_data_path', help='Path of data labelled with TURNING and SITTING_DOWN activities', type=str, required=True)
    parser.add_argument('--tts_data_path', help='Path of data labelled with TURN_TO_SIT activity', type=str, required=True)
    parser.add_argument('--reports_output_path', help='Path to store the generated reports', type=str, required=True)
    args = parser.parse_args()
    
    x_sp_ts, y_sp_ts = load_data(args.ts_data_path, 'sp', True, TURNING_AND_SITTING_MAPPING)
    x_sw_ts, y_sw_ts = load_data(args.ts_data_path, 'sw', True, TURNING_AND_SITTING_MAPPING)

    x_sp_tts, y_sp_tts = load_data(args.tts_data_path, 'sp', True, TURN_TO_SIT_MAPPING)
    x_sw_tts, y_sw_tts = load_data(args.tts_data_path, 'sw', True, TURN_TO_SIT_MAPPING)

    datasets = {
        'sw_turning_and_sitting': [x_sw_ts, y_sw_ts],
        'sp_turning_and_sitting': [x_sp_ts, y_sp_ts],
        'sw_turn_to_sit': [x_sw_tts, y_sw_tts],
        'sp_turn_to_sit': [x_sp_tts, y_sp_tts],
    }

    reports = training_report_from_datasets(datasets)
    #store_reports(reports, args.reports_output_path)

    
