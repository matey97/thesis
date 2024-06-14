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

"""Hyperparameters Grid Search script.

Performs an hyperparameter Grid Search on the specified model. The selected hyperparameters for the search
can be found in `tuning_configuration.py`.

**Example**:

    $ python 02_hyperparameter-optimization.py 
        --data_dir <PATH_OF_DATA> 
        --model <MLP,CNN,LSTM,CNN-LSTM>
        --phase <initial,extra-layers>
        --batch_size <BATCH_SIZE>
        --epochs <EPOCHS>
        --executions <EXECUTIONS>
"""


import argparse
import os

import sys
sys.path.append("../../..")

from libs.chapter3.pipeline.data_reshapers import get_reshaper
from libs.chapter3.pipeline.hyperparameters_tuning import get_model_builder, create_tuner, tune, get_tuning_summary
from libs.chapter3.pipeline.tuning_configuration import get_tuning_configuration
from libs.common.data_loading import load_data
from libs.common.data_grouping import merge_subjects_datasets
from libs.common.utils import save_json, set_seed


TUNING_DIR = 'GRID_SEARCH_{0}'
TUNING_SUMMARY_FILE = 'summary.json'

ACTIVITIES = {"SEATED": 0, "STANDING_UP": 1, "WALKING": 2, "TURNING": 3, "SITTING_DOWN": 4}

BATCH_SIZE = 64
EPOCHS = 50
N_EXECUTIONS = 5


def tune_model(data, model_type, batch_size, epochs, n_executions, phase):
    set_seed()
    model_builder = get_model_builder(model_type)
    reshaper = get_reshaper(model_type)
    optimizing_layers = phase == 'extra-layers' 

    for source, (x, y) in data.items():
        x, y = merge_subjects_datasets(x, y, list(x.keys()))
        if reshaper is not None:
            x = reshaper(x)
        
        features_dimension = x.shape[-1] if model_type in ['lstm', 'cnn-lstm'] else x.shape[1]
        tuning_configuration = get_tuning_configuration(model_type, source if optimizing_layers else None)
        tuning_configuration['features_dimension'] = features_dimension
        tuning_project = f'{model_type}_{source}{"_layers" if optimizing_layers else ""}'
        print(f'Tuning {model_type} model with {source} data')
        tuner = create_tuner(
            model_builder, 
            n_executions, 
            tuning_configuration, 
            TUNING_DIR.format(phase), 
            tuning_project
        )

        tuner = tune(tuner, x, y, epochs, batch_size)
        save_tuning_summary(tuner, os.path.join(TUNING_DIR, tuning_project))


def save_tuning_summary(tuner, tuning_dir):
    save_json(get_tuning_summary(tuner), tuning_dir, TUNING_SUMMARY_FILE)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='data directory', type=str, required=True)
    parser.add_argument('--model', help='optimize hyperparameters for selected model', type=str, choices=['mlp', 'lstm', 'cnn', 'cnn-lstm'])
    parser.add_argument('--phase', help='tuning phase: <initial> to tune layer hyperparameters and <extra-layers> to tune number of layers' , type=str, choices=['initial', 'extra-layers'])
    parser.add_argument('--batch_size', help='training batch size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', help='training epochs', type=int, default=EPOCHS)
    parser.add_argument('--executions', help='executions per trial', type=int, default=N_EXECUTIONS)
    args = parser.parse_args()

    use_raw_data = args.model != 'mlp'

    x_sp, y_sp = load_data(args.data_dir, 'sp', use_raw_data, ACTIVITIES)
    x_sw, y_sw = load_data(args.data_dir, 'sw', use_raw_data, ACTIVITIES)
    x_fused, y_fused = load_data(args.data_dir, 'fused', use_raw_data, ACTIVITIES)
    data = {
        'sp': (x_sp, y_sp),
        'sw': (x_sw, y_sw),
	    'fused': (x_fused, y_fused)
    }
    tune_model(data, args.model, args.batch_size, args.epochs, args.executions, args.phase)  
    