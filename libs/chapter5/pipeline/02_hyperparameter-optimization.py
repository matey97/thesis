"""Hyperparameters Grid Search script.

Performs an hyperparameter Grid Search on the specified model. The selected hyperparameters for the search
can be found in `tuning_configuration.py`.

**Example**:

    $ python 02_hyperparameter-optimization.py 
        --data_dir <PATH_OF_DATA> 
        --model <MLP,CNN>
        --phase <initial,extra-layers>
        --batch_size <BATCH_SIZE>
        --epochs <EPOCHS>
        --executions <EXECUTIONS>
"""


import argparse
import os

import sys
sys.path.append("../../..")

from libs.chapter5.pipeline.data_loading import load_data
from libs.chapter5.pipeline.data_grouping import combine_windows
from libs.chapter5.pipeline.hyperparameters_tuning import get_model_builder, create_tuner, tune, get_tuning_summary
from libs.chapter5.pipeline.tuning_configuration import get_tuning_configuration
from libs.common.data_loading import ground_truth_to_categorical
from libs.common.utils import save_json, set_seed

TUNING_DIR = 'GRID_SEARCH_{0}'
TUNING_SUMMARY_FILE = 'summary.json'

BATCH_SIZE = 32
EPOCHS = 50
N_EXECUTIONS = 5

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

def tune_model(data, model_type, batch_size, epochs, n_executions, phase):
    set_seed()    
    model_builder = get_model_builder(model_type)
    optimizing_layers = phase == 'extra-layers' 

    for source, (x, y) in data.items():
        features_dimension = x.shape[1]
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
    parser.add_argument('--model', help='optimize hyperparameters for selected model', type=str, choices=['mlp', 'cnn'])
    parser.add_argument('--phase', help='tuning phase: <initial> to tune layer hyperparameters and <extra-layers> to tune number of layers' , type=str, choices=['initial', 'extra-layers'])
    parser.add_argument('--batch_size', help='training batch size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', help='training epochs', type=int, default=EPOCHS)
    parser.add_argument('--executions', help='executions per trial', type=int, default=N_EXECUTIONS)
    args = parser.parse_args()

    d1_windows, d1_labels = load_data(args.data_dir)
    y = ground_truth_to_categorical(d1_labels, MAPPING)    
    x, y = combine_windows(d1_windows, y)
    print(x.shape)
    
    data = {
        'csi': (x, y)
    }
    tune_model(data, args.model, args.batch_size, args.epochs, args.executions, args.phase)    