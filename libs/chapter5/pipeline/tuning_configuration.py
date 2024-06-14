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


initial_hyperparemter_config = {
    'mlp': {
        'input_units': [128, 256, 512, 1024],
        'hidden_layers': {
            'units': [128, 256, 512, 1024],
            'amount': [1]
        },
        'lr': [0.001, 0.0005, 0.0001]
    },
    'cnn': {
        'cnn_layers': {
            'conv': ['1D', '2D'],
            'filters': [32, 64, 128],
            'filter_sizes_1d': [5, 10, 25],
            'filter_sizes_2d_x': [3, 5],
            'filter_sizes_2d_y': [5, 10, 25],
            'batch_norm': [False, True],
            'max_pooling': [False, True],
            'extra_amount': [0]
        },
        'dense_layers': {
            'units': [128, 256, 512],
            'dropout': [False, True],
            'amount': [1]
        },
        'lr': [0.001, 0.0005, 0.0001]
    },
}


extra_layers_config = {
    'mlp_csi': {
        'input_units': [128],
        'hidden_layers': {
            'units': [1024],
            'amount': [1, 2, 3, 4]
        },
        'lr': [0.0005]
    },
    'cnn_csi': {
        'cnn_layers': {
            'conv': ['2D'],
            'filters': [128], 
            'filter_sizes_2d_x': [5],
            'filter_sizes_2d_y': [25],
            'batch_norm': [True],
            'max_pooling': [True],
            'extra_amount': [0, 1, 2]
        },
        'dense_layers': {
            'units': [512],
            'dropout': [False],
            'amount': [1, 2, 3]
        },
        'lr': [0.0001]
    }
}


def get_tuning_configuration(model_type, source=None):
    if source is None:
        return initial_hyperparemter_config[model_type]
    if (key := f'{model_type}_{source}') in extra_layers_config:
        return extra_layers_config[key]
    raise Exception(f'Unknown tuning configuration for model_type={model_type} and source={source}')