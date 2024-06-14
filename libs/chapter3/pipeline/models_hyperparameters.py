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


configurations = {
    'mlp_sp': {
        'input_layer': 256,
        'hidden_layers': [512],
        'lr': 0.001
    },
    'mlp_sw': {
        'input_layer': 256,
        'hidden_layers': [512],
        'lr': 0.001
    },
    'mlp_fused': {
        'input_layer': 512,
        'hidden_layers': [256], 
        'lr': 0.001
    },
    'lstm_sp': {
        'lstm_layers': [128, 128],
        'dense_layers': [256, 256, 256],
        'lr': 0.0005
    },
    'lstm_sw': {
        'lstm_layers': [128],
        'dense_layers': [512],
        'lr': 0.001
    },
    'lstm_fused': {
        'lstm_layers': [64, 64],
        'dense_layers': [256, 256, 256],
        'lr': 0.001
    },
    'cnn_sp': {
        'conv_layers': [(32, 5)],
        'dense_layers': [512],
        'lr': 0.0005
    },
    'cnn_sw': {
        'conv_layers': [(128, 25)],
        'dense_layers': [256],
        'lr': 0.0005
    },
    'cnn_fused': {
        'conv_layers': [(32, 5)],
        'dense_layers': [512],
        'lr': 0.001
    },
    'cnn-lstm_sp': {
        'cnn_layers': [(64, 7), (64, 7)],
        'lstm_layers': [128],
        'dense_layers': [128],
        'lr': 0.0005
    },
    'cnn-lstm_sw': {
        'cnn_layers': [(64, 7)],
        'lstm_layers': [128],
        'dense_layers': [128],
        'lr': 0.0005
    },
    'cnn-lstm_fused': {
        'cnn_layers': [(64, 7)],
        'lstm_layers': [128],
        'dense_layers': [128],
        'lr': 0.001
    }
}


def get_model_configuration(model_type, source):
    key = f'{model_type}_{source}'
    
    if key not in configurations:
        raise Exception(f'Unknown configuration model_type={model_type} and source={source}')
    return configurations[key]