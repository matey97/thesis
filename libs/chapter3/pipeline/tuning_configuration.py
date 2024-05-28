initial_hyperparemter_config = {
    'mlp': {
        'input_units': [128, 256, 512],
        'hidden_layers': {
            'units': [128, 256, 512],
            'amount': [1]
        },
        'lr': [0.001, 0.0005, 0.0001]
    },
    'lstm': {
        'lstm_layers': {
            'units': [32, 64, 128],
            'extra_amount': [0]
        },
        'dense_layers': {
            'units': [128, 256, 512],
            'amount': [1]
        },
        'lr': [0.001, 0.0005, 0.0001]
    },
    'cnn': {
        'cnn_layers': {
            'filters': [32, 64, 128],
            'filter_sizes': [5, 10, 25],
            'extra_amount': [0]
        },
        'dense_layers': {
            'units': [128, 256, 512],
            'amount': [1]
        },
        'lr': [0.001, 0.0005, 0.0001]
    },
    'cnn-lstm': {
        'cnn_layers': {
            'filters': [32, 64, 128],
            'filter_sizes': [3, 5, 7],
            'extra_amount': [0]
        },
        'lstm_layers': {
            'units': [32, 64, 128],
            'amount': [1]
        },
        'dense_layers': {
            'units': [128, 256, 512],
            'amount': [1]
        },
        'lr': [0.001, 0.0005, 0.0001]
    }
}


extra_layers_config = {
    'mlp_sp': {
        'input_units': [256],
        'hidden_layers': {
            'units': [512],
            'amount': [1, 2, 3]
        },
        'lr': [0.001]
    },
    'mlp_sw': {
        'input_units': [256],
        'hidden_layers': {
            'units': [512],
            'amount': [1, 2, 3]
        },
        'lr': [0.001]
    },
    'mlp_fused': {
        'input_units': [512],
        'hidden_layers': {
            'units': [256],
            'amount': [1, 2, 3]
        },
        'lr': [0.001]
    },
    'lstm_sp': {
        'lstm_layers': {
            'units': [128],
            'extra_amount': [0, 1, 2]
        },
        'dense_layers': {
            'units': [256],
            'amount': [1, 2, 3]
        },
        'lr': [0.0005]
    },
    'lstm_sw': {
        'lstm_layers': {
            'units': [128],
            'extra_amount': [0, 1, 2]
        },
        'dense_layers': {
            'units': [512],
            'amount': [1, 2, 3]
        },
        'lr': [0.001]
    },
    'lstm_fused': {
        'lstm_layers': {
            'units': [64],
            'extra_amount': [0, 1, 2]
        },
        'dense_layers': {
            'units': [256],
            'amount': [1, 2, 3]
        },
        'lr': [0.001]
    },
    'cnn_sp': {
        'cnn_layers': {
            'filters': [32],
            'filter_sizes': [5],
            'extra_amount': [0, 1, 2]
        },
        'dense_layers': {
            'units': [512],
            'amount': [1, 2, 3]
        },
        'lr': [0.0005]
    },
    'cnn_sw': {
        'cnn_layers': {
            'filters': [128],
            'filter_sizes': [25],
            'extra_amount': [0, 1, 2]
        },
        'dense_layers': {
            'units': [256],
            'amount': [1, 2, 3]
        },
        'lr': [0.0005]
    },
    'cnn_fused': {
        'cnn_layers': {
            'filters': [32],
            'filter_sizes': [5],
            'extra_amount': [0, 1, 2]
        },
        'dense_layers': {
            'units': [512],
            'amount': [1, 2, 3]
        },
        'lr': [0.001]
    },
    'cnn-lstm_sp': {
        'cnn_layers': {
            'filters': [64],
            'filter_sizes': [7],
            'extra_amount': [0, 1, 2]
        },
        'lstm_layers': {
            'units': [128],
            'amount': [1, 2, 3]
        },
        'dense_layers': {
            'units': [128],
            'amount': [1, 2, 3]
        },
        'lr': [0.0005]
    },
    'cnn-lstm_sw': {
        'cnn_layers': {
            'filters': [64],
            'filter_sizes': [7],
            'extra_amount': [0, 1, 2]
        },
        'lstm_layers': {
            'units': [128],
            'amount': [1, 2, 3]
        },
        'dense_layers': {
            'units': [128],
            'amount': [1, 2, 3]
        },
        'lr': [0.0005]
    },
    'cnn-lstm_fused': {
        'cnn_layers': {
            'filters': [64],
            'filter_sizes': [7],
            'extra_amount': [0, 1, 2]
        },
        'lstm_layers': {
            'units': [128],
            'amount': [1, 2, 3]
        },
        'dense_layers': {
            'units': [128],
            'amount': [1, 2, 3]
        },
        'lr': [0.001]
    }
}


def get_tuning_configuration(model_type, source=None):
    if source is None:
        return initial_hyperparemter_config[model_type]
    if (key := f'{model_type}_{source}') in extra_layers_config:
        return extra_layers_config[key]
    raise Exception(f'Unknown tuning configuration for model_type={model_type} and source={source}')