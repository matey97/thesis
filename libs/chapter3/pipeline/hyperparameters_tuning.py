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


import keras_tuner
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, TimeDistributed
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from libs.common.utils import RANDOM_SEED

#tf.config.set_visible_devices([], 'GPU')

def mlp_architecture(hp, model_configuration):
    hidden_layers = model_configuration['hidden_layers']
    return Sequential([
        Dense(hp.Choice('input_layer', model_configuration['input_units']), activation='relu', input_shape=(model_configuration['features_dimension'],)),
        *[Dense(hp.Choice(f'hidden_layer', hidden_layers['units']), activation='relu') for layer in range(hp.Choice('n_hidden', hidden_layers['amount']))],
        Dense(5, activation='softmax')
    ])


def lstm_architecture(hp, model_configuration):
    lstm_layers = model_configuration['lstm_layers']
    extra_lstm_layers = hp.get('extra_lstm') if 'extra_lstm' in hp.values else None
    dense_layers = model_configuration['dense_layers']
    return Sequential([
        LSTM(hp.Choice('input_lstm', lstm_layers['units']), return_sequences=extra_lstm_layers!=0, activation='relu', input_shape=(50, model_configuration['features_dimension'])),
        *[LSTM(hp.Choice(f'lstm', lstm_layers['units']), return_sequences=extra_lstm_layers-1!=layer) for layer in range(hp.Choice('extra_lstm', lstm_layers['extra_amount']))],
        *[Dense(hp.Choice(f'dense_layer', dense_layers['units']), activation='relu') for layer in range(hp.Choice('n_dense', dense_layers['amount']))],
        Dense(5, activation='softmax')
    ])


def cnn_architecture(hp, model_configuration):
    cnn_layers = model_configuration['cnn_layers']
    dense_layers = model_configuration['dense_layers']
    return Sequential([
        Conv1D(
            filters=hp.Choice('input_cnn_filters', cnn_layers['filters']),
            kernel_size=hp.Choice('input_cnn_filter_size', cnn_layers['filter_sizes']),
            padding='same', activation='relu', input_shape=(model_configuration['features_dimension'], 50)),
        *[
            Conv1D(
                filters=hp.Choice(f'cnn_filters', cnn_layers['filters']),
                kernel_size=hp.Choice(f'filter_size', cnn_layers['filter_sizes']),
                padding='same', activation='relu'
            ) for layer in range(hp.Choice('extra_cnn', cnn_layers['extra_amount']))
        ],
        Flatten(),
        *[Dense(hp.Choice(f'dense_laye', dense_layers['units']), activation='relu') for layer in range(hp.Choice('n_dense', dense_layers['amount']))],
        Dense(5, activation='softmax')
    ])


def cnn_lstm_architecture(hp, model_configuration):
    cnn_layers = model_configuration['cnn_layers']
    #extra_cnn_layers = hp.get('extra_cnn') if 'extra_cnn' in hp.values else None 
    lstm_layers = model_configuration['lstm_layers']
    n_lstm =  hp.get('n_lstm') if 'n_lstm' in hp.values else 1 
    dense_layers = model_configuration['dense_layers']
    return Sequential([
        TimeDistributed(Conv1D(
            filters=hp.Choice('input_cnn_filters', cnn_layers['filters']),
            kernel_size=hp.Choice('input_cnn_filter_size', cnn_layers['filter_sizes']),
            padding='same', activation='relu'),
            input_shape=(5, 10, model_configuration['features_dimension']),
        ),
        *[
            TimeDistributed(Conv1D(
                filters=hp.Choice(f'cnn_filters', cnn_layers['filters']),
                kernel_size=hp.Choice(f'filter_size', cnn_layers['filter_sizes']),
                padding='same', activation='relu'
            )) for layer in range(hp.Choice('extra_cnn', cnn_layers['extra_amount']))
        ],
        TimeDistributed(Flatten()),
        *[LSTM(hp.Choice(f'lstm', lstm_layers['units']), return_sequences=n_lstm-1!=layer) for layer in range(hp.Choice('n_lstm', lstm_layers['amount']))],
        *[Dense(hp.Choice(f'dense_layer', dense_layers['units']), activation='relu') for layer in range(hp.Choice('n_dense', dense_layers['amount']))],
        Dense(5, activation='softmax')
    ])


def get_model_builder(model_type):
    if model_type == 'mlp':
        return mlp_architecture
    elif model_type == 'lstm':
        return lstm_architecture
    elif model_type == 'cnn':
        return cnn_architecture
    elif model_type == 'cnn-lstm':
        return cnn_lstm_architecture
    else:
        raise Exception(f'Unknown model type: {model_type}')


class Hypermodel(keras_tuner.HyperModel):
    it = 0
    seeds = [5354, 5355, 5356, 5357, 5358]
    
    def __init__(self, model_builder, model_configuration):
        self.model_builder = model_builder
        self.model_configuration = model_configuration
    
    def build(self, hp):
        keras.backend.clear_session()
        model = self.model_builder(hp, self.model_configuration)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(hp.Choice('lr', self.model_configuration['lr'])), metrics=['accuracy'])
        return model
    
    def fit(self, hp, model, x, y, validation_data=None, **kwargs):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=self.next_seed())
        return model.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=0, **kwargs)    
    
    def next_seed(self):
        i = self.it % 5
        self.it += 1
        return self.seeds[i]


def create_tuner(model_builder, n_executions, model_configuration, tune_dir, tune_project):
    return keras_tuner.GridSearch(
        Hypermodel(model_builder, model_configuration),
        objective='val_loss',
        executions_per_trial=n_executions,
        overwrite=True,
        directory=tune_dir,
        project_name=tune_project,
        seed=RANDOM_SEED
    )


def tune(tuner, x, y, epochs, batch_size):
    tuner.search(
        x, y, epochs=epochs, batch_size=batch_size, validation_data=None,
        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5)]
    )

    return tuner


def get_tuning_summary(tuner):
    n_trials = len(tuner.oracle.get_state()['end_order'])
    best_trials = tuner.oracle.get_best_trials(num_trials=n_trials)
    summary = [{'hyperparameters': trial.hyperparameters.get_config()['values'], 'score': trial.score} for trial in best_trials]
    return summary