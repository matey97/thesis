import keras_tuner
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, Flatten, BatchNormalization, Dropout, MaxPooling2D, MaxPooling1D, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from libs.common.utils import RANDOM_SEED

#tf.config.set_visible_devices([], 'GPU')


def mlp_architecture(hp, model_configuration):
    hidden_layers = model_configuration['hidden_layers']
    return Sequential([
        Dense(hp.Choice('input_layer', model_configuration['input_units']), activation='relu', input_shape=(model_configuration['features_dimension'],)),
        *[Dense(hp.Choice(f'hidden_layer', hidden_layers['units']), activation='relu') for layer in range(hp.Choice('n_hidden', hidden_layers['amount']))],
        Dense(10, activation='softmax')
    ])


def cnn_architecture(hp, model_configuration):
    cnn_layers = model_configuration['cnn_layers']
    dense_layers = model_configuration['dense_layers']

    model = Sequential()

    conv1d = hp.Choice('conv', cnn_layers['conv']) == '1D'
    conv_func, pool_func = (Conv1D, MaxPooling1D) if conv1d else (Conv2D, MaxPooling2D)
    n_filters = hp.Choice('input_cnn_filters', cnn_layers['filters'])
    kernel = hp.Choice('input_cnn_filter_size', cnn_layers['filter_sizes_1d']) if conv1d else (hp.Choice('input_cnn_filter_size_x', cnn_layers['filter_sizes_2d_x']), hp.Choice('input_cnn_filter_size_y', cnn_layers['filter_sizes_2d_y']))
    input_shape = (model_configuration['features_dimension'], 50) if conv1d else (model_configuration['features_dimension'], 50, 1)
    model.add(conv_func(
        filters=n_filters,
        kernel_size=kernel,
        padding='same', input_shape=input_shape
    ))

    batch_norm = hp.Choice('batch_norm', cnn_layers['batch_norm'])
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))

    max_pool = hp.Choice('max_pooling', cnn_layers['max_pooling'])
    if max_pool:
        model.add(pool_func())

    for layer in range(hp.Choice('extra_cnn', cnn_layers['extra_amount'])):
        model.add(conv_func(
            filters=n_filters,
            kernel_size=kernel,
            padding='same')
        )

        if batch_norm:
            model.add(BatchNormalization())
        model.add(Activation('relu'))

        if max_pool:
            model.add(pool_func())

    model.add(Flatten())

    dropout = hp.Choice('dropout', dense_layers['dropout'])
    for layer in range(hp.Choice('n_dense', dense_layers['amount'])):
        model.add(Dense(hp.Choice(f'dense_laye', dense_layers['units']), activation='relu'))

        if dropout:
            model.add(Dropout(0.2))       
    model.add(Dense(10, activation='softmax'))

    return model

def get_model_builder(model_type):
    if model_type == 'mlp':
        return mlp_architecture
    elif model_type == 'cnn':
        return cnn_architecture
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
        print(f"fit: {self.it % 5}")
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