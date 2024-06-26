import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, TimeDistributed
from tensorflow.keras.optimizers import Adam

from timeit import default_timer as timer

from .models_hyperparameters import get_model_configuration

#tf.config.set_visible_devices([], 'GPU')

NUM_CLASSES = 5

def mlp_architecture(model_configuration):
    return Sequential([
        Dense(model_configuration['input_layer'], activation='relu', input_shape=(model_configuration['features_dimension'],)),
        *[Dense(hidden_layer, activation='relu') for hidden_layer in model_configuration['hidden_layers']],
        Dense(NUM_CLASSES, activation='softmax')
    ])


def lstm_architecture(model_configuration):
    lstm_layers = model_configuration['lstm_layers']
    extra_lstm_layers = len(lstm_layers) - 1
    dense_layers = model_configuration['dense_layers']
    return Sequential([
        LSTM(lstm_layers[0], return_sequences=extra_lstm_layers!=0, input_shape=(50, model_configuration['features_dimension'])),
        *[LSTM(lstm_layer, return_sequences=extra_lstm_layers-1!=i) for i, lstm_layer in enumerate(lstm_layers[1:])],
        *[Dense(dense_layer, activation='relu') for dense_layer in dense_layers],
        Dense(NUM_CLASSES, activation='softmax')
    ])


def cnn_architecture(model_configuration):
    conv_layers = model_configuration['conv_layers']
    dense_layers = model_configuration['dense_layers']
    return Sequential([
        Conv1D(filters=conv_layers[0][0], kernel_size=conv_layers[0][1], padding='same', activation='relu', input_shape=(model_configuration['features_dimension'], 50)),
        *[Conv1D(filters=conv_layer[0], kernel_size=conv_layer[1], padding='same', activation='relu') for conv_layer in conv_layers[1:]],
        Flatten(),
        *[Dense(dense_layer, activation='relu') for dense_layer in dense_layers],
        Dense(NUM_CLASSES, activation='softmax')
    ])


def cnn_lstm_architecture(model_configuration):
    cnn_layers = model_configuration['cnn_layers']
    lstm_layers = model_configuration['lstm_layers']
    n_lstm = len(lstm_layers)
    dense_layers = model_configuration['dense_layers']
    return Sequential([
        TimeDistributed(Conv1D(
            filters=cnn_layers[0][0], kernel_size=cnn_layers[0][1],
            padding='same', activation='relu'),
            input_shape=(5, 10, model_configuration['features_dimension']),
        ),
        *[
            TimeDistributed(Conv1D(
                filters=layer[0], kernel_size=layer[1],
                padding='same', activation='relu'
            )) for layer in cnn_layers[1:]
        ],
        TimeDistributed(Flatten()),
        *[LSTM(layer, return_sequences=n_lstm-1!=i) for i, layer in enumerate(lstm_layers)],
        *[Dense(layer, activation='relu') for layer in dense_layers],
        Dense(NUM_CLASSES, activation='softmax')
    ])


def create_model(model_type, model_configuration):
    keras.backend.clear_session()
    model = None
    if model_type == 'mlp':
        model = mlp_architecture(model_configuration)
    elif model_type == 'lstm':
        model = lstm_architecture(model_configuration)
    elif model_type == 'cnn':
        model = cnn_architecture(model_configuration)
    elif model_type == 'cnn-lstm':
        model = cnn_lstm_architecture(model_configuration)
    else:
        raise Exception(f'Unknown model type: {model_type}')
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(model_configuration['lr']), metrics=['accuracy'])
    return model


class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

  
def create_trainer(model_type, source, batch_size, epochs):
    def train_model(x, y, validation_data=None, verbose=0):
        cb = TimingCallback()
        features_dimension = x.shape[-1] if model_type in ['lstm', 'conv-lstm', 'cnn-lstm'] else x.shape[1]
        model_configuration = get_model_configuration(model_type, source)
        model_configuration['features_dimension'] = features_dimension
        model = create_model(model_type, model_configuration)
        model.fit(
            x, y, batch_size=batch_size, epochs=epochs, 
            callbacks=[cb, EarlyStopping(monitor='loss', min_delta=0.001, patience=5)],
            validation_data=validation_data, verbose=verbose
        )
        return model, sum(cb.logs)
    
    return train_model