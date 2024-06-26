from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam


NUM_CLASSES = 5

def create_model():
    model = Sequential([
        Conv1D(filters=64, kernel_size=10, padding='same', activation='relu', input_shape=(6, 50)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model



def create_trainer(batch_size, epochs):
    def train_model(x, y, validation_data=None, verbose=0):
        model = create_model()
        model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_data=validation_data, verbose=verbose)
        return model
    
    return train_model