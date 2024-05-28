def lstm_reshaper(data):
    return data.transpose(0, 2, 1)


def cnn_lstm_reshaper(data):
    n_windows, n_features, n_timesteps = data.shape
    return data.reshape((n_windows, n_features * n_timesteps)).reshape((n_windows, 5, 10, n_features))


def create_reshaper(reshaper):
    def do_reshape(x):
        return reshaper(x)

    return do_reshape


def get_reshaper(model_type):
    if model_type == 'lstm':
        return create_reshaper(lstm_reshaper)
    if model_type == 'cnn-lstm':
        return create_reshaper(cnn_lstm_reshaper)
    return None