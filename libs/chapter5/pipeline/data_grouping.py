import numpy as np


def combine_windows(windows, windows_labels):
    x = np.vstack([ window for window in windows.values() ])
    y = np.concatenate([ window_labels for window_labels in windows_labels.values() ])
    return x, y 


def split_train_test(windows, windows_labels, train_ids, test_ids):
    train_windows = {id: windows[id] for id in train_ids}
    train_labels = {id: windows_labels[id] for id in train_ids}
    test_windows = {id: windows[id] for id in test_ids}
    test_labels = {id: windows_labels[id] for id in test_ids}

    x_train, y_train = combine_windows(train_windows, train_labels)
    x_test, y_test = combine_windows(test_windows, test_labels)

    return (x_train, y_train), (x_test, y_test)
