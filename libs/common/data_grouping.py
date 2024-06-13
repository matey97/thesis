import numpy as np


def merge_subjects_datasets(x, y, subjects):
    dim1 = x['s01'].shape[1]
    n_dims = len(x['s01'].shape)
    x_shape = (0, dim1) if n_dims == 2 else (0, dim1, 50)
    x_dataset = np.empty(x_shape, dtype='float64')
    y_dataset = np.empty((0, 5))
    for subject in subjects:
        x_dataset = np.append(x_dataset, x[subject], axis=0)
        y_dataset = np.append(y_dataset, y[subject], axis=0)
    
    return x_dataset, y_dataset


def generate_training_and_test_sets(x, y, train_subjects, test_subjects):
    x_train, y_train = merge_subjects_datasets(x, y, train_subjects)
    x_test, y_test = merge_subjects_datasets(x, y, test_subjects)
    
    return x_train, y_train, x_test, y_test