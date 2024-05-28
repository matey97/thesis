import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from .utils import list_folder, list_subjects_folders


def load_subjects_data(path, source, use_raw_data):
    x = {}
    y = {}
    
    for subject in list_subjects_folders(path):
        subject_dir = os.path.join(path, subject)
        
        x[subject] = np.load(os.path.join(subject_dir, f'{subject}_{source}{"_features" if not use_raw_data else ""}.npy'))
        y[subject] = np.load(os.path.join(subject_dir, f'{subject}_{source}_gt.npy'))
    
    return x, y


def ground_truth_to_categorical(y, mapping):
    y_copy = y.copy()
    for subject, gt in y_copy.items():
        mapped_gt = list(map(lambda i : mapping[i], gt))
        y_copy[subject] = to_categorical(mapped_gt, len(mapping))
        
    return y_copy


def load_data(path, source, use_raw_data, gt_mapping):
    x, y = load_subjects_data(path, source, use_raw_data)
    y = ground_truth_to_categorical(y, gt_mapping)
    
    return x, y