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


import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from .utils import list_subjects_folders


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