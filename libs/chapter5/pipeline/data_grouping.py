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
