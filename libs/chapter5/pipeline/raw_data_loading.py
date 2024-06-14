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
import pandas as pd

DATASET_PATH = os.path.join('data', 'chapter5', 'preliminar-dataset', 'labelled', '{0}')

def load_labelled_data(path):
    executions = {}
    labels = {}
    files = os.listdir(path)
    files.sort()
    for file in files:
        if not file.endswith('.npy'):
            continue
    
        file_path = os.path.join(path, file)
        file_name_parts = file.split('-')
        execution_id, file_type = file_name_parts[0], file_name_parts[1]
        data = np.load(file_path, allow_pickle=True)
        if file_type.startswith('x'):
            executions[execution_id] = data
        else:
            labels[execution_id] = data
    return executions, labels


LABELS = ['SEATED_RX','STANDING_UP_RX','WALKING_TX','TURN_TX','SITTING_DOWN_TX', 'SEATED_TX', 'STANDING_UP_TX','WALKING_RX','TURN_RX','SITTING_DOWN_RX']

def count_samples(datasets_labels):
    dfs = []
    for i, dataset_labels in enumerate(datasets_labels):
        recount = dict(zip(LABELS, [0] * len(LABELS)))

        for labels in dataset_labels.values():
            unique, counts = np.unique(labels, return_counts=True)
            count = dict(zip(unique, counts))
            for key, value in count.items():
                recount[key] += value
        dfs.append(pd.DataFrame(recount.values(), index=recount.keys(), columns=[f'D{i+1}']))
    df = pd.concat(dfs, axis=1)
    df.loc['Total',:] = df.sum(axis=0)
    return df