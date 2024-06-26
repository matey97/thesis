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