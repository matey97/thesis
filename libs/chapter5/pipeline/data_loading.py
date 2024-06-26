import os
import numpy as np


def load_data(directory):
    executions = {}
    labels = {}
    files = os.listdir(directory)
    files.sort()
    for file in files:
        if not file.endswith('.npy'):
            continue
    
        file_path = os.path.join(directory, file)
        file_name_parts = file.split('-')
        execution_id, file_type = file_name_parts[0], file_name_parts[1]
        data = np.load(file_path)
        if file_type.startswith('x'):
            executions[execution_id] = data
        else:
            labels[execution_id] = data
    return executions, labels