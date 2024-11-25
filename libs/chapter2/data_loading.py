'''
Provides functions to load the collected data and associated metadata files.
'''

import os
import pandas as pd


def _list_folder(path):
    
    items = os.listdir(path)
    items.sort()
    return items


def _list_subjects_folders(path):
    subjects = _list_folder(path)
    return list(filter(lambda name : os.path.isdir(os.path.join(path, name)) and name.startswith('s'), subjects))


def load_data(path=os.path.join('data','chapter2')):
    '''
    Loads the accelerometer and gyroscope data for each execution.
    
    Args:
        path (str): Root directory of the data.
        
    Returns:
        data (dict): Dict containing pandas dataframes with the accelerometer and gyroscope data for each execution.
    '''
    
    subjects = _list_subjects_folders(path)
    data = {}

    for subject in subjects:        
        subject_dir = os.path.join(path, subject)
        subject_files = _list_folder(subject_dir)

        for file in subject_files:
            file_path = os.path.join(subject_dir, file)
            file_desc = file.split('.')[0]
            if not os.path.isfile(file_path) or not file_path.endswith('.csv'):
                continue

            data[file_desc] = pd.read_csv(file_path)
    
    return data


def load_subjects_info(path=os.path.join('data', 'chapter2', 'subjects_info.csv')):
    '''
    Loads the 'subjects_info.csv' file containing information about the subjects (age, gender, executions)
    
    Args:
        path (str): Path of the file. 
        
    Returns:
        subjects_info (`pandas.DataFrame`): DataFrame with the contents of the file
    '''
    subjects_info = pd.read_csv(path)
    return subjects_info


def load_executions_info(path=os.path.join('data', 'chapter2', 'executions_info.csv')):
    '''
    Loads the 'executions_info.csv' file containing information about the executions (id, phone orientation, turns direction)
    
    Args:
        path (str): Path of the file. 
        
    Returns:
        executions_info (`pandas.DataFrame`): DataFrame with the contents of the file
    '''
    executions_info = pd.read_csv(path)
    return executions_info