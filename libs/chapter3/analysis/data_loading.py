'''
Provides functions to load the obtained results.
'''


import os
import pandas as pd


def load_reports(path=os.path.join('data', 'chapter3', 'model-reports')):
    '''
    Loads the ML and DL reports generated from the ILOSO evaluation.
    
    Args:
        path (str): Root directory of the data.
        
    Returns:
        (`pandas.DataFrame`): DataFrame containing the generated reports.
    '''
    reports = []
    for file in os.listdir(path):
        if not file.endswith('.csv'):
            continue
        model_type, data_source = file.split('_')[:2]
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
        
        df['model'] = model_type
        df['source'] = data_source
        reports.append(df)

    return pd.concat(reports)


def load_best_significant(path):
    '''
    Loads a CSV file containing the number of best significant data sources/models for each combination of number of training
    subjects and models/data sources.

    Args:
        path (str): Path to the CSV file.
        
    Returns:
        (`pandas.DataFrame`): DataFrame containing specified CSV.
    '''
    return pd.read_csv(path, header=[0, 1], index_col=0)