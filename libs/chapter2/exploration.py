'''
Provides functions to compute statistics regarding subjects and collected data.
'''

import pandas as pd
    

def subjects_age_range(subjects_info):
    '''
    Computes age range statisitcs from the subjects of the data collection.
    
    Args:
        subjects_info (`pandas.DataFrame`): DataFrame with the information of the subjects. See: `utils.data_loading.load_subjects_info()`.
    
    Returns:
        age_stats (`pandas.DataFrame`): DataFrame with age range statistics.
    '''
    
    return subjects_info['age'].describe().to_frame().transpose()


def subjects_age_range_by_gender(subjects_info):
    '''
    Computes age range statisitcs grouped by gender from the subjects of the data collection.
    
    Args:
        subjects_info (`pandas.DataFrame`): DataFrame with the information of the subjects. See: `utils.data_loading.load_subjects_info()`.
    
    Returns:
        gender_stats (`pandas.DataFrame`): DataFrame with age range statistics grouped by gender.
    '''
    
    return subjects_info[['age', 'gender']].groupby(['gender']).describe()


def executions_by_gender(subjects_info):
    '''
    Counts the number of executions grouped by gender.
    
    Args:
        subjects_info (`pandas.DataFrame`): DataFrame with the information of the subjects. See: `utils.data_loading.load_subjects_info()`.
    
    Returns:
        execution_stats (`pandas.DataFrame`): DataFrame with executions count grouped by gender.
    '''
    
    df = subjects_info[['executions', 'gender']].groupby(['gender']).sum().transpose()
    df['Total'] = df.sum(axis=1)
    return df


def count_samples(data_collection):   
    '''
    Counts the number of collected samples for each activity and device.
    
    Args:
        data_collection (dict): Collected data. Use `utils.data_loading.load_data()` to load the collected data.
        
    Returns:
        counts (`pandas.DataFrame`): DataFrame with the count of collected samples.
    
    '''
    
    counts = {
        'sp': {
            'SEATED': 0,
            'STANDING_UP': 0,
            'WALKING': 0,
            'TURNING': 0,
            'SITTING_DOWN': 0
        },
        'sw': {  
            'SEATED': 0,
            'STANDING_UP': 0,
            'WALKING': 0,
            'TURNING': 0,
            'SITTING_DOWN': 0
        },
    }

    for data_id, data in data_collection.items(): 
        source = data_id.split('_')[-1]
        if not isinstance(data, pd.DataFrame):
            counts[source]['SEATED'] += data.count('SEATED')
            counts[source]['STANDING_UP'] += data.count('STANDING_UP')
            counts[source]['WALKING'] += data.count('WALKING')
            counts[source]['TURNING'] += data.count('TURNING')
            counts[source]['SITTING_DOWN'] += data.count('SITTING_DOWN')
            continue
            
        count = data['label'].value_counts()
        counts[source]['SEATED'] += count['SEATED']
        counts[source]['STANDING_UP'] += count['STANDING_UP']
        counts[source]['WALKING'] += count['WALKING']
        counts[source]['TURNING'] += count['TURNING']
        counts[source]['SITTING_DOWN'] += count['SITTING_DOWN']
        
    df = pd.DataFrame(counts).transpose()
    df['TOTAL'] = df.sum(axis=1)
    return df