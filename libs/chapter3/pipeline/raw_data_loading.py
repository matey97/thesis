import os
import json
import pandas as pd
from libs.common.utils import list_folder, list_subjects_folders


def load_json(data_file_path):
    with open(data_file_path, 'r') as file:
        return json.load(file)

    
def records_to_dataframe(records_file):
    tidy_samples = []

    for records in records_file:
        for sample in records['samples']:
            sample_type = records['type']
            tidy_samples.append({
                "type": sample_type,
                "timestamp": sample['timestamp'],
                "x": sample['x'],
                "y": sample['y'],
                "z": sample['z']
            })
    df = pd.DataFrame(tidy_samples)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def load_raw_data(raw_data_path):
    subjects = list_subjects_folders(raw_data_path)
    raw_data = {}
    segments_dfs = []

    for subject in subjects:        
        subject_dir = os.path.join(raw_data_path, subject)
        subject_files = list_folder(subject_dir)

        for file in subject_files:
            file_path = os.path.join(subject_dir, file)
            file_desc = file.split('.')[0]
            if not os.path.isfile(file_path) or not file_path.endswith('.json'):
                continue

            records_file = load_json(file_path)
            raw_data[file_desc] = records_to_dataframe(records_file)
            
        subject_segments_file = os.path.join(subject_dir, f'{subject}_segments.txt')
        subject_segments = pd.read_csv(subject_segments_file, converters={'phase': lambda x: x.strip()})
        segments_dfs.append(subject_segments)
           
    segments = pd.concat(segments_dfs, axis=0, ignore_index=True)
    segments.set_index(['execution', 'phase'], inplace=True)
    
    return raw_data, segments


def load_subjects_info(path):
    subjects_info = pd.read_csv(path)
    age = subjects_info['afe']
    age_info = f'Age info: min={age.min():.2f}, max={age.max():.2f}, mean={age.mean():.2f}, std={age.std():.2f}'
    gender = subjects_info['gender'].value_counts()
    male_count = gender['M']
    female_count = gender['F']
    gender_info = f'Gender info: male={male_count} ({male_count/(male_count+female_count) * 100}), female={gender["F"]} ({female_count/(male_count+female_count) * 100})'
    return subjects_info, age_info, gender_info