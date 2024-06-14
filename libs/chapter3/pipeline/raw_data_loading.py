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
import pandas as pd

from libs.common.utils import list_folder, list_subjects_folders, load_json

    
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