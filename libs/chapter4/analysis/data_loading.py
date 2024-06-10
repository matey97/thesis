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

'''
Provides functions to load the obtained results.
'''

import json
import os

import pandas as pd

from libs.chapter4.analysis.tug_results_processing import extract_system_results, extract_manual_results, SP_RESULTS_FILE, SW_RESULTS_FILE, MAN_RESULTS_FILE
from libs.common.utils import list_subjects_folders 


def load_reports(reports_path=os.path.join('data','chapter4', 'splitting-approach', 'reports.json')):
    '''
    Loads the DL reports generated from the splitting approach evaluation.
    
    Args:
        reports_path (str): Root directory of the data.
        
    Returns:
        (`dict`): Dictionary containing the generated reports.
    '''

    with open(reports_path, 'r') as file:
        return json.load(file) 
    

def load_subjects_info(path=os.path.join('data', 'chapter4', 'system-results', 'subjects.csv')):
    '''
    Loads a CSV file containing the information regarding the participants in the evaluation of the system.

    Args:
        path (str): Path to the CSV file.

    Returns:
        (`pandas.DataFrame`): DataFrame with the information of the participants.
        (str): Formatted string with statistics regarding participants' age (e.g., range, mean, std).
        (str): Formatted string with statistics regarding participants' gender (e.g., male/female ratio).
    '''

    subjects_info = pd.read_csv(path)
    age = subjects_info['Age']
    age_info = f'Age info: min={age.min():.2f}, max={age.max():.2f}, mean={age.mean():.2f}, std={age.std():.2f}'
    gender = subjects_info['Gender'].value_counts()
    male_count = gender['M']
    female_count = gender['F']
    gender_info = f'Gender info: male={male_count} ({male_count/(male_count+female_count) * 100}), female={gender["F"]} ({female_count/(male_count+female_count) * 100})'
    return subjects_info, age_info, gender_info


def load_experiment_results(path=os.path.join('data', 'chapter4', 'system-results')):
    '''
    Loads the results obtained in the experiment by each participant.
    
    Args:
        path (str): Directory containing the results of the experiment.

    Returns:
        (`pandas.DataFrame`): DataFrame with the loaded experiment results.
    '''

    subjects = list_subjects_folders(path)
    sw_results = []
    sp_results = []
    man_results = []
    
    for subject in subjects:      
        subject_dir = os.path.join(path, subject)
        sw_results_file = os.path.join(subject_dir, SW_RESULTS_FILE.format(subject))
        sp_results_file = os.path.join(subject_dir, SP_RESULTS_FILE.format(subject))
        man_results_file = os.path.join(subject_dir, MAN_RESULTS_FILE.format(subject))

        sw_results.append(extract_system_results(subject, sw_results_file))
        sp_results.append(extract_system_results(subject, sp_results_file))

        man_results.append(extract_manual_results(subject, man_results_file))
    
    return pd.concat(sw_results, axis=0, ignore_index=True), pd.concat(sp_results, axis=0, ignore_index=True), pd.concat(man_results, axis=0, ignore_index=True)


      