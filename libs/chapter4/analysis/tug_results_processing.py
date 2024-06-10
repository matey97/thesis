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
Provides functions to process the raw results of the TUG test obtained by the developed system and the reference method.
'''


import numpy as np
import pandas as pd

from libs.common.utils import datetime_from, load_json

SW_RESULTS_FILE = '{0}_sw.json'
SP_RESULTS_FILE = '{0}_sp.json'
MAN_RESULTS_FILE = '{0}_results.csv'

PHASES = ['standing_up', 'first_walk', 'first_turn', 'second_walk', 'second_turn', 'sitting_down']
DURATION_AND_PHASES = ['duration'] + PHASES

def extract_system_results(subject, results_file):
    '''
    Extracts the TUG results generated by the system.

    Args:
        subject (str): ID of the subject whose results are extracted.
        results_file (str): Path of the file containing the results.

    Returns:
        (`pandas.DataFrame`): DataFrame containing the extracted results.
    '''

    def extract_phases_results(phases_results):
        results = {}
        for phase in PHASES:
            results[phase] = phases_results[phase.upper()]['duration'] if phase.upper() in phases_results else np.nan
        return results
            
    full_results = load_json(results_file)
    results = []
    
    for i, result in enumerate(full_results):
        results.append({
            'subject': subject,
            'duration': result['duration'] if result['duration'] != -1 else np.nan,
            **extract_phases_results(result['activitiesResults'])
        })
        
    return pd.DataFrame(results)


def extract_manual_results(subject, results_file):
    '''
    Extracts the TUG results generated by the reference system.

    Args:
        subject (str): ID of the subject whose results are extracted.
        results_file (str): Path of the file containing the results.

    Returns:
        (`pandas.DataFrame`): DataFrame containing the extracted results.
    '''

    def as_milliseconds(timedelta):
        return int(timedelta.total_seconds() * 1000)
    
    results_df = pd.read_csv(results_file, converters={'phase': lambda x: x.strip()})
    results_df.set_index(['execution', 'phase'], inplace=True)
    results = []
    
    for i, (index, df) in enumerate(results_df.groupby(level=0)):
        stand_start = datetime_from(df.loc[(index, 'stand_start')])
        stand_end = datetime_from(df.loc[(index, 'stand_end')])
        turn1_start = datetime_from(df.loc[(index, 'turn1_start')])
        turn1_end = datetime_from(df.loc[(index, 'turn1_end')])
        turn2_start = datetime_from(df.loc[(index, 'turn2_start')])
        turn2_end = datetime_from(df.loc[(index, 'turn2_end')])
        sit_end = datetime_from(df.loc[(index, 'sit_end')])
        results.append({
            'subject': subject,
            'duration': as_milliseconds(sit_end - stand_start),
            'standing_up': as_milliseconds(stand_end - stand_start),
            'first_walk': as_milliseconds(turn1_start - stand_end),
            'first_turn': as_milliseconds(turn1_end - turn1_start),
            'second_walk': as_milliseconds(turn2_start - turn1_end),
            'second_turn': as_milliseconds(turn2_end - turn2_start),
            'sitting_down': as_milliseconds(sit_end - turn2_end),
        })
        
    return pd.DataFrame(results)


def invalidate_executions(df, executions):
    '''
    Invalidates TUG executions not considered as failures by the system but that were wrongly executed due to external factors.

    Args:
        df (`pandas.DataFrame`): DataFrame containing the TUG test system's results.
        executions (dict): Dict indicating which executions (values) of which subject (key) have to be invalidated.
    
    Returns:
        (`pandas.DataFrame`): DataFrame with the specified samples invalidated.
    '''

    for subject_id, invalid_executions in executions.items():
        invalid_executions = np.array(invalid_executions) - 1
        index = df[df.subject == subject_id].index[invalid_executions]
        df.loc[index, 'duration'] = np.nan
    return df


def check_status(row):
    '''
    Checks the status of the system's generated results of a TUG execution. An execution is classified
    as `success` when all measures where obtained, `partial_success` when the duration of the test was obtained
    but the measure of one or more subphases is missing, `failure` when the duration of the test was not obtained. 

    Args: 
        row (`pandas.Series`): Results of a TUG execution.

    Returns:
        (str): The status of the execution. One of `success`, `partial_success` or `failure`.
    '''

    if pd.isna(row['duration']):
        print(f'Found failure: {row.name}')
        return 'failure'
    elif any(pd.isna(row)):
        print(f'Found partial success: {row.name}')
        return 'partial_success'
    else:
        return 'success'
    

def compute_errors_by_subject(systems_results, man_results):
    '''
    Computes the error in milliseconds between the measures obtained by the systems and the reference measures.

    Args:
        systems_results (dict): Dict containing the results (values) generated by both system's configurations: `C1` and `C2` (keys).
        man_results (`pandas.DataFrame`): DataFrame containing the reference measures.
    
    Returns:
        (`pandas.DataFrame`): DataFrame containing the errors between systems' and reference measures.
    '''

    dfs = []
    columns = DURATION_AND_PHASES
    for system_id, system_results in systems_results.items():
        
        errors = system_results.copy()
        errors[columns] = (system_results[columns] - man_results[columns])
        errors['system'] = system_id
        dfs.append(errors)
        
    return pd.concat(dfs, ignore_index=True)


def compute_rmse_by_subject(errors_df):
    '''
    Computes the intra-subject RMSE between the systems' and reference methods.
    
    Args:
        errors_df (`pandas.DataFrame`): DataFrame containing the error in ms of the system measures and the reference method for all subjects.

    Returns:
        (`pandas.DataFrame`): DataFrame containing the intra-subject RMSE for each subject, system and measure.
    '''
    def RMSE(values):
        values = values[~np.isnan(values)]
        return round(np.sqrt(np.sum(np.power(values, 2)) / len(values)))
    
    rmse_duration_df = errors_df[errors_df.status != 'failure'].groupby(['subject', 'system'])['duration'].apply(RMSE).to_frame('duration')
    rmse_phases_df = errors_df[errors_df.status == 'success'].groupby(['subject','system'])[PHASES].apply(RMSE)
    return pd.concat([rmse_duration_df, rmse_phases_df], axis=1)