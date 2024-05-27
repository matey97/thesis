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