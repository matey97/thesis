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