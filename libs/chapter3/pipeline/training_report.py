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


import pandas as pd
        
        
def report_to_dataframe(test_subject, n, i, report):
    reports_tidy = []

    for metric, value in report.items():
        if isinstance(value, dict):
            for prop in value:
                reports_tidy.append([test_subject, n, i, metric, prop, value[prop]])
        else:
            reports_tidy.append([test_subject, n, i, 'model', metric, value]) 
            
    return pd.DataFrame(reports_tidy, columns=['test_subject', 'n', 'i', 'target', 'metric', 'value'])


def report_writer(file_path):
    first = False
    def writer(test_subject, n, i, report):
        nonlocal first
        df = report_to_dataframe(test_subject, n, i, report)
        if first:
            df.to_csv(file_path, index=False, mode='w')
            first = False
        else:
            df.to_csv(file_path, index=False, header=False, mode='a')
            
    return writer