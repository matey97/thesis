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
Provides functions to process the battery consumption results
'''

SP_CAPACITY_MAH = 5160
SW_CAPACITY_MAH = 580


def _mA_consumption_from_percentage(x, percentage_attr):
    percentage = x[percentage_attr]
    return percentage * SP_CAPACITY_MAH / 100 if x.name[1].startswith('sp') else percentage * SW_CAPACITY_MAH / 100


def mean_consumption_per_device(battery_df):
    '''
    Computes the mean battery consumption per device and system. The battery consumption is reported in mA and the corresponding
    ratio in terms of the battery's total capacity.

    Args:
        battery_df (`pandas.DataFrame`):

    Returns:
        (`pandas.DataFrame`):
    '''
    battery_by_device_df = battery_df[['configuration','device', 'ratio']].groupby(['configuration','device']).mean()
    battery_by_device_df['ratio_ma'] = battery_by_device_df.apply(lambda x: _mA_consumption_from_percentage(x, 'ratio'), axis=1)
    battery_by_device_df = battery_by_device_df.rename(columns={'ratio': 'consumption (%)', 'ratio_ma': 'consumption (mA)'})
    return battery_by_device_df.sort_index(level=1, ascending=False)
