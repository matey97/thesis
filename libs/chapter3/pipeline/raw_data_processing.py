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


import numpy as np
import pandas as pd

from libs.common.utils import datetime_from


def split(dataframe, types=['accelerometer', 'gyroscope']):
    splits = []
    for typ in types:
        splits.append(dataframe[dataframe.type == typ])
    return splits


def temporal_trim(df1, df2, exec_segments):
    def global_start(manual_start):
        start_df1 = df1['timestamp'].min()
        start_df2 = df2['timestamp'].min()
        return max(manual_start, start_df1, start_df2)
    
    def global_end(manual_end):
        end_df1 = df1['timestamp'].max()
        end_df2 = df2['timestamp'].max()
        return min(manual_end, end_df1, end_df2) # Not really needed, but good in case there is a mistake when inspecting the video
    
    def trim(df, start, end):
        def add_to_microsecond(time, amount):
            updated_microsecond = time.microsecond + amount
            if 0 <= updated_microsecond <= 999999:
                time = time.replace(microsecond=updated_microsecond)
            elif updated_microsecond < 0:
                time = time.replace(second=time.second - 1)
                time = time.replace(microsecond=updated_microsecond + 1000000)
            else:
                time = time.replace(second=time.second + 1)
                time = time.replace(microsecond=updated_microsecond - 1000000)
            return time
                
        start_thresh = add_to_microsecond(start, -1000) ## To align samples from sources, it is common a drift of 1 ms between them
        end_thresh = add_to_microsecond(end, 1000)
        return df[(df.timestamp >= start_thresh) & (df.timestamp <= end_thresh)]
    
    start = global_start(datetime_from(exec_segments.loc['start']))
    end = global_end(datetime_from(exec_segments.loc['end']))
    #print(start, end)
    
    return trim(df1, start, end), trim(df2, start, end)


def merge(df_acc, df_gyro):
    def copy_reset(df):
        return df.copy().reset_index(drop=True)
    
    def drop(df, columns):
        return df.drop(columns, axis=1)
    
    df_acc = drop(copy_reset(df_acc), ['type'])
    df_gyro = drop(copy_reset(df_gyro), ['type'])
    
    df_merged = df_acc.join(df_gyro, lsuffix="_acc", rsuffix="_gyro")
    return df_merged.dropna()


def add_labels(df, execution_segments):
    df_copy = df.copy()
    stand_start = datetime_from(execution_segments.loc['stand_start'])
    stand_end = datetime_from(execution_segments.loc['stand_end'])
    turn1_start = datetime_from(execution_segments.loc['turn1_start'])
    turn1_end = datetime_from(execution_segments.loc['turn1_end'])
    turn2_start = datetime_from(execution_segments.loc['turn2_start'])
    turn2_end = datetime_from(execution_segments.loc['turn2_end'])
    sit_end = datetime_from(execution_segments.loc['sit_end'])
    end = datetime_from(execution_segments.loc['end'])
    
    def compute_label(time):
        if time < stand_start or sit_end <= time <= end:
            return 'SIT'
        if stand_start <= time < stand_end:
            return 'STANDING'
        if turn1_start <= time < turn1_end or turn2_start <= time < turn2_end:
            return 'TURNING'
        if turn2_end < time < sit_end:
            return 'SITTING'
        if end <= time:
            return 'NOISE'
        return 'WALKING' 
    
    df_copy['gt'] = df_copy.timestamp_acc.apply(compute_label)
    df_copy = df_copy.drop(df_copy[df_copy['gt'] == 'NOISE'].index)
    
    return df_copy


SW_SCALING_RANGES = {
    "acc_max_res": 78.4532,
    "gyro_max_res": 34.906586
}

SP_SCALING_RANGES = {
    "acc_max_res": 78.45317840576172,
    "gyro_max_res": 17.45274543762207
}

def scale(df, source):
    scaling_ranges = SW_SCALING_RANGES if source == "sw" else SP_SCALING_RANGES
    
    def min_max_scale(value, maxVal, minVal, ran=[-1, 1]):
        return (ran[1] - ran[0]) * (value - minVal) / (maxVal - minVal) + ran[0]
    
    def scale_acc(value):
        scaling_range = scaling_ranges['acc_max_res']
        return min_max_scale(value, scaling_range, -scaling_range)
    
    def scale_gyro(value):
        scaling_range = scaling_ranges['gyro_max_res']
        return min_max_scale(value, scaling_range, -scaling_range)
    
    df_copy = df.copy()
    
    df_copy[['x_acc', 'y_acc', 'z_acc']] = df_copy[['x_acc', 'y_acc', 'z_acc']].apply(scale_acc)
    df_copy[['x_gyro', 'y_gyro', 'z_gyro']] = df_copy[['x_gyro', 'y_gyro', 'z_gyro']].apply(scale_gyro)
    return df_copy


def windows(data, window_size, step):
    r = np.arange(len(data))
    s = r[::step]
    z = list(zip(s, s + window_size))
    f = '{0[0]}:{0[1]}'.format
    g = lambda step : data.iloc[step[0]:step[1]]
    return pd.concat(map(g, z), keys=map(f, z))


def compute_best_class(df):
    ground_truth = df['label'].value_counts()
    best = ground_truth.index[0]
    return pd.Series([best], index=['CLASS'])


def count_data(data_collection):        
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
        'fused': {  
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
    return df.to_markdown()
