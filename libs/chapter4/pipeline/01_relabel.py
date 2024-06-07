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

"""Data relabelling script.

Relabels the windowed data by replacing the TURNING and SITTING_DOWN labels by the TURN_TO_SIT label. Note that
only the TURNING activities inmediately before the SITTING_DOWN activity are replaced by TURN_TO_SIT.

**Example**:

    $ python 01_relabel.py --input_data_path <PATH_OF_WINDOWED_DATA> --output_data_path <PATH_TO_STORE_RELABELLED_DATA>
"""


import argparse
import numpy as np
import os
import pandas as pd
import sys

sys.path.append("../../..")

from alive_progress import alive_bar
from libs.common.data_loading import load_subjects_data


def relabel(gt):
    relabelled_gt = {}
    for subject, data in gt.items():
        data_copy = np.copy(data)
        changes = np.where(np.roll(data,1) != data)[0]

        for i, change in enumerate(changes):
            if change == 0:
                continue
            if data_copy[change] == 'SITTING_DOWN':
                if i+1 != len(changes):
                    data_copy[changes[i-1]:changes[i+1]] = 'TURN_TO_SIT'
                else:
                    data_copy[changes[i-1]:] = 'TURN_TO_SIT'
        relabelled_gt[subject] = data_copy
    return relabelled_gt


def count_data(data_collection):        
    recount = {
        'sp': {
            'SEATED': 0,
            'STANDING_UP': 0,
            'WALKING': 0,
            'TURNING': 0,
            'TURN_TO_SIT': 0
        },
        'sw': {  
            'SEATED': 0,
            'STANDING_UP': 0,
            'WALKING': 0,
            'TURNING': 0,
            'TURN_TO_SIT': 0
        }
    }

    for source, data in data_collection.items(): 
        for subject, subject_data in data.items():    
            unique, counts = np.unique(subject_data, return_counts=True)
            value_counts = dict(zip(unique, counts))
            recount[source]['SEATED'] += value_counts['SEATED']
            recount[source]['STANDING_UP'] += value_counts['STANDING_UP']
            recount[source]['WALKING'] += value_counts['WALKING']
            recount[source]['TURNING'] += value_counts['TURNING']
            recount[source]['TURN_TO_SIT'] += value_counts['TURN_TO_SIT']
        
    df = pd.DataFrame(recount).transpose()
    df['TOTAL'] = df.sum(axis=1)
    return df.to_markdown()



def store_windowed_data(windowed_data, ground_truth, path):
    def store_as_npy(path, data):
        with open(path, 'wb') as f:
            np.save(f, np.array(data)) 
            
    with alive_bar(len(windowed_data), title=f'Storing windowed data in {path}', force_tty=True, monitor='[{percent:.0%}]') as progress_bar:
        for source, subjects_data in windowed_data.items():
            for subject, data in subjects_data.items():
                subject_path = os.path.join(path, subject)
                if not os.path.exists(subject_path):
                    os.makedirs(subject_path)

                store_as_npy(os.path.join(subject_path, f'{subject}_{source}.npy'), data)
                store_as_npy(os.path.join(subject_path, f'{subject}_{source}_gt.npy'), ground_truth[source][subject])
            progress_bar()  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path', help='Path of input windowed data', type=str, required=True)
    parser.add_argument('--output_data_path', help='Path to store the relabelled windowed data', type=str, required=True)
    args = parser.parse_args()
    
    sp_windowed_data, sp_gt = load_subjects_data(args.input_data_path, 'sp', True)
    sw_windowed_data, sw_gt = load_subjects_data(args.input_data_path, 'sw', True)

    sp_gt_relabelled = relabel(sp_gt)
    sw_gt_relabelled = relabel(sw_gt)

    windowed_data = {
        'sp': sp_windowed_data,
        'sw': sw_windowed_data
    }     

    gt = {
        'sp': sp_gt_relabelled,
        'sw': sw_gt_relabelled,
    }

    print(count_data(gt))

    store_windowed_data(windowed_data, gt, args.output_data_path)
