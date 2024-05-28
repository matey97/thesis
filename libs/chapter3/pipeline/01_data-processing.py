import argparse
import os
import numpy as np
import sys

sys.path.append("../../..")

from alive_progress import alive_bar
from libs.chapter2.data_loading import load_data
from libs.chapter3.pipeline.feature_extraction import apply_feature_extraction
from libs.chapter3.pipeline.data_fusion import fuse_data
from libs.chapter3.pipeline.raw_data_processing import scale, windows, compute_best_class, count_data


WINDOW_SIZE = 50
STEP_SIZE = WINDOW_SIZE // 2


def clean_raw_data(raw_data):
    clean_data = {}

    with alive_bar(len(raw_data), title=f'Data cleanning', force_tty=True, monitor='[{percent:.0%}]') as progress_bar:
        for desc, data in raw_data.items():
            _, source = desc.rsplit('_', 1)
            clean_data[desc] = scale(data, source)
            progress_bar()
            
    return clean_data


def fuse_sources(clean_data):
    fused_clean_data = {}
    for execution, data in clean_data.items():
        *exec_id, device = execution.split('_')
        exec_id = '_'.join(exec_id)

        if not exec_id in fused_clean_data:
            fused_clean_data[exec_id] = {}
        fused_clean_data[exec_id][device] = data

    for execution, data in fused_clean_data.items():
        fused = fuse_data(data['sp'], data['sw'])
        clean_data[f'{execution}_fused'] = fused
    
    return clean_data


def get_windowed_data(clean_data, window_size, step_size): 
    windowed_data = {}
    gt = {}
    
    with alive_bar(len(clean_data), title=f'Data windowing', force_tty=True, monitor='[{percent:.0%}]') as progress_bar:
        for desc, data in clean_data.items():
            desc_components = desc.split('_')
            subject_sensor_desc = f'{desc_components[0]}_{desc_components[2]}'

            windowed_df = windows(data, window_size, step_size)
            desc_instances = []
            desc_gt = []

            columns = ['x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro'] if desc_components[2] != 'fused' else ['x_acc_sp', 'y_acc_sp', 'z_acc_sp', 'x_gyro_sp', 'y_gyro_sp', 'z_gyro_sp',
                         'x_acc_sw', 'y_acc_sw', 'z_acc_sw', 'x_gyro_sw', 'y_gyro_sw', 'z_gyro_sw']

            for i in range(0, data.shape[0], step_size):
                window = windowed_df.loc["{0}:{1}".format(i, i+window_size)]
                values = window[columns].transpose()
                groundtruth = compute_best_class(window)
                if (values.shape[1] != window_size):
                    break
                desc_instances.append(values.values.tolist())
                desc_gt.append(groundtruth.values[0])

            if subject_sensor_desc in windowed_data:
                windowed_data[subject_sensor_desc] += desc_instances
                gt[subject_sensor_desc] += desc_gt
            else:
                windowed_data[subject_sensor_desc] = desc_instances
                gt[subject_sensor_desc] = desc_gt
                
            progress_bar()
            
    return windowed_data, gt


def extract_features(windowed_data):
    featured_data = {}
    with alive_bar(len(windowed_data.items()), title=f'Feature extraction', force_tty=True, monitor='[{percent:.0%}]') as progress_bar:
        for subject, windows in windowed_data.items():
            data_type = subject.split('_')[-1]
            features = []
            for window in windows:
                window = np.array(window)
                if data_type != 'fused':
                    features.append(apply_feature_extraction(window))
                else:
                    part_a = window[:6,:]
                    part_b = window[6:,:]

                    features_a = apply_feature_extraction(part_a)
                    features_b = apply_feature_extraction(part_b)
                    features.append(np.concatenate((features_a, features_b)))
            
            featured_data[subject] = np.array(features)
            progress_bar()
    return featured_data
        
        
def store_windowed_data(windowed_data, features, ground_truth, path):
    def store_as_npy(path, data):
        with open(path, 'wb') as f:
            np.save(f, np.array(data)) 
            
    with alive_bar(len(windowed_data), title=f'Storing windowed data in {path}', force_tty=True, monitor='[{percent:.0%}]') as progress_bar:
        for desc, data in windowed_data.items():
            subject = desc.split('_')[0]
            subject_path = os.path.join(path, subject)
            if not os.path.exists(subject_path):
                os.makedirs(subject_path)

            store_as_npy(os.path.join(subject_path, f'{desc}.npy'), data)
            store_as_npy(os.path.join(subject_path, f'{desc}_features.npy'), features[desc])
            store_as_npy(os.path.join(subject_path, f'{desc}_gt.npy'), ground_truth[desc])
            progress_bar()  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path', help='Path of input data', type=str, required=True)
    parser.add_argument('--windowed_data_path', help='Path to store windowed data', type=str, required=True)
    args = parser.parse_args()
    
    raw_data = load_data(args.input_data_path)
    
    clean_data = clean_raw_data(raw_data)
    clean_fused_data = fuse_sources(clean_data)

    print('\nClean data:')
    print(count_data(clean_fused_data), '\n')
    print('\n')

    
    windowed_data, gt = get_windowed_data(clean_fused_data, WINDOW_SIZE, STEP_SIZE)
    print('\nWindowed data:')
    print(count_data(gt), '\n')
        
    features = extract_features(windowed_data)
    store_windowed_data(windowed_data, features, gt, args.windowed_data_path)